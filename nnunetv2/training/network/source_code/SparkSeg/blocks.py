"""SparkSeg building blocks (paper Fig. 1b–c).

TAE, the scalar gate, PGTS, the block-attention kernel, WinMHSA3D, MSBA, PriorHead,
the decoder attention gate, the SparkSeg Block and the U-Net decoder with AG.
These are the definitions;
:mod:`nnunetv2.training.network.source_code.SparkSeg.model` assembles them.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    _init_sigmoid_gate_closed, align_spatial3d, apply_topk_jitter, delta_energy_per_sample, topk_coords_from_map,
)

class TAE(nn.Module):
    """TAE — tri-axial enhancement (paper Fig. 1b, Eq. 1).

    ``Δ_orient = mean(DW_D, DW_H, DW_W)(F_i)``, returned as an additive residual:
    ``forward(f) -> (f + Δ, Δ)``.

    * The three depthwise branches (5×1×1 ∥ 1×5×1 ∥ 1×1×5) run in parallel and are
      averaged — they are not stacked into one 3-stage convolution.
    * ``Δ`` also feeds the residual-energy probe ``e = delta_energy(Δ)`` used by the
      prior; it is never a segmentation output on its own.
    * Cold start: weights and biases are zero, so ``Δ ≡ 0`` and ``h ≡ f``; inserting
      TAE does not perturb the host CNN.
    """

    def __init__(self, dim: int, kernel: int = 5) -> None:
        super().__init__()
        k = int(kernel)
        p = k // 2
        # one depthwise kernel per axis: mixes along that axis only, channel-wise
        self.dw_d = nn.Conv3d(dim, dim, (k, 1, 1), padding=(p, 0, 0), groups=dim)
        self.dw_h = nn.Conv3d(dim, dim, (1, k, 1), padding=(0, p, 0), groups=dim)
        self.dw_w = nn.Conv3d(dim, dim, (1, 1, k), padding=(0, 0, p), groups=dim)
        for m in (self.dw_d, self.dw_h, self.dw_w):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # parallel axes -> mean -> residual
        delta = (self.dw_d(x) + self.dw_h(x) + self.dw_w(x)) / 3.0
        return x + delta, delta


class _ScalarGate3D(nn.Module):
    """Global scalar gate σ: GAP -> 1×1 MLP -> Sigmoid, broadcast over the volume.

    Used as ``ba_gate``: ``out = base + σ · (cand − base)``.
    Cold start: the default bias −2.5 gives σ ≈ 0.07 (a GateOpen variant may use −1.5
    for σ ≈ 0.18).
    """

    def __init__(self, channels: int, bias: float = -2.5):
        super().__init__()
        h = max(int(channels) // 4, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, h, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(h, 1, 1),
            nn.Sigmoid(),
        )
        _init_sigmoid_gate_closed(self.net[-2], bias=float(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PGTS(nn.Module):
    """PGTS — prior-gated token selection (paper Fig. 1b, Eq. 4–6).

    Scores importance, gates it with the effective prior P_eff and keeps the hard Top-K
    anchors (``K_s = min(512, max(32, 8·2^(S−s)))`` -> 128/64/32/32 for Stages 2–5).

    Pipeline:
      1) ``importance_raw = σ(CNN(h))`` — a light 3×3 -> 1×1 scorer
      2) with ``spatial_prior`` (normally P_eff):
           ``prior_gate = max(prior[:, 1:])``      # foreground only; ch0 = background
           ``importance = prior_gate × (ε + (1−ε)·raw)``   # prior leads, CNN refines; ε = 0.1
      3) ``x_refined = h + 1×1(h ⊙ importance)`` — differentiable refine, zero-init -> identity
      4) ``idx = topk(importance)`` — indices are not differentiable; the caller may add jitter

    Returns ``(importance, idx[B, K, 3], x_refined)``.
    """

    def __init__(self, in_channels: int, target_num_tokens: int = 64):
        super().__init__()
        self.target_num_tokens = int(target_num_tokens)
        mid = max(in_channels // 4, 8)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid, 3, padding=1),
            nn.InstanceNorm3d(mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
        )
        self.refine = nn.Conv3d(in_channels, in_channels, 1, bias=True)
        nn.init.zeros_(self.refine.weight)
        if self.refine.bias is not None:
            nn.init.zeros_(self.refine.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        compute_token_indices: bool = True,
        spatial_prior: Optional[torch.Tensor] = None,
        spatial_prior_eps: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, d, h, w = x.shape
        importance_map = torch.sigmoid(self.conv(x))
        if spatial_prior is not None:
            prior = spatial_prior
            if prior.ndim == 4:
                prior = prior.unsqueeze(1)
            if prior.shape[2:] != (d, h, w):
                prior = F.interpolate(
                    prior, size=(d, h, w), mode="trilinear", align_corners=False
                )
            # multi-class prior -> foreground max as the soft gate (ch0 = background)
            # the prior decides *where*; the raw score refines *which* anchor inside it
            if prior.shape[1] > 1:
                prior_gate = prior[:, 1:].max(dim=1, keepdim=True).values
            else:
                prior_gate = prior
            prior_gate = prior_gate.clamp(0.0, 1.0)
            eps = min(max(float(spatial_prior_eps), 0.0), 1.0)
            importance_map = prior_gate * (eps + (1.0 - eps) * importance_map)
        x_refined = x + self.refine(x * importance_map.expand_as(x))
        if not compute_token_indices:
            empty = torch.zeros(b, 0, 3, dtype=torch.long, device=x.device)
            return importance_map, empty, x_refined
        k = min(self.target_num_tokens, d * h * w)
        idx = topk_coords_from_map(importance_map, k)
        return importance_map, idx, x_refined


class _SourceWiseAttention(nn.Module):
    """Block-level cross-attention over the sparse anchors (not standard self-attention).

    Tensor layout::

        Q  = partial                  [B, K, C]      (current Top-K features)
        KV = stack(hist_blocks + [partial])  [S, B, K, C]   (S = history sources + 1)
        softmax(dim=0): every query normalises over the *source* dimension (hist ∪ self)

    Design notes: the partial token is both query and key/value on purpose (an anchor may
    keep itself), and with S = 1 (no history) the softmax degenerates to 1, i.e. identity.
    """

    def __init__(self, hidden_size: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must divide num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = float(attn_drop)
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, blocks: List[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
        mem = torch.stack(blocks + [partial], dim=0)
        s, b, t, d = mem.shape
        hh, dh = self.num_heads, self.head_dim
        kv = self.norm_kv(mem)
        q = self.q_proj(self.norm_q(partial)).view(b, t, hh, dh)
        k = self.k_proj(kv).view(s, b, t, hh, dh)
        v = self.v_proj(kv).view(s, b, t, hh, dh)
        attn = (torch.einsum("bthd,sbthd->sbth", q, k) * self.scale).softmax(0)
        if self.training and self.attn_drop > 0:
            attn = F.dropout(attn, p=self.attn_drop)
        return torch.einsum("sbth,sbthd->bthd", attn, v).reshape(b, t, d)


class WinMHSA3D(nn.Module):
    """WinMHSA3D — windowed multi-head self-attention (paper Fig. 1b).

    4×4×4 windows, 4 heads; a regular pass and a half-window-shifted pass (shift = 2)
    produce Δ_regular and Δ_shift, which are summed and scaled by one shared σ_win gate.
    Complementary to the sparse MSBA read: this branch is dense *inside* windows, and
    windows stay isolated from each other by default.

    * ``shift_size > 0`` rolls first and pads afterwards, so zero padding never enters a
      valid window; the caller runs the dual pass (``SparkSegBlock`` sums Δ0 + Δs without
      halving, and only for stages at or below ``window_shift_min_stage``).
    * Cold start: the final ``proj`` is zero-initialised, so ``forward(x) ≈ x`` before
      the σ_win gate.
    * Non-divisible spatial sizes are padded and cropped back; the shifted pass carries
      no attention mask.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 4,
        window_size: int = 4,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        heads = max(1, int(num_heads))
        if dim % heads != 0:
            raise ValueError(f"WinMHSA3D: dim {dim} not divisible by heads {heads}")
        self.dim = dim
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.window_size = max(1, int(window_size))
        self.attn_drop = float(attn_drop)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        _, _, d, h, w = x.shape
        ws = self.window_size
        pd = (ws - d % ws) % ws
        ph = (ws - h % ws) % ws
        pw = (ws - w % ws) % ws
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x, (d, h, w)

    def _window_attn(self, xp: torch.Tensor) -> torch.Tensor:
        """xp: padded B,C,D,H,W → same shape projected delta (on padded grid)."""
        b, c, d, h, w = xp.shape
        ws = self.window_size
        n_d, n_h, n_w = d // ws, h // ws, w // ws
        windows = (
            xp.view(b, c, n_d, ws, n_h, ws, n_w, ws)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .contiguous()
            .view(-1, ws * ws * ws, c)
        )
        n_tok = windows.shape[1]
        hh, dh = self.num_heads, self.head_dim
        qkv = self.qkv(self.norm(windows)).view(-1, n_tok, 3, hh, dh)
        q, k, v = qkv.unbind(dim=2)
        attn = (torch.einsum("bthd,bshd->bhts", q, k) * self.scale).softmax(dim=-1)
        if self.training and self.attn_drop > 0:
            attn = F.dropout(attn, p=self.attn_drop)
        out = torch.einsum("bhts,bshd->bthd", attn, v).reshape(-1, n_tok, c)
        out = self.proj(out)
        return (
            out.view(b, n_d, n_h, n_w, ws, ws, ws, c)
            .permute(0, 7, 1, 4, 2, 5, 3, 6)
            .contiguous()
            .view(b, c, d, h, w)
        )

    def forward(self, x: torch.Tensor, shift_size: int = 0) -> torch.Tensor:
        """Return a candidate ``≈ x`` at cold start.

        Shift order is ``Roll -> Pad -> Attn -> Crop -> UnRoll``; padding before rolling
        would drag zero padding into valid windows.
        """
        assert x.shape[1] == self.dim
        sh = int(shift_size) % self.window_size
        work = x
        if sh:
            work = torch.roll(work, shifts=(-sh, -sh, -sh), dims=(2, 3, 4))
        xp, (d_o, h_o, w_o) = self._pad(work)
        out = self._window_attn(xp)
        if out.shape[2:] != (d_o, h_o, w_o):
            out = out[:, :, :d_o, :h_o, :w_o]
        if sh:
            out = torch.roll(out, shifts=(sh, sh, sh), dims=(2, 3, 4))
        return x + out


class MSBA(nn.Module):
    """MSBA — multi-source sparse block attention with sparse write-back (paper Fig. 1b).

    Gathers only at the Top-K anchors, attends over hist/global/self with a softmax along
    the source dimension (cost O(K·S_src) instead of O(N²)) and scatters the update back
    only there: ``F_ba = F_base ⊕ σ_ba · Δ_tokens``.

    Steps:
      1) sample current / hist features at ``token_indices`` (hist is 1×1-projected to
         ``dim`` first, then aligned to the current grid)
      2) optionally add a global-reference dictionary (``global_kv``) as another source
      3) ``_SourceWiseAttention`` produces the token update; an optional per-token gate blends it
      4) scatter ``(out − partial)`` at the Top-K coordinates only; other voxels are untouched

    Gradients flow through features and gates, never through the hard Top-K indices; an
    empty index set or an empty KV returns the input unchanged.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float,
        hist_in_channels: Sequence[int],
        *,
        gated_writeback: bool = True,
    ):
        super().__init__()
        # history sources may differ in width from dim (cascaded stages) -> project first
        self.hist_projs = nn.ModuleList(
            [nn.Conv3d(int(c), dim, 1, bias=False) for c in hist_in_channels]
        )
        self.attn = _SourceWiseAttention(dim, num_heads, attn_drop)
        if gated_writeback:
            h = max(dim // 2, 8)
            # per-token gate: [partial ∥ out] -> σ, then partial + σ·(out − partial)
            self.write_gate = nn.Sequential(
                nn.Linear(dim * 2, h), nn.GELU(), nn.Linear(h, 1), nn.Sigmoid()
            )
            # cold start σ ≈ 0.075, aligned with ba_gate / win_gate
            _init_sigmoid_gate_closed(self.write_gate[-2], bias=-2.5)
        else:
            self.write_gate = None

    def forward(
        self,
        current: torch.Tensor,
        historical: Sequence[torch.Tensor],
        token_indices: torch.Tensor,
        global_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if token_indices.numel() == 0:
            return current
        if len(historical) == 0 and global_kv is None:
            return current
        b, _, d, h, w = current.shape
        _, k, _ = token_indices.shape
        bd = torch.arange(b, device=current.device)[:, None].expand(b, k)
        di = token_indices[..., 0].clamp(0, d - 1)
        hi = token_indices[..., 1].clamp(0, h - 1)
        wi = token_indices[..., 2].clamp(0, w - 1)
        sz = (d, h, w)

        def sample(feat: torch.Tensor) -> torch.Tensor:
            # shares align_spatial3d with _prep_hist (avg_pool when downsampling)
            feat = align_spatial3d(feat, sz)
            return feat[bd, :, di, hi, wi]

        hist_toks = [sample(proj(hb)) for hb, proj in zip(historical, self.hist_projs)]
        if global_kv is not None:
            # global reference: one pooled token broadcast to every anchor, parallel to hist
            for li in range(global_kv.shape[1]):
                hist_toks.append(global_kv[:, li, :].unsqueeze(1).expand(-1, k, -1))
        partial = sample(current)
        out_toks = self.attn(hist_toks, partial)
        if self.write_gate is not None:
            g = self.write_gate(torch.cat([partial, out_toks], dim=-1))
            out_toks = partial + g * (out_toks - partial)
        # accumulate in fp32 before casting: avoids premature fp16 truncation under AMP
        bsz, ch = current.shape[0], current.shape[1]
        delta_write = torch.zeros(
            bsz, ch, d, h, w, dtype=torch.float32, device=current.device
        )
        delta_write[bd, :, di, hi, wi] = (out_toks - partial).float()
        return current + delta_write.to(dtype=current.dtype)


class PriorHead(nn.Module):
    """Prior head: 1×1 Conv -> softmax, a voxel-wise multi-class soft prior P_loc.

    One structure, two mount points:
      * inside the block (Stages 2–5): ``P_loc(h)`` is combined with the residual energy
        into ``P_eff`` and only drives anchor selection — never a segmentation output;
      * at the host bottleneck (Stage 6): ``PriorHead(f5')`` -> ``prior``, supervised by
        ``λ_prior · L_prior``.

    Cold start: ``weight ~ N(0, 0.01²)`` and zero bias keep the softmax near-uniform with a
    small spatial perturbation, so the first read is not driven by a flat prior.
    """

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Conv3d(int(in_ch), int(num_classes), 1, bias=True)
        nn.init.normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(x), dim=1)


class AttentionGate(nn.Module):
    """AG — decoder attention gate (paper Fig. 1c).

    Attention-U-Net style spatial gate that scales a skip with the upsampled semantics g:
    ``α = σ(ψ(ReLU(W_g(g) + W_x(x))))``, ``skip' = skip ⊙ α``.  Complementary to the
    encoder-side sparse read.

    Cold start: zero-init ``ψ`` with bias +4 gives α ≈ 0.98, so skips are not suppressed
    early in training.
    """

    def __init__(self, f_g: int, f_l: int, f_int: Optional[int] = None) -> None:
        super().__init__()
        mid = int(f_int) if f_int is not None else max(int(f_l) // 2, 1)
        self.W_g = nn.Conv3d(int(f_g), mid, kernel_size=1, bias=True)
        self.W_x = nn.Conv3d(int(f_l), mid, kernel_size=1, bias=True)
        self.psi = nn.Conv3d(mid, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.psi.weight)
        nn.init.constant_(self.psi.bias, 4.0)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = align_spatial3d(g, x.shape[2:])
        a = F.relu(self.W_g(g) + self.W_x(x), inplace=True)
        alpha = torch.sigmoid(self.psi(a))
        return x * alpha


class SparkSegBlock(nn.Module):
    """SparkSeg Block — one budgeted read/write operator per guided stage.

    Paper **SparkSeg Block** (Fig. 1b): one per guided stage (Stages 2–5); TAE ->
    PriorHead/PGTS -> MSBA read + gated write-back, with WinMHSA3D running in parallel.

    Modes:
      - orient_only (enable_read=False): TAE only (Stage 1 / E0)
      - full (enable_read=True): TAE -> P_eff -> PGTS -> MSBA(BaOnBase) || WinMHSA3D

    Returns ``(out, hist≡out, p_eff|None)``.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        *,
        axial_kernel: int = 5,
        anchor_k: int = 32,
        anchor_heads: int = 4,
        anchor_drop: float = 0.0,
        enable_read: bool = True,
        energy_fallback_lambda: float = 0.25,
        enable_window_mhsa: bool = False,
        window_size: int = 4,
        window_heads: int = 4,
        window_use_shift: bool = True,
        hist_ch: Optional[int] = None,
        topk_jitter: float = 1e-2,
        gate_bias_init: float = -2.5,
    ) -> None:
        super().__init__()
        self.enable_read = bool(enable_read)
        self.energy_fallback_lambda = float(energy_fallback_lambda)
        self.enable_window_mhsa = bool(enable_window_mhsa)
        self.window_use_shift = bool(window_use_shift)
        self.topk_jitter = float(topk_jitter)
        self.gate_bias_init = float(gate_bias_init)

        self.tae = TAE(int(dim), kernel=int(axial_kernel))
        if self.enable_read:
            self.gamma = nn.Parameter(torch.zeros(1))
            self.prior_head = PriorHead(int(dim), int(num_classes))
            hist_in = int(hist_ch) if hist_ch is not None else int(dim)
            self.hist_ch = hist_in
            self.pgts = PGTS(int(dim), target_num_tokens=int(anchor_k))
            self.msba = MSBA(
                int(dim),
                int(anchor_heads),
                float(anchor_drop),
                (hist_in,),
                gated_writeback=True,
            )
            self.ba_gate = _ScalarGate3D(int(dim), bias=self.gate_bias_init)
            self.global_proj = nn.Linear(int(dim), int(dim))
        else:
            self.gamma = None
            self.prior_head = None
            self.hist_ch = int(dim)
            self.pgts = None
            self.msba = None
            self.ba_gate = None
            self.global_proj = None

        if self.enable_window_mhsa:
            self.win_mhsa3d = WinMHSA3D(
                int(dim),
                num_heads=int(window_heads),
                window_size=int(window_size),
                attn_drop=float(anchor_drop),
            )
            win_bias = -3.2 if self.window_use_shift else self.gate_bias_init
            self.win_gate = _ScalarGate3D(int(dim), bias=float(win_bias))
        else:
            self.win_mhsa3d = None
            self.win_gate = None

        self._last_p_eff: Optional[torch.Tensor] = None

    def _prep_hist(self, hist_prev: Optional[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        if hist_prev is None:
            if int(h.shape[1]) != int(self.hist_ch):
                raise RuntimeError(
                    f"SparkSegBlock self-hist: dim {h.shape[1]} != hist_ch {self.hist_ch}"
                )
            return h
        hist = hist_prev.detach()
        if int(hist.shape[1]) != int(self.hist_ch):
            raise RuntimeError(
                f"SparkSegBlock hist channels {hist.shape[1]} != expected hist_ch {self.hist_ch}"
            )
        if hist.shape[2:] != h.shape[2:]:
            hist = align_spatial3d(hist, h.shape[2:])
        return hist

    def _p_eff(self, h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        p_loc = self.prior_head(h)
        e = delta_energy_per_sample(delta)
        gamma = F.softplus(self.gamma)
        scale = 1.0 + gamma * e
        if p_loc.shape[1] > 1:
            p_star = p_loc.clone()
            p_star[:, 1:] = p_loc[:, 1:] * scale
        else:
            p_star = p_loc * scale
        fg = p_star[:, 1:].sum(dim=1, keepdim=True) if p_star.shape[1] > 1 else p_star
        fg = fg.clamp(0.0, 1.0)
        p_eff = (p_star + self.energy_fallback_lambda * e * (1.0 - fg)).clamp(0.0, 1.0)
        self._last_p_eff = p_eff
        return p_eff

    def forward(
        self,
        f: torch.Tensor,
        hist_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h, delta = self.tae(f)

        if not self.enable_read:
            return h, h, None

        p_eff = self._p_eff(h, delta)
        hist = self._prep_hist(hist_prev, h)

        importance, idx, base = self.pgts(
            h, compute_token_indices=True, spatial_prior=p_eff
        )
        if self.training and self.topk_jitter > 0:
            jittered = apply_topk_jitter(importance, self.topk_jitter)
            idx = topk_coords_from_map(jittered, int(self.pgts.target_num_tokens))

        g = F.adaptive_avg_pool3d(h, 1).flatten(1)
        kv = self.global_proj(g).unsqueeze(1)
        attn = self.msba(base, [hist], idx, global_kv=kv)
        # BaOnBase: refine passes through; sigma only scales sparse residual
        out = base + self.ba_gate(h) * (attn - base)

        if self.win_mhsa3d is not None and self.win_gate is not None:
            d0 = self.win_mhsa3d(h, shift_size=0) - h
            if self.window_use_shift:
                sh = self.win_mhsa3d.window_size // 2
                ds = self.win_mhsa3d(h, shift_size=sh) - h
                cand_delta = d0 + ds
            else:
                cand_delta = d0
            out = out + self.win_gate(h) * cand_delta

        return out, out, p_eff


class SparkSegDecoder(nn.Module):
    """Decoder with attention gates (paper Fig. 1c).

    Wraps the dynamic-network-architectures ``UNetDecoder`` and gates every skip after the
    transposed convolution and before concatenation.  A degenerate decoder without
    ``stages`` falls back to forwarding the wrapped ``UNetDecoder`` unchanged.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        n_conv_per_stage: Union[int, Sequence[int]],
        deep_supervision: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

        self.decoder = UNetDecoder(
            encoder,
            int(num_classes),
            n_conv_per_stage,
            bool(deep_supervision),
            **kwargs,
        )
        self.deep_supervision = bool(deep_supervision)
        gates: List[nn.Module] = []
        n_dec = len(getattr(self.decoder, "stages", []))
        out_ch = getattr(encoder, "output_channels", None)
        if n_dec > 0 and out_ch is not None and len(out_ch) >= n_dec + 1:
            for s in range(n_dec):
                skip_ch = int(out_ch[-(s + 2)])
                gates.append(AttentionGate(skip_ch, skip_ch))
        self.attn_gates = nn.ModuleList(gates)

    def forward(self, skips: List[torch.Tensor]):
        if len(self.attn_gates) == 0:
            return self.decoder(skips)

        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        stages = self.decoder.stages
        for s in range(len(stages)):
            g = self.decoder.transpconvs[s](lres_input)
            skip = self.attn_gates[s](g, skips[-(s + 2)])
            x = torch.cat((g, skip), 1)
            x = stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[s](x))
            elif s == (len(stages) - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            return seg_outputs[0]
        return seg_outputs


__all__ = [
    # building blocks
    "TAE",
    "PGTS",
    "MSBA",
    "WinMHSA3D",
    "PriorHead",
    "AttentionGate",
    # composed blocks
    "SparkSegBlock",
    "SparkSegDecoder",
]
