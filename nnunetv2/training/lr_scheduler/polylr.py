from torch.optim.lr_scheduler import LRScheduler


class PolyLRScheduler(LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class WarmupPolyLRScheduler(LRScheduler):
    """
    WarmupPolyLRScheduler，采用 Linear Warmup + PolyLR Decay 两阶段策略：
    Linear Warmup + PolyLR Decay 学习率调度器
    专为 Transformer 架构设计
    lr
    ↑
    │        ╱‾‾‾‾╲
    │      ╱        ╲
    │    ╱            ╲
    │  ╱                ╲
    │╱                    ╲___
    └─────────────────────────→ epoch
    0    50                  1000
    warmup     poly decay

    Phase 1 (warmup):  lr = initial_lr * (step + 1) / warmup_steps   [线性增长]
    Phase 2 (decay):   lr = initial_lr * (1 - progress) ^ exponent    [多项式衰减]

    其中 progress = (step - warmup_steps) / (max_steps - warmup_steps)

    参考：
    - "Attention Is All You Need" (Vaswani et al., 2017) 中的 Noam Warmup 思想
    - 医学图像 Transformer (SwinUNETR, nnFormer) 普遍采用的 Warmup + Decay 策略
    """

    def __init__(self, optimizer, initial_lr: float, max_steps: int,
                 warmup_steps: int = 50, exponent: float = 0.9,
                 current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.warmup_steps = max(1, warmup_steps)
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:
            # Phase 1: Linear Warmup — 从 initial_lr / warmup_steps 线性增长到 initial_lr
            new_lr = self.initial_lr * (current_step + 1) / self.warmup_steps
        else:
            # Phase 2: PolyLR Decay — 从 initial_lr 多项式衰减到 ~0
            decay_steps = self.max_steps - self.warmup_steps
            if decay_steps <= 0:
                new_lr = self.initial_lr
            else:
                progress = (current_step - self.warmup_steps) / decay_steps
                progress = min(progress, 1.0)
                new_lr = self.initial_lr * (1 - progress) ** self.exponent

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
