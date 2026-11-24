"""
SwinUnet Trainer for nnUNet v2
Based on: "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"
https://arxiv.org/abs/2105.05537

官方配置 (HuCaoFighting/Swin-Unet config.py):
- Optimizer: AdamW, lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8
- LR Scheduler: Cosine, warmup_epochs=20, min_lr=5e-6
- Epochs: 300, clip_grad=5.0
- Loss: 0.4*CE + 0.6*Dice
"""

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple
from nnunetv2.training.network.SwinUnet import SwinUnet
from nnunetv2.training.lr_scheduler.warmup_cosine import LinearWarmupCosineAnnealingLR


class nnUNetTrainerSwinUNET(nnUNetTrainerNoDeepSupervision):
    """
    Swin-Unet trainer based on: "Cao et al.,
    Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
    <https://arxiv.org/abs/2105.05537>"
    
    与官方 HuCaoFighting/Swin-Unet config.py 一致。
    """
    
    def __init__(self,
                 plans: dict,
                 configuration: str,
                 fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 调整 patch_size 以符合 Swin-Unet 要求
        self._adjust_patch_size_for_swin_unet()
        
        # 官方 Swin-Unet 配置 (config.py)
        # 医学图像数据集较小，weight_decay=0.05 易导致过正则化、模型偏向预测背景，降为 0.01
        self.grad_scaler = None
        self.initial_lr = 5e-4   # BASE_LR=5e-4
        self.weight_decay = 0.05
        self.warmup_epochs = 5   # WARMUP_EPOCHS=20
        self.min_lr = 5e-6        # MIN_LR=5e-6
        self.clip_grad = 5.0      # CLIP_GRAD=5.0
        self.num_epochs = 200
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.current_epoch = 0
        self.enable_deep_supervision = False
        
        # SwinUnet 特定参数
        self.swin_embed_dim = 96
        self.swin_depths = (2, 2, 2, 2)
        self.swin_depths_decoder = (1, 2, 2, 2)
        self.swin_num_heads = (3, 6, 12, 24)
        self.swin_window_size = 8  # Swin-Unet 推荐使用 8 或 7
        # Swin-Unet MLP 比例
        self.swin_mlp_ratio = 4.0
        # Swin-Unet patch size 推荐使用 4 或 8
        self.swin_patch_size = 4

    def _do_i_compile(self) -> bool:
        """
        禁用 torch.compile，因为在 DDP 模式下与 torch.compile 存在兼容性问题
        """
        return False

    def _adjust_patch_size_for_swin_unet(self):
        """调整 patch_size 以符合 Swin-Unet 的要求"""
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = list(original_patch_size)
        
        # Swin-Unet: patch size 必须能被 patch_size * 2^(num_layers-1) 整除
        # 对于 4 层网络，需要能被 4 * 8 = 32 整除
        min_divisor = 32
        
        for i in range(len(original_patch_size)):
            if original_patch_size[i] % min_divisor != 0:
                new_patch_size[i] = ((original_patch_size[i] // min_divisor) + 1) * min_divisor
            
            # 确保最小尺寸
            if new_patch_size[i] < min_divisor:
                new_patch_size[i] = min_divisor
        
        if new_patch_size != list(original_patch_size):
            self.configuration_manager.configuration['patch_size'] = new_patch_size
            self.print_to_log_file(f"Patch size changed from {original_patch_size} to {new_patch_size}")
            self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size

    def get_default_search_space(self) -> dict:
        """官方 Swin-Unet 超参数搜索空间 (config.py)"""
        return {
            "initial_lr": (1e-4, 1e-3),
            "weight_decay": (0.01, 0.1),
            "warmup_epochs": (10, 30),
            "oversample_foreground_percent": (0.25, 0.4),
        }

    def _build_loss(self):
        """官方 Swin-Unet: 0.4*CE + 0.6*Dice"""
        if self.label_manager.has_regions:
            raise NotImplementedError("Swin-Unet official loss only supports non-regions")
        return DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
            weight_ce=0.4, weight_dice=0.6,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

    def configure_optimizers(self):
        """
        官方 Swin-Unet 配置 (config.py): AdamW + Cosine + Warmup
        """
        if hasattr(self, "hyperparameters"):
            initial_lr = self.hyperparameters.get("initial_lr", self.initial_lr)
            weight_decay = self.hyperparameters.get("weight_decay", self.weight_decay)
            warmup_epochs = self.hyperparameters.get("warmup_epochs", self.warmup_epochs)
        else:
            initial_lr = self.initial_lr
            weight_decay = self.weight_decay
            warmup_epochs = self.warmup_epochs

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        min_lr = getattr(self, "min_lr", 5e-6)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=self.num_epochs,
            eta_min=min_lr,
        )
        self.print_to_log_file(
            f"Swin-Unet config.py: AdamW lr={initial_lr}, wd={weight_decay}, "
            f"betas=(0.9,0.999), eps=1e-8, "
            f"WarmupCosine(warmup={warmup_epochs}, max={self.num_epochs}, min_lr={min_lr}), "
            f"clip_grad={self.clip_grad}, loss=0.4*CE+0.6*Dice"
        )
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = False
    ) -> nn.Module:
        """构建 SwinUnet 网络架构"""
        import math
        
        # 获取配置参数
        patch_size = arch_init_kwargs.get('patch_size')
        spatial_dims = len(patch_size)  # 2D 或 3D
        
        # 获取 SwinUnet 参数
        embed_dim = arch_init_kwargs.get("embed_dim", 96)
        depths = arch_init_kwargs.get("depths", (2, 2, 2, 2))
        depths_decoder = arch_init_kwargs.get("depths_decoder", (1, 2, 2, 2))
        num_heads = arch_init_kwargs.get("num_heads", (3, 6, 12, 24))
        window_size = arch_init_kwargs.get("window_size", 7)
        mlp_ratio = arch_init_kwargs.get("mlp_ratio", 4.0)
        swin_patch_size = arch_init_kwargs.get("swin_patch_size", 4)
        
        # 获取 dropout 参数
        drop_rate = arch_init_kwargs.get("drop_rate", 0.0)
        attn_drop_rate = arch_init_kwargs.get("attn_drop_rate", 0.0)
        drop_path_rate = arch_init_kwargs.get("drop_path_rate", 0.1)
        
        # 使用 checkpoint 节省显存
        use_checkpoint = arch_init_kwargs.get("use_checkpoint", False)
        
        # 计算经过 patch embedding 后的特征图尺寸
        feature_sizes = [p // swin_patch_size for p in patch_size]
        
        # 调整 window_size 以确保能够整除所有层的特征图尺寸
        if isinstance(window_size, int):
            # 计算所有层的最小特征图尺寸
            num_layers = len(depths)
            min_feature_sizes = []
            for layer_idx in range(num_layers):
                layer_sizes = [fs // (2 ** layer_idx) for fs in feature_sizes]
                min_feature_sizes.extend(layer_sizes)
            
            # 找到所有特征图尺寸的 GCD
            gcd_size = min_feature_sizes[0]
            for fs in min_feature_sizes[1:]:
                if fs > 0:
                    gcd_size = math.gcd(gcd_size, fs)
            
            # 选择合适的 window_size
            # 优先选择能整除所有尺寸的最大值，但不超过原始 window_size
            if gcd_size >= window_size:
                adjusted_window_size = window_size
            else:
                # 找到 gcd_size 的因子中最接近原始 window_size 的
                factors = [i for i in range(2, gcd_size + 1) if gcd_size % i == 0]
                if factors:
                    adjusted_window_size = max([f for f in factors if f <= window_size], default=gcd_size)
                else:
                    adjusted_window_size = gcd_size
            
            adjusted_window_size = max(2, adjusted_window_size)  # 至少为 2
        else:
            adjusted_window_size = window_size

        print(f"SwinUnet config: spatial_dims={spatial_dims}, img_size={patch_size}, "
              f"embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}, "
              f"window_size={adjusted_window_size}, patch_size={swin_patch_size}")
        print(f"Feature sizes after patch embed: {feature_sizes}")
        print(f"Dropout config: drop_rate={drop_rate:.3f}, attn_drop_rate={attn_drop_rate:.3f}, "
              f"drop_path_rate={drop_path_rate:.3f}")

        model = SwinUnet(
            img_size=patch_size,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            patch_size=swin_patch_size,
            embed_dim=embed_dim,
            depths=depths,
            depths_decoder=depths_decoder,
            num_heads=num_heads,
            window_size=adjusted_window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"SwinUnet model created successfully with {num_params:,} parameters")
        
        return model

    def initialize(self):
        """重写 initialize 方法以支持 patch_size 传递"""
        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, 
                self.configuration_manager,
                self.dataset_json
            )
            
            # 创建包含 patch_size 和其他参数的 arch_init_kwargs
            arch_init_kwargs = dict(self.configuration_manager.network_arch_init_kwargs)
            arch_init_kwargs['patch_size'] = self.configuration_manager.patch_size
            
            # 添加 SwinUnet 特定参数
            arch_init_kwargs.setdefault('embed_dim', self.swin_embed_dim)
            arch_init_kwargs.setdefault('depths', self.swin_depths)
            arch_init_kwargs.setdefault('depths_decoder', self.swin_depths_decoder)
            arch_init_kwargs.setdefault('num_heads', self.swin_num_heads)
            arch_init_kwargs.setdefault('window_size', self.swin_window_size)
            arch_init_kwargs.setdefault('mlp_ratio', self.swin_mlp_ratio)
            arch_init_kwargs.setdefault('swin_patch_size', self.swin_patch_size)
            arch_init_kwargs.setdefault('drop_rate', getattr(self, 'drop_rate', 0.0))
            arch_init_kwargs.setdefault('attn_drop_rate', getattr(self, 'attn_drop_rate', 0.0))
            arch_init_kwargs.setdefault('drop_path_rate', getattr(self, 'drop_path_rate', 0.1))

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            
            # 编译网络以提升速度
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            
            # DDP 支持
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def on_train_start(self):
        """训练开始前初始化学习率（warmup 从 epoch 0 生效）"""
        super().on_train_start()
        self.lr_scheduler.step(0)  # 设置 epoch 0 的 lr，避免 PyTorch 的 step-before-optimizer 警告

    def on_train_epoch_start(self):
        """Swin-Unet: 每 epoch 开始"""
        self.network.train()
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_train_epoch_end(self, train_outputs: list):
        """训练 epoch 结束后更新学习率（符合 PyTorch 建议：optimizer.step 后再 scheduler.step）"""
        super().on_train_epoch_end(train_outputs)
        self.lr_scheduler.step(self.current_epoch + 1)

    def train_step(self, batch: dict) -> dict:
        """训练步骤"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
        self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """验证步骤"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            output = self.network(data)
            del data
            l = self.loss(output, target)

            axes = [0] + list(range(2, output.ndim))
            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target != self.label_manager.ignore_label).float()
                    target = target.clone()
                    target[target == self.label_manager.ignore_label] = 0
                else:
                    mask = 1 - target[:, -1:]
                    target = target[:, :-1]
            else:
                mask = None

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard
        }

    def set_deep_supervision_enabled(self, enabled: bool):
        """禁用深度监督（SwinUnet 不支持深度监督）"""
        self.enable_deep_supervision = False


# class nnUNetTrainerSwinUNET_Small(nnUNetTrainerSwinUNET):
#     """小型 SwinUnet 变体，适合显存较小的情况"""
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.swin_embed_dim = 48
#         self.swin_depths = (2, 2, 2, 2)
#         self.swin_depths_decoder = (1, 2, 2, 2)
#         self.swin_num_heads = (3, 6, 12, 24)


# class nnUNetTrainerSwinUNET_Large(nnUNetTrainerSwinUNET):
#     """大型 SwinUnet 变体，适合显存较大的情况"""
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.swin_embed_dim = 128
#         self.swin_depths = (2, 2, 6, 2)
#         self.swin_depths_decoder = (2, 6, 2, 2)
#         self.swin_num_heads = (4, 8, 16, 32)

