"""
LinearWarmupCosineAnnealingLR - 与官方 SwinUNETR (MONAI research-contributions) 一致

官方配置 (SwinUNETR/BRATS21/main.py):
- lrschedule: warmup_cosine
- warmup_epochs: 50
- max_epochs: 300

Phase 1 (warmup): lr 从 0 线性增长到 initial_lr
Phase 2 (cosine): lr 按余弦曲线从 initial_lr 衰减到 eta_min
"""
import math
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """
    Linear Warmup + Cosine Annealing 学习率调度器
    与 MONAI SwinUNETR 官方实现一致
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Phase 1: Linear Warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Phase 2: Cosine Annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            progress = min(progress, 1.0)
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * 0.5
                * (1.0 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
