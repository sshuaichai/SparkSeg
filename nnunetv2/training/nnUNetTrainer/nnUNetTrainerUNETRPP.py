"""
UNETR++ Trainer for nnUNet v2
Based on: "UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"

官方配置 (Amshaker/unetr_plus_plus unetr_pp_trainer_synapse.py):
- Optimizer: SGD, lr=1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5
- LR Scheduler: Poly decay (epoch-based), lr = initial_lr * (1 - epoch/max_epochs)^0.9
- Epochs: 1000
- Loss: DC_and_CE (default)
"""
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from typing import Union, List, Tuple
import numpy as np
import torch
from torch import nn
from nnunetv2.training.network.UNETRPP import UNETRPP


def _poly_lr_epoch(epoch: int, max_epochs: int, initial_lr: float, exponent: float = 0.9) -> float:
    """UNETR++ 官方 epoch-based poly lr"""
    return initial_lr * (1 - epoch / max_epochs) ** exponent


class nnUNetTrainerUNETRPP(nnUNetTrainerNoDeepSupervision):
    def __init__(self,
                 plans: dict,
                 configuration: str,
                 fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.num_output_channels = self.label_manager.num_segmentation_heads
        
        # 获取原始patch_size并进行调整以适应UNETR++的要求
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            # UNETR++使用4的倍数作为patch size，确保与ConvNet stem兼容
            if (original_patch_size[i] / 4) < 1 or ((original_patch_size[i] / 4) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 4 + 0.5) * 4
            else:
                new_patch_size[i] = original_patch_size[i]
                
        # 更新配置文件中的patch_size
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        
        # 将调整后的patch_size保存到plans文件中
        self._update_plans_with_patch_size(new_patch_size, configuration)
        self.print_to_log_file(f"Patch size changed from {original_patch_size} to {new_patch_size}")
        self.print_to_log_file(f"Patch size {new_patch_size} for UNETR++")
        
        # 设置架构参数，确保patch_size能传递到build_network_architecture
        self.arch_init_kwargs = {'patch_size': new_patch_size}

        # 官方 UNETR++ 配置 (unetr_pp_trainer_synapse.py)
        self.initial_lr = 1e-2  # 官方 initial_lr=1e-2
        self.weight_decay = 3e-5  # 继承 Trainer_synapse
        self.momentum = 0.99  # 官方 momentum=0.99, nesterov=True
        self.grad_scaler = None
        self.num_epochs = 200
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.current_epoch = 0
        self.enable_deep_supervision = False  # 禁用深度监督

    def _update_plans_with_patch_size(self, new_patch_size: list, configuration: str):
        """仅更新内存中的 plans（会随训练保存到输出目录）。不写回预处理目录，避免不同 Trainer 互相覆盖。"""
        try:
            plans = self.plans_manager.plans
            if 'configurations' not in plans or configuration not in plans['configurations']:
                return
            cfg = plans['configurations'][configuration]
            cfg['patch_size'] = new_patch_size
            arch = cfg.get('architecture', {})
            if 'network_arch_init_kwargs' not in arch:
                arch['network_arch_init_kwargs'] = {}
            arch['network_arch_init_kwargs']['patch_size'] = new_patch_size
            self.print_to_log_file(f"Updated patch_size in memory (output dir only): {new_patch_size}")
        except Exception as e:
            self.print_to_log_file(f"Error updating plans: {str(e)}")

    def get_default_search_space(self) -> dict:
        """官方 UNETR++ 超参数搜索空间"""
        return {
            "initial_lr": (5e-3, 2e-2),
            "weight_decay": (1e-5, 1e-4),
            "momentum": (0.95, 0.99),
            "oversample_foreground_percent": (0.25, 0.4),
        }
    
    def configure_optimizers(self):
        """
        官方 UNETR++ 配置: SGD + Poly decay (epoch-based)
        lr = initial_lr * (1 - epoch/max_epochs)^0.9, 每 epoch 开始时更新
        """
        if hasattr(self, "hyperparameters"):
            initial_lr = self.hyperparameters.get("initial_lr", self.initial_lr)
            weight_decay = self.hyperparameters.get("weight_decay", self.weight_decay)
            momentum = self.hyperparameters.get("momentum", self.momentum)
        else:
            initial_lr = self.initial_lr
            weight_decay = self.weight_decay
            momentum = self.momentum

        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=initial_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        # 官方 UNETR++: poly_lr 在 on_train_epoch_start 中手动更新，lr_scheduler 占位
        lr_scheduler = None
        self.print_to_log_file(
            f"UNETR++ official config: SGD lr={initial_lr}, momentum={momentum}, nesterov=True, "
            f"wd={weight_decay}, PolyLR(epoch-based, exp=0.9)"
        )
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        """官方 UNETR++: epoch-based poly lr 在每 epoch 开始时更新"""
        self.network.train()
        new_lr = _poly_lr_epoch(
            self.current_epoch, self.num_epochs, self.initial_lr, exponent=0.9
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.print_to_log_file(f"Current weight decay: {np.round(self.weight_decay, decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = False  # 默认禁用深度监督
    ) -> nn.Module:
        """
        构建UNETR++网络架构
        
        Args:
            architecture_class_name: 架构类名
            arch_init_kwargs: 架构初始化参数
            arch_init_kwargs_req_import: 需要导入的模块
            num_input_channels: 输入通道数
            num_output_channels: 输出通道数
            enable_deep_supervision: 是否启用深度监督（已禁用）
        
        Returns:
            UNETR++网络模型
        """
        print(f"arch_init_kwargs: {arch_init_kwargs}")
        
        # 从arch_init_kwargs中获取patch_size
        patch_size = arch_init_kwargs.get('patch_size')
        if patch_size is None:
            raise ValueError("patch_size is required in arch_init_kwargs for UNETR++")
            
        spatial_dims = len(patch_size)
        print(f"Building UNETR++ with patch_size: {patch_size}, spatial_dims: {spatial_dims}")

        # 获取其他网络参数，设置UNETR++的默认值
        feature_size = arch_init_kwargs.get("feature_size", 16)
        hidden_size = arch_init_kwargs.get("hidden_size", 256)
        num_heads = arch_init_kwargs.get("num_heads", 4)
        dropout_rate = arch_init_kwargs.get("dropout_rate", 0.0)
        
        # 根据patch_size计算适合的dims和depths
        dims = arch_init_kwargs.get("dims", [feature_size, feature_size*2, feature_size*4, feature_size*8])
        depths = arch_init_kwargs.get("depths", [3, 3, 3, 3])

        try:
            # 构建UNETR++模型 - 强制禁用深度监督并传递patch_size
            model = UNETRPP(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                feature_size=feature_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                pos_embed="perceptron",  # UNETR++论文中的位置编码类型
                norm_name="instance",    # 实例归一化
                dropout_rate=dropout_rate,
                depths=depths,
                dims=dims,
                conv_op=nn.Conv3d if spatial_dims == 3 else nn.Conv2d,
                do_ds=False,  # 强制禁用深度监督
                patch_size=patch_size  # 传递patch_size用于动态计算input_size
            )
        except Exception as e:
            print(f"Error creating UNETR++: {str(e)}")
            raise

        return model

     ### 重写的initialize方法以支持patch_size ###
    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            # 创建包含patch_size的arch_init_kwargs 以添加patch_size
            arch_init_kwargs = dict(self.configuration_manager.network_arch_init_kwargs)  # 复制原有参数
            arch_init_kwargs['patch_size'] = self.configuration_manager.patch_size  

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                arch_init_kwargs,  # 使用包含patch_size的参数
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def train_step(self, batch: dict) -> dict:
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
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """参考umamba项目的验证步骤"""
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

            # the following is needed for online evaluation. Fake dice (green line)
            axes = [0] + list(range(2, output.ndim))

            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                # no need for softmax
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target != self.label_manager.ignore_label).float()
                    # CAREFUL that you don't rely on target after this line!
                    target[target == self.label_manager.ignore_label] = 0
                else:
                    mask = 1 - target[:, -1:]
                    # CAREFUL that you don't rely on target after this line!
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

            return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def set_deep_supervision_enabled(self, enabled: bool):
        
        self.enable_deep_supervision = False
        
        # 如果网络有相关属性，设置为False
        if hasattr(self.network, 'decoder') and hasattr(self.network.decoder, 'deep_supervision'):
            self.network.decoder.deep_supervision = False
        
        if hasattr(self.network, 'enable_deep_supervision'):
            self.network.enable_deep_supervision(False)

