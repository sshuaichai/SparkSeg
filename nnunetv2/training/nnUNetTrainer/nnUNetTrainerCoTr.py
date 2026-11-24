"""
CoTr Trainer - 官方 YtongXie/CoTr 配置

官方 nnUNetTrainerV2_ResTrans:
- Optimizer: SGD, lr=1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5
- LR Scheduler: WarmupPolyLR (warmup=50 epochs, exponent=0.9)
- Deep supervision: True (官方)
"""
from typing import Union, List, Tuple
from torch import nn
import torch
from nnunetv2.training.lr_scheduler.polylr import WarmupPolyLRScheduler
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.network.CoTr import (
    CoTr,
    get_cotr_config,
    create_cotr_from_config,
    CONFIGS_CoTr
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision


class nnUNetTrainerCoTr(nnUNetTrainerNoDeepSupervision):
    """
    nnUNet Trainer for CoTr (CNN-Transformer) with default configuration.
    
    CoTr: Efficient 3D Medical Image Segmentation by bridging CNN and Transformer
    Paper: https://arxiv.org/abs/2103.03024
    """
    
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.num_output_channels = self.label_manager.num_segmentation_heads
        self.img_size = self.configuration_manager.patch_size
        
        # CoTr 配置名称
        self.cotr_config_name = 'CoTr-Default'
        
        # 将配置保存到plans文件中
        self._update_plans_with_cotr_config(self.img_size, self.cotr_config_name, configuration)
        
        # 同步到内存中的配置
        if hasattr(self.configuration_manager, 'configuration'):
            self.configuration_manager.configuration['cotr_config'] = self.cotr_config_name
            if 'architecture' not in self.configuration_manager.configuration:
                self.configuration_manager.configuration['architecture'] = {}
            if 'network_arch_init_kwargs' not in self.configuration_manager.configuration['architecture']:
                self.configuration_manager.configuration['architecture']['network_arch_init_kwargs'] = {}
            self.configuration_manager.configuration['architecture']['network_arch_init_kwargs']['cotr_config'] = self.cotr_config_name
            self.configuration_manager.configuration['architecture']['network_arch_init_kwargs']['patch_size'] = self.img_size
        
        self.arch_init_kwargs = {
            'patch_size': self.img_size,
            'cotr_config': self.cotr_config_name
        }
        self.print_to_log_file(f"Patch size {self.img_size} for CoTr with config {self.cotr_config_name}")

        # 训练参数 - 官方 nnUNetTrainerV2_ResTrans
        self.grad_scaler = None
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.momentum = 0.99
        self.nesterov = True
        self.num_epochs = 200
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.current_epoch = 0
        self.warmup_epochs = 5
        
        self.enable_deep_supervision = True

    def _update_plans_with_cotr_config(self, new_patch_size: list, cotr_config: str, configuration: str):
        """仅更新内存中的 plans（会随训练保存到输出目录）。不写回预处理目录，避免不同 Trainer 互相覆盖。"""
        try:
            plans = self.plans_manager.plans
            if 'configurations' not in plans or configuration not in plans['configurations']:
                return
            cfg = plans['configurations'][configuration]
            cfg['patch_size'] = new_patch_size
            cfg['cotr_config'] = cotr_config
            if 'architecture' not in cfg:
                cfg['architecture'] = {}
            arch = cfg['architecture']
            if 'network_arch_init_kwargs' not in arch:
                arch['network_arch_init_kwargs'] = {}
            arch['network_arch_init_kwargs']['patch_size'] = new_patch_size
            arch['network_arch_init_kwargs']['cotr_config'] = cotr_config
            self.print_to_log_file(f"Updated patch_size and cotr_config in memory (output dir only): {new_patch_size}, {cotr_config}")
        except Exception as e:
            self.print_to_log_file(f"Error updating plans: {str(e)}")

    def get_default_search_space(self) -> dict:
        """官方 CoTr 超参数搜索空间 (SGD + PolyLR)"""
        return {
            "initial_lr": (1e-3, 1e-1),
            "weight_decay": (1e-5, 1e-3),
            "momentum": (0.9, 0.99),
            "oversample_foreground_percent": (0.25, 0.4),
        }
    
    def configure_optimizers(self):
        """
        官方 nnUNetTrainerV2_ResTrans 配置: SGD + PolyLR
        (YtongXie/CoTr CoTr_package/CoTr/training/network_training/nnUNetTrainerV2_ResTrans.py)
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
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=self.nesterov,
        )

        warmup = getattr(self, "warmup_epochs", 50)
        lr_scheduler = WarmupPolyLRScheduler(
            optimizer, initial_lr, self.num_epochs,
            warmup_steps=warmup, exponent=0.9
        )
        self.print_to_log_file(
            f"CoTr official config: SGD lr={initial_lr}, momentum={momentum}, wd={weight_decay}, "
            f"nesterov=True, WarmupPolyLR(warmup={warmup}, exp=0.9)"
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
        """
        构建 CoTr 模型。
        """
        print(f"arch_init_kwargs: {arch_init_kwargs}")
        
        patch_size = arch_init_kwargs.get('patch_size')
        cotr_config = arch_init_kwargs.get('cotr_config', 'CoTr-Default')
        
        if patch_size is None:
            raise ValueError("patch_size is required in arch_init_kwargs")
            
        spatial_dims = len(patch_size)
        print(f"Building CoTr with patch_size: {patch_size}, spatial_dims: {spatial_dims}, config: {cotr_config}")
        
        try:
            model = create_cotr_from_config(
                cotr_config,
                spatial_dims=spatial_dims,
                in_channels=num_input_channels,
                num_classes=num_output_channels,
                patch_size=tuple(patch_size),
                deep_supervision=enable_deep_supervision
            )
            print(f"Successfully created CoTr using config: {cotr_config}")
            
        except (ValueError, KeyError) as e:
            print(f"Config factory failed ({e}), falling back to direct creation...")
            
            model = CoTr(
                spatial_dims=spatial_dims,
                in_channels=num_input_channels,
                num_classes=num_output_channels,
                patch_size=tuple(patch_size),
                deep_supervision=enable_deep_supervision,
                norm_cfg='BN',
                activation_cfg='ReLU',
                weight_std=False,
                transformer_d_model=384,
                transformer_nhead=6,
                transformer_num_layers=6,
                transformer_dim_feedforward=1536,
                transformer_dropout=0.1,
                transformer_activation='gelu',
                transformer_num_feature_levels=2,
                transformer_enc_n_points=4
            )
            print(f"Successfully created CoTr with default parameters")
                
        return model
    
    def get_available_configs(self) -> List[str]:
        """返回可用的 CoTr 配置列表"""
        configs = list(CONFIGS_CoTr.keys())
        print(f"Available CoTr configurations: {configs}")
        return configs

    def set_cotr_config(self, config_name: str):
        """设置 CoTr 配置并更新 plans 文件"""
        if config_name in CONFIGS_CoTr:
            old_config = self.cotr_config_name
            self.cotr_config_name = config_name
            self.arch_init_kwargs['cotr_config'] = config_name
            
            self._update_plans_with_cotr_config(
                self.img_size, 
                config_name, 
                self.configuration_manager.configuration_name
            )
            
            self.print_to_log_file(f"CoTr config changed from {old_config} to {config_name}")
        else:
            available = list(CONFIGS_CoTr.keys())
            raise ValueError(f"Config '{config_name}' not available. Available configs: {available}")

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
            
            arch_init_kwargs = dict(self.configuration_manager.network_arch_init_kwargs)
            arch_init_kwargs['patch_size'] = self.configuration_manager.patch_size
            arch_init_kwargs['cotr_config'] = self.cotr_config_name

            self.print_to_log_file(f"Initializing CoTr with:")
            self.print_to_log_file(f"  - Config: {self.cotr_config_name}")
            self.print_to_log_file(f"  - Patch size: {self.configuration_manager.patch_size}")
            self.print_to_log_file(f"  - Input channels: {self.num_input_channels}")
            self.print_to_log_file(f"  - Output channels: {self.label_manager.num_segmentation_heads}")
            self.print_to_log_file(f"  - Deep supervision: {self.enable_deep_supervision}")

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True
        else:
            raise RuntimeError("Trainer already initialized.")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target = target[0] if len(target) == 1 else torch.stack(target)
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
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target = target[0] if len(target) == 1 else torch.stack(target)
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

            return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if getattr(self, "is_ddp", False) else self.network
        enabled = bool(enabled)
        fn = getattr(mod, "enable_deep_supervision", None)
        if callable(fn):
            fn(enabled)
        elif hasattr(mod, "do_ds"):
            mod.do_ds = enabled

    def _get_deep_supervision_scales(self):
        # CoTr returns 4 maps: full, then three coarser decoder stages.
        if not self.enable_deep_supervision:
            return None
        return [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]]
