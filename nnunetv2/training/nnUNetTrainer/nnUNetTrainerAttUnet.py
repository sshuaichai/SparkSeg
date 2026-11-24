from typing import Union, List, Tuple
from torch import nn, autocast
from nnunetv2.training.network.AttentionUnet import AttentionUnet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
import torch
from nnunetv2.training.lr_scheduler.warmup_cosine import LinearWarmupCosineAnnealingLR
from torch.optim import Adam
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class nnUNetTrainerAttUnet(nnUNetTrainerNoDeepSupervision):
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
        img_size = self.configuration_manager.patch_size

        # 将调整后的patch_size保存到plans文件中
        self._update_plans_with_patch_size(img_size , configuration)
        self.print_to_log_file(f"Patch size {img_size} for AttentionUnet.")
        self.arch_init_kwargs = {'patch_size': img_size}

        self.grad_scaler = None
        self.initial_lr = 1e-4  # 常见医学分割配置 (Otkay et al. Attention U-Net)
        self.weight_decay = 1e-5
        self.warmup_epochs = 5  # 短 warmup，缓解 Adam 前期梯度不稳定
        self.num_epochs = 200
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.current_epoch = 0
        self.enable_deep_supervision = False

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
        """Attention U-Net 常见医学分割超参数搜索空间"""
        return {
            "initial_lr": (5e-5, 5e-4),
            "weight_decay": (1e-6, 1e-4),
            "warmup_epochs": (2, 10),
            "oversample_foreground_percent": (0.25, 0.4),
            "dropout": (0.0, 0.2),
        }
    
    def configure_optimizers(self):
        """
        Attention U-Net 常见医学分割配置: Adam + WarmupCosineAnnealingLR
        (Otkay et al. 论文无官方训练配置，常用 lr=1e-4, weight_decay=1e-5)
        """
        if hasattr(self, "hyperparameters"):
            initial_lr = self.hyperparameters.get("initial_lr", self.initial_lr)
            weight_decay = self.hyperparameters.get("weight_decay", self.weight_decay)
            warmup_epochs = self.hyperparameters.get("warmup_epochs", self.warmup_epochs)
        else:
            initial_lr = self.initial_lr
            weight_decay = self.weight_decay
            warmup_epochs = self.warmup_epochs

        optimizer = Adam(
            self.network.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
        )

        eta_min = initial_lr * 0.01
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=self.num_epochs,
            eta_min=eta_min,
        )
        self.print_to_log_file(
            f"AttUnet config: Adam lr={initial_lr}, wd={weight_decay}, "
            f"WarmupCosine(warmup={warmup_epochs}, max={self.num_epochs}, eta_min={eta_min:.2e})"
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
        Attention Unet based on
        Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
        https://arxiv.org/abs/1804.03999
        """
        # 从arch_init_kwargs中获取patch_size
        patch_size = arch_init_kwargs.get('patch_size')
        if patch_size is None:
            raise ValueError("patch_size is required in arch_init_kwargs")
            
        spatial_dims = len(patch_size)
        print(f"Building AttentionUnet with patch_size: {patch_size}, spatial_dims: {spatial_dims}")

        # 获取网络参数
        channels = arch_init_kwargs.get("channels", (32, 64, 128, 256, 512))
        strides = arch_init_kwargs.get("strides", None)

        # 确保channels至少有2个元素
        if len(channels) < 2:
            channels = (32, 64, 128, 256, 512)

        # 如果strides为None或长度不正确，则根据channels生成正确长度的strides
        if strides is None or len(strides) != len(channels) - 1:
            strides = (2,) * (len(channels) - 1)
            print(f"Auto-generating strides of length {len(strides)}: {strides}")

        kernel_size = arch_init_kwargs.get("kernel_size", 3)
        up_kernel_size = arch_init_kwargs.get("up_kernel_size", 3)
        dropout = arch_init_kwargs.get("dropout", 0.2)

        try:
            model = AttentionUnet(
                spatial_dims=spatial_dims,
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                channels=channels,
                strides=strides,
                kernel_size=kernel_size,
                up_kernel_size=up_kernel_size,
                dropout=dropout
            )
        except Exception as e:
            print(f"Error creating AttentionUnet: {str(e)}")
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
       
        self.enable_deep_supervision = False
        
        # 如果网络有相关属性，设置为False
        if hasattr(self.network, 'decoder') and hasattr(self.network.decoder, 'deep_supervision'):
            self.network.decoder.deep_supervision = False
        
        if hasattr(self.network, 'enable_deep_supervision'):
            self.network.enable_deep_supervision(False)