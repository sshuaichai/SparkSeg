from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from typing import Union, List, Tuple
import torch
from torch import nn
from nnunetv2.training.network.UNETR import UNETR


class nnUNetTrainerUNETR(nnUNetTrainerNoDeepSupervision):
    def __init__(self,
                 plans: dict,
                 configuration: str,
                 fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.num_output_channels = self.label_manager.num_segmentation_heads
        self.img_size = self.configuration_manager.patch_size

        # 获取原始patch_size并进行调整
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            ## 16 is ViT's fixed patch size
            if (original_patch_size[i] / 16) < 1 or ((original_patch_size[i] / 16) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 16 + 0.5) * 16
            else:
                new_patch_size[i] = original_patch_size[i]
                
        # 更新配置文件中的patch_size
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        
        # 关键：将调整后的patch_size保存到plans文件中
        self._update_plans_with_patch_size(new_patch_size, configuration)
        self.print_to_log_file(f"Patch size changed from {original_patch_size} to {new_patch_size}")
        self.print_to_log_file(f"Patch size {new_patch_size} for UNETR")
        
        # 设置架构参数，确保new_patch_size能传递到build_network_architecture
        self.arch_init_kwargs = {'patch_size': new_patch_size}

        # 参考umamba项目的超参数设置
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        self.grad_scaler = None  # UNETR不使用grad_scaler
        self.num_epochs = 200
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.current_epoch = 0
        self.enable_deep_supervision = False

    
    def _update_plans_with_patch_size(self, new_patch_size: list, configuration: str):
        """仅更新内存中的 plans（会随训练保存到输出目录 plans.json）。不写回预处理目录，避免不同 Trainer 互相覆盖导致混乱。"""
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
        """基于官方BTCV示例优化的搜索空间"""
        return {
            "initial_lr": (1e-5, 5e-4),  # 基于官方示例的较小学习率范围
            "weight_decay": (1e-6, 1e-4),  # 较小的权重衰减范围
            "oversample_foreground_percent": (0.25, 0.4),  # 官方推荐范围
            # 优化器配置 - 基于官方示例偏好AdamW
            "eps": (1e-8, 1e-7),  # 保守的eps设置
            "beta1": (0.9, 0.95),  # 标准范围
            "beta2": (0.999, 0.9999),  # 高beta2有助于稳定性
            # UNETR特定参数 - 基于官方示例
            "dropout_rate": (0.0, 0.1),  # 官方示例使用0.0，允许小幅调整
        }
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Optimizer:
        - AdamW is chosen as it is well-suited for training transformer-based architectures like UNETR.
        - It provides better stability and convergence for models with large numbers of parameters due to its decoupled weight decay.

        Scheduler:
        - CosineAnnealingLR is selected for its ability to gradually reduce the learning rate, mimicking a cosine curve.
        - This approach is particularly effective for transformer models, ensuring smoother convergence and avoiding abrupt changes in learning rate.

        Supports hyperparameter tuning by allowing dynamic adjustment of learning rate, weight decay, and other optimizer parameters.
        """
        # 获取当前超参数，如果在超参数搜寻中被设置
        # Check if hyperparameters have been set during hyperparameter tuning
        if hasattr(self, "hyperparameters"):
            initial_lr = self.hyperparameters.get("initial_lr", self.initial_lr)
            weight_decay = self.hyperparameters.get("weight_decay", self.weight_decay)
            eps = self.hyperparameters.get("eps", 1e-8)
            beta1 = self.hyperparameters.get("beta1", 0.9)
            beta2 = self.hyperparameters.get("beta2", 0.999)
        else:
            initial_lr = self.initial_lr
            weight_decay = self.weight_decay
            eps = 1e-8
            beta1 = 0.9
            beta2 = 0.999

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
            eps=eps,
            betas=(beta1, beta2),
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=initial_lr * 0.01)

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
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

        """
        print(f"arch_init_kwargs: {arch_init_kwargs}")
        # 从arch_init_kwargs中获取patch_size
        img_size = arch_init_kwargs.get('patch_size')      
        spatial_dims = len(img_size)
        print(f"Building UNETR with patch_size: {img_size}, spatial_dims: {spatial_dims}")

        # 参考umamba项目的UNETR参数配置
        model = UNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            img_size=img_size,
            feature_size=16,
            hidden_size=768,  
            mlp_dim=3072,    
            num_heads=12,     
            proj_type="conv", 
            norm_name="instance",
            res_block=True,   
            dropout_rate=0.0, 
            spatial_dims=spatial_dims,
            qkv_bias=False,  
            save_attn=False, 
        )
        
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
                # if we train with regions all segmentation heads predict some kind of foreground. In conventional
                # (softmax training) there needs tobe one output for the background. We are not interested in the
                # background Dice
                # [1:] in order to remove background
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
