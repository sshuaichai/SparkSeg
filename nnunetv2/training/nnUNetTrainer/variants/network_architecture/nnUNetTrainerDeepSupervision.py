from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # 先调用父类的__init__
        super().__init__(plans, configuration, fold, dataset_json, device)
        # 然后设置self.enable_deep_supervision = False
        # 这个设置会覆盖父类中的任何值
        self.enable_deep_supervision = True
