from libs.modules.includes import *
from libs.modules.soft_dtw_cuda import SoftDTW
from libs.modules.thirdPart import dilate_loss


def loss_function(loss_function_name: str):
    """The return needs to be instantiated externally."""
    loss_function_name = loss_function_name.lower()
    loss_function_dic = {
        "mse": torch.nn.MSELoss,
        "mseloss": torch.nn.MSELoss,

        "softdtw": SoftDTW,
        "softdtwloss": SoftDTW,

        "l1": torch.nn.L1Loss,
        "l1loss": torch.nn.L1Loss,

        "smoothl1": torch.nn.SmoothL1Loss,
        "smoothl1loss": torch.nn.SmoothL1Loss,
    }
    if loss_function_name not in loss_function_dic:
        raise Exception(f"Cannot find loss function : {loss_function_name}.")
    return loss_function_dic[loss_function_name]



