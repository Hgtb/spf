import os
import sys
sys.path.append(os.getcwd())

from libs.modules.ResNetLSTM import *
from libs.modules.soft_dtw_cuda import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 未测试



