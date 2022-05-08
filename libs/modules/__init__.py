import os
import sys
sys.path.append(os.getcwd())

# from Test import Test
from libs.modules.ResNetLSTM import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 未测试



