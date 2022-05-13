import os
import sys
sys.path.append(os.getcwd())

from libs.modules.ResNetSeq2SeqAttention import *
from libs.modules.Seq2SeqAttention import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 未测试



