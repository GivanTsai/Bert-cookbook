"""
深度学习pytorch训练代码模板
"""

# 01.导入包以及设置随机种子
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 02.以类的方式定义超参数
class argparse():
        pass

args = argparse()
args.epochs, args.learning_rate, args.patience = [30, 0.001, 4]
args.hidden_size, args.input_size= [40, 30]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]

# 03.定义自己的模型
