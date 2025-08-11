"""
设置随机种子
"""

import numpy as np
import os
import torch
import random

# 对所有模块使用统一的随机种子
def random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # 支持GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False