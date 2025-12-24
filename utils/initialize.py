import random
import torch
import numpy as np


def initialize(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.enabled = True#
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True