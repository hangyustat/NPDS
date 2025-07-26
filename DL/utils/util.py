import torch
import random
import numpy as np
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class WeightedAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, res_num, device='cpu', fmt=':f'):
        self.name = name
        self.res_num = res_num
        self.device = device
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = torch.zeros(self.res_num, dtype=torch.float32).to(self.device, non_blocking=True)
        self.count = torch.zeros(self.res_num, dtype=torch.float32).to(self.device, non_blocking=True)

    def update(self, val, total_weights):
        self.val = val
        self.sum += val * total_weights
        self.count += total_weights
        self.avg = (self.sum / self.count).sum() / self.res_num
        self.val = self.val.sum() / self.res_num

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def get_message(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ('\t').join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def meanSquaredError(output, target):
    """
    Computes the Mean Squared Error (MSE) between the output and target.

    Args:
        output (torch.Tensor): Predicted values (batch_size, num_classes or regression values).
        target (torch.Tensor): True values (batch_size, num_classes or regression values).

    Returns:
        torch.Tensor: The computed MSE as a scalar tensor.
    """
    with torch.no_grad():
        # Ensure output and target have the same shape
        if output.shape != target.shape:
            raise ValueError("Output and target must have the same shape.")

        # Compute the squared differences
        squared_diff = (output - target) ** 2

        # Calculate the mean of the squared differences
        mse_value = squared_diff.mean()

    return mse_value
        

def parse_gpus(gpu_ids):
    gpus = gpu_ids.split(',')
    gpu_ids = []
    for g in gpus:
        g_int = int(g)
        if g_int >= 0:
            gpu_ids.append(g_int)
    if not gpu_ids:
        return None
    return gpu_ids


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
