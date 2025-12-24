import torch
import torch.nn as nn
import torch.nn.functional as F


#この損失関数はInception-typeである。
def crossentropy(pred, gold):
    return F.cross_entropy(pred, gold,reduction='none')


# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1,train=True):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
      if self.train:
        return self._compute_losses(x, target).mean()
      else:
        return self._compute_losses(x, target)