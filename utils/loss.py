import torch.nn as nn


# BCE loss
class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight, size_average)

    def forward(self, pred, target):
        sizep = pred.size(0)
        sizet = target.size(0)
        pred_flat = pred.view(sizep, -1)
        target_flat = target.view(sizet, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss

#Dice loss

class DiceLoss(nn.Module):
    def __init__(self,size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = target.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/ (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

# BCE + Dice  loss
class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss(size_average)

    def forward(self, pred, target):
        bceloss = self.bce(pred,target)
        diceloss = self.dice(pred,target)

        #w = 0.6
        loss = diceloss + bceloss

        return loss

