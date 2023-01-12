import os
import torch
from .save_img import save_binary_img, save_img
from tqdm import tqdm


def evaluate(model, dataloader, total_batch, args):

    model.eval()

    recall = 0
    specificity = 0
    precision = 0
    F1 = 0
    F2 = 0
    ACC_overall = 0
    IoU_poly = 0
    IoU_bg = 0
    IoU_mean = 0

    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']
            inp = img.clone().detach()
            target = gt.clone().detach()
            if args.GPUs:
                inp = inp.cuda()
                target = target.cuda()

            output = model(inp)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate_batch(output, target, args, i, inp)
            recall += _recall.item()
            specificity += _specificity.item()
            precision += _precision.item()
            F1 += _F1.item()
            F2 += _F2.item()
            ACC_overall += _ACC_overall.item()
            IoU_poly += _IoU_poly.item()
            IoU_bg += _IoU_bg.item()
            IoU_mean += _IoU_mean.item()
    recall /= total_batch
    specificity /= total_batch
    precision /= total_batch
    F1 /= total_batch
    F2 /= total_batch
    ACC_overall /= total_batch
    IoU_poly /= total_batch
    IoU_bg /= total_batch
    IoU_mean /= total_batch
    return recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean


def evaluate_batch(output, gt, args, iid=None, img=None):
    pred = output[0]

    pred_binary = (pred >= 0.5).float()

    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()

    gt_binary_inverse = (gt_binary == 0).float()

    if args.save_pred == True:
        inpimg = output[-5]
        #cut_ori = output[-1]
        save_img(img, iid, 'img')
        save_img(inpimg, iid, 'inp')
        #save_img(cut_ori, iid, 'cut')
        save_binary_img(pred_binary, iid, 'pred')
        save_binary_img(gt_binary, iid, 'gt')


    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.Tensor([1]).cuda()

    # recall
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    Specificity = TN / (TN + FP)

    # Precision or positive predictive value
    Precision = TP / (TP + FP)

    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)

    # F2 score
    F2 = 5 * Precision * Recall / (4 * Precision + Recall)
    
    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)

    # IoU for poly
    IoU_poly = TP / (TP + FP + FN)

    # IoU for background
    IoU_bg = TN / (TN + FP + FN)

    # mean IoU
    IoU_mean = (IoU_poly + IoU_bg) / 2.0

    return Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean
