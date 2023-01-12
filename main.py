import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from data.build_dataset import build_dataset
from models.build_model import build_model
from utils.evaluate import evaluate
from opt import args
from utils.loss import BceDiceLoss
import math


def DeepSupSeg(pred, gt):
    
    d0, d1, d2, d3, d4 = pred  ##
    
    criterion = BceDiceLoss()
   
    loss0 = criterion(d0, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True) ##
    loss1 = criterion(d1, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion(d2, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion(d3, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gt)
   
    return loss0 + loss1 + loss2 + loss3 + loss4


def DeepSupInp(pred, gt, mask):

    criterion = nn.L1Loss()
    loss = 0
    for i in range(len(pred)):
       select_pred = torch.masked_select(pred[i], mask[i]>0.5)
       select_target = torch.masked_select(gt, mask[i]>0.5)
       loss += criterion(select_pred, select_target)
       gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    return loss


def SupInp(pred, gt, mask):

    criterion = nn.L1Loss()
    
    select_pred = torch.masked_select(pred, mask>0.5)
    select_target = torch.masked_select(gt, mask>0.5)
    loss = criterion(select_pred, select_target)
    
    return loss


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1-float(iter)/max_iter)**power)


def adjust_lr_rate(argsimizer, iter, total_batch):
    lr = lr_poly(args.lr, iter, args.nEpoch*total_batch, args.power)
    argsimizer.param_groups[0]['lr'] = lr
    return lr


def train():

    
    """load data"""
    train_l_data, train_u_data, valid_data, test_data = build_dataset(args)


    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    #train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_total_batch = int(len(test_data) / 1)
    """load model"""
    model = build_model(args)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    # train
    print('\n---------------------------------')
    print('Start training')
    print('---------------------------------\n')

    F1_best, F1_test_best = 0, 0

    for epoch in range(args.nEpoch):
        model.train()
      
        print("Epoch: {}".format(epoch))
        total_batch = math.ceil(len(train_l_data) / args.batch_size)
        bar = tqdm(enumerate(train_l_dataloader), total=total_batch)
        for batch_id, data_l in bar:
            #data_l, data_u = next(loader)
            
            #total_batch = len(train_u_dataloader)
            #total_batch = len(train_l_dataloader)
            itr = total_batch * epoch + batch_id

            img_l, gt = data_l['image'], data_l['label']
           
            if args.GPUs:
                img_l = img_l.cuda()
                gt = gt.cuda()
              

            optim.zero_grad()

            pred_l = model(img_l)
            mask_l = pred_l[:5]
            inp_l = pred_l[5:]
            loss_l_seg = DeepSupSeg(mask_l, gt) 
           
            loss_l =  loss_l_seg
            
           
            loss = loss_l
            loss.backward()
            
            optim.step()

            
            adjust_lr_rate(optim, itr, total_batch)

        if valid_sign == True:
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean = evaluate(model, valid_dataloader, val_total_batch, args)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean))

            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "checkpoint/exp" + str(args.expID) + "/ck_%.4f.pth" % F1)

        else:
            recall_test, specificity_test, precision_test, F1_test, F2_test, \
            ACC_overall_test, IoU_poly_test, IoU_bg_test, IoU_mean_test = evaluate(model, test_dataloader, test_total_batch, args)
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' \
                % (recall_test, specificity_test, precision_test, F1_test, F2_test, ACC_overall_test, IoU_poly_test, IoU_bg_test, IoU_mean_test))

            if (F1_test > F1_test_best):
                F1_test_best = F1_test
                torch.save(model.state_dict(), args.root + "checkpoint/exp" + str(args.expID) + "/ck_%.4f.pth" % F1_test)





def train_semi():

    
    """load data"""
    train_l_data, train_u_data, valid_data, test_data = build_dataset(args)


    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        val_total_batch = int(len(valid_data) / 1)

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_total_batch = int(len(test_data) / 1)
    """load model"""
    model = build_model(args)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)

    # train
    print('\n---------------------------------')
    print('Start training')
    print('---------------------------------\n')

    F1_best, F1_test_best = 0, 0

    for epoch in range(args.nEpoch):
        model.train()
      
        print("Epoch: {}".format(epoch))

        loader = iter(zip(cycle(train_l_dataloader), train_u_dataloader))
        #loader = iter(zip(train_l_dataloader, cycle(train_u_dataloader)))
        bar = tqdm(range(len(train_u_dataloader)))
        #bar = tqdm(range(len(train_l_dataloader)))
        for batch_id in bar:
            data_l, data_u = next(loader)
            
            total_batch = len(train_u_dataloader)
            #total_batch = len(train_l_dataloader)
            itr = total_batch * epoch + batch_id

            img_l, gt = data_l['image'], data_l['label']
            img_u = data_u

            if args.GPUs:
                img_l = img_l.cuda()
                gt = gt.cuda()
                img_u = img_u.cuda()

            optim.zero_grad()

            pred_l = model(img_l)
            mask_l = pred_l[:5]
            inp_l = pred_l[5:]
            loss_l_seg = DeepSupSeg(mask_l, gt) 
           
            loss_l =  loss_l_seg
            
            pred_u = model(img_u)
            mask_u = pred_u[:5]
            inp_u = pred_u[5:]
            loss_u = DeepSupInp(inp_u, img_u.detach(), [m.detach() for m in mask_u])
            
           
            loss = 2 * loss_l + loss_u
            loss.backward()
            
            optim.step()

            
            adjust_lr_rate(optim, itr, total_batch)

        if valid_sign == True:
            recall, specificity, precision, F1, F2, \
            ACC_overall, IoU_poly, IoU_bg, IoU_mean = evaluate(model, valid_dataloader, val_total_batch, args)

            print("Valid Result:")
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean))

            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "checkpoint/exp" + str(args.expID) + "/ck_%.4f.pth" % F1)

        else:
            recall_test, specificity_test, precision_test, F1_test, F2_test, \
            ACC_overall_test, IoU_poly_test, IoU_bg_test, IoU_mean_test = evaluate(model, test_dataloader, test_total_batch, args)
            print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' \
                % (recall_test, specificity_test, precision_test, F1_test, F2_test, ACC_overall_test, IoU_poly_test, IoU_bg_test, IoU_mean_test))

            if (F1_test > F1_test_best):
                F1_test_best = F1_test
                torch.save(model.state_dict(), args.root + "checkpoint/exp" + str(args.expID) + "/ck_%.4f.pth" % F1_test)



def test():
  
    print('loading data......')
    test_data = build_dataset(args)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    total_batch = int(len(test_data) / 1)
    model = build_model(args)

    model.eval()

    recall, specificity, precision, F1, F2, \
    ACC_overall, IoU_poly, IoU_bg, IoU_mean = evaluate(model, test_dataloader, total_batch, args)

    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' \
        % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean))

    return recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean


if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUs
    if args.manner == 'full':
        print('---{}-Seg Train---'.format(args.dataset))
        train()
    elif args.manner =='semi':
        print('---{}-seg Semi-Train--'.format(args.dataset))
        train_semi()
    elif args.manner == 'test':
        print('---{}-Seg Test---'.format(args.dataset))
        test()
    print('Done')



