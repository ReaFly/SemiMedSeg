import argparse
import os


parse = argparse.ArgumentParser(description='PyTorch Semi-Medical-Seg Implement')

"-------------------GPU option----------------------------"
parse.add_argument('--GPUs', type=str, default='0')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/data/ruifei/SemiMedSeg/')
parse.add_argument('--dataset', type=str, default='polyp', choices=['polyp', 'skin', 'optic'])
parse.add_argument('--ratio', type=int, default=10)
parse.add_argument('--polyp', type=str, default='data_polyp')
parse.add_argument('--skin', type=str, default='data_skin')
parse.add_argument('--optic', type=str, default='data_optic')



"-------------------training option-----------------------"
parse.add_argument('--manner', type=str, default='full', choices=['full', 'semi', 'test'])
parse.add_argument('--mode', type=str, default='train')
parse.add_argument('--nEpoch', type=int, default=80)
parse.add_argument('--batch_size', type=float, default=4)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--load_ckpt', type=str, default=None)
parse.add_argument('--model', type=str, default='MyModel')
parse.add_argument('--expID', type=int, default=0)

"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--power',type=float, default=0.9)
parse.add_argument('--betas', default=(0.9, 0.999))
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--eps', type=float, default=1e-8)
parse.add_argument('--mt', type=float, default=0.9)

parse.add_argument('--nclasses', type=int, default=1)
parse.add_argument('--save_pred', default=False, action='store_true')

args = parse.parse_args()
