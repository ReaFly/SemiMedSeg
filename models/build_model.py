import torch
import models
import os


def build_model(args):
    model = getattr(models, args.model)(args.nclasses)
    # model = nn.DataParallel(model)
    if args.GPUs:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    if args.load_ckpt is not None:

        model_dict = model.state_dict()
        load_ckpt_path = os.path.join(args.root, "checkpoint/exp" + str(args.expID), args.load_ckpt + '.pth')
        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        checkpoint = torch.load(load_ckpt_path)
        new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print('Done')

    return model
