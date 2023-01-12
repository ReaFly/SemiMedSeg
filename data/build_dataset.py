from .polyp import PolypDataSet
from .skin import SkinDataSet
from .optic import OpticDataSet


def build_dataset(args):
    if args.manner == 'test':
        if args.dataset == 'polyp':
            test_data = PolypDataSet(args.root, args.polyp, mode='test')
        elif args.dataset == 'skin':
            test_data = SkinDataSet(args.root, args.skin, mode='test')
        elif args.dataset == 'optic':
            test_data = OpticDataSet(args.root, args.optic, mode='test')
        return test_data
    else:
        if args.dataset == 'polyp':
            train_data = PolypDataSet(args.root, args.polyp, mode='train', ratio=args.ratio, sign='label')
            valid_data = PolypDataSet(args.root, args.polyp, mode='valid')
            test_data = PolypDataSet(args.root, args.polyp, mode='test')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = PolypDataSet(args.root, args.polyp, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'skin':
            train_data = SkinDataSet(args.root, args.skin, mode='train', ratio=args.ratio, sign='label')
            valid_data = None
            test_data = SkinDataSet(args.root, args.skin, mode='test')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = SkinDataSet(args.root, args.skin, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'optic':
            train_data = OpticDataSet(args.root, args.optic, mode='train', ratio=args.ratio, sign='label')
            valid_data = None
            test_data = OpticDataSet(args.root, args.optic, mode='test')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = OpticDataSet(args.root, args.optic, mode='train', ratio=args.ratio, sign='unlabel')

        return train_data, train_u_data, valid_data, test_data

