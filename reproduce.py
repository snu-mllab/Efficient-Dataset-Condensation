def set_arguments(args):
    if args.dataset == 'cifar10':
        pass
    elif args.dataset == 'imagenet':
        if args.nclass == 10:
            pass
        elif args.nclass == 100:
            pass
    elif args.dataset == 'svhn':
        pass
    elif args.dataset == 'mnist':
        pass
    elif args.dataset == 'fashion':
        pass
    else:
        raise AssertionError("Not supported dataset!")

    return args
