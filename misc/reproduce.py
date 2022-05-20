def set_arguments(args):
    args.n_data = 500
    if args.dataset != 'imagenet':
        args.net_type = 'convnet'
        args.depth = 3
        args.niter = 2000
        if args.dataset == 'cifar10':
            args.metric = 'mse'
            args.lr_img = 5e-3
            args.n_data = 2000
        elif args.dataset == 'svhn':
            args.metric = 'mse'
            args.lr_img = 5e-3
        elif args.dataset == 'mnist':
            args.metric = 'l1'
            args.lr_img = 1e-4
            if args.factor > 1:
                args.aug_type = 'color_crop'
                args.mixup_net = 'vanilla'
                args.mixup = 'vanilla'
                args.dsa_strategy = 'color_crop_scale_rotate'
        elif args.dataset == 'fashion':
            args.metric = 'l1'
            args.lr_img = 1e-4
        else:
            raise AssertionError("Not supported dataset!")
    else:
        args.net_type = 'resnet_ap'
        args.depth = 10
        args.niter = 500
        if args.nclass == 10:
            args.metric = 'l1'
            args.lr_img = 3e-3
            args.early = 10
        elif args.nclass == 100:
            args.metric = 'l1'
            args.lr_img = 1e-3

        if args.factor >= 3 and args.ipc >= 20:
            args.decode_type = 'bound'

    log = f"Arguments are loaded!"
    log += f", net: {args.net_type}-{args.depth}"
    log += f", metric: {args.metric}"
    log += f", lr_img: {args.lr_img}"
    log += f", n_data: {args.n_data}"
    if args.early > 0:
        log += f", early: {args.early}"
    if args.decode_type != 'single':
        log += f", decode: {args.decode_type}"
    print(log)

    return args
