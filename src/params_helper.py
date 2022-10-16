import torch


# https://www.cnblogs.com/quarryman/p/pytorch_weight_decay.html
def split_parameters(module):
    predictor_params_decay = []
    predictor_params_no_decay = []
    params_decay = []
    params_no_decay = []
    for name, m in module.named_modules():
        print(name)
        if name.startswith("module.predictor"):
            print("larger leraning rate for ", name)
            if isinstance(m, torch.nn.Linear):
                predictor_params_decay.append(m.weight)
                if m.bias is not None:
                    predictor_params_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                predictor_params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                predictor_params_decay.extend([*m.parameters()])
            else:
                print("Missing layer! ", name)
        elif isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    return params_decay, params_no_decay, predictor_params_decay, predictor_params_no_decay


def get_params(net, args):
    params_decay, params_no_decay, predictor_params_decay, predictor_params_no_decay = split_parameters(net)
    if args.dataset == 'CIFAR-100':
        WEIGHT_DECAY = 5e-4
    elif args.dataset == 'ImageNet-C16':
        WEIGHT_DECAY = 1e-4
    else:
        pass
    print("WEIGHT_DECAY ", WEIGHT_DECAY)
    params = [{'params': params_no_decay, 'weight_decay': 0},  # 0 in BYOL
              {'params': params_decay, 'weight_decay': WEIGHT_DECAY},
              {'params': predictor_params_no_decay, 'weight_decay': 0, 'lr': args.lr},  # 0 in BYOL
              {'params': predictor_params_decay, 'weight_decay': WEIGHT_DECAY, 'lr': args.lr}]
    return params


def print_parameters_info(parameters):
    for k, param in enumerate(parameters):
        print('[{}/{}] {}'.format(k+1, len(parameters), param.shape))