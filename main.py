import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import os

from lib.LinearAverage import LinearAverage
from lib.NCA import NCACrossEntropy
from src.network import ResNet50CIFAR100, ResNet50ImageNet
from src.test_helper import kNN, get_probs, compute_mAP_cond_optimized_ImageNetC16, compute_mAP_cond_optimized_CIFAR100
from src.main_helper import trainCifar100, trainImageNetC16, adjust_learning_rate_warmup, adjust_memory_update_rate
from src.params_helper import get_params
from datasets.dataset_helper import get_cifar100_train_test_set, get_ImageNetC16_train_test_set


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR-100', type=str)
parser.add_argument('--save_checkpoint_folder', default='./checkpoints', type=str)
parser.add_argument('--load_checkpoint_path', default='./checkpoints/epoch313.pth', type=str)
parser.add_argument('--TRAIN', action='store_true', default=False)
parser.add_argument('--TEST', action='store_true', default=False)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--epochs', default=314, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='init learning rate for cosine scheduler')
parser.add_argument('--embedding_dim', default=256, type=int, help='length of the vector used for retrieval')
parser.add_argument('--temperature', default=0.05, type=float, help='temperature for NCA as in the paper')
parser.add_argument('--memory_momentum', default=0.5, type=float, help='momentum to update the embeddings')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if not os.path.exists(args.save_checkpoint_folder):
    os.makedirs(args.save_checkpoint_folder)

if args.dataset == 'CIFAR-100':
    model = ResNet50CIFAR100
    train = trainCifar100
    compute_mAP = compute_mAP_cond_optimized_CIFAR100
elif args.dataset == 'ImageNet-C16':
    model = ResNet50ImageNet
    train = trainImageNetC16
    compute_mAP = compute_mAP_cond_optimized_ImageNetC16
else:
    pass


if args.TRAIN:
    print('==> Building model..')
    print("Using ResNet50")
    net = model(args)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    if args.dataset == 'CIFAR-100':
        trainset, testset = get_cifar100_train_test_set(args)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True,
                                                  num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        ndata = trainset.__len__()
        stored_embeddings = LinearAverage(args.embedding_dim, ndata, args.temperature, args.memory_momentum)
        criterion = NCACrossEntropy(torch.LongTensor(trainloader.dataset.targets))
        if use_cuda:
            stored_embeddings.cuda()
            criterion.cuda()
    elif args.dataset == 'ImageNet-C16':
        trainset, testset = get_ImageNetC16_train_test_set(args)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
                                                  pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        ndata = trainset.__len__()
        stored_embeddings = LinearAverage(args.embedding_dim, ndata, args.temperature, args.memory_momentum)
        criterion = NCACrossEntropy(torch.LongTensor(trainloader.dataset.targets))
        if use_cuda:
            stored_embeddings.cuda()
            criterion.cuda()
    else:
        pass

    params = get_params(net, args)
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    for epoch in range(0, args.epochs):
        # learning rate
        total_warm_up_epochs = 5
        print('\nEpoch: %d' % epoch)
        if epoch < total_warm_up_epochs:
            adjust_learning_rate_warmup(optimizer, total_warm_up_epochs, epoch, args.lr)
        else:
            scheduler.step()
        print("Learning rate{}".format(optimizer.param_groups[0]['lr']))
        # adjust memory rate for embeddings
        adjust_memory_update_rate(stored_embeddings, epoch)
        # train
        train(args, epoch, net, optimizer, trainloader, criterion, stored_embeddings, use_cuda)

        # kNN acc
        acc = kNN(args, net, stored_embeddings, trainloader, testloader, 30, args.temperature)

        save = False
        # save
        if acc > best_acc:
            best_acc = acc
            save = True
            filename = args.save_checkpoint_folder + '/best.pth'
        elif epoch in [73, 153, 313] or (epoch + 1) % 30 == 0:
            save = True
            filename = args.save_checkpoint_folder + '/epoch' + str(epoch) + '.pth'
        else:
            pass

        if save:
            print('Saving..')
            state = {
                'net': net.module if use_cuda else net,
                'stored_embeddings': stored_embeddings,
                'acc': 0,
                'epoch': epoch,
                'masks': net.module.ensembleBlock.mask_pos if use_cuda else net.ensembleBlock.mask_pos,
                'float_mask': net.module.ensembleBlock.float_mask if use_cuda else net.ensembleBlock.float_mask,
            }
            torch.save(state, filename)

        print('best accuracy: {:.2f}'.format(best_acc * 100))


# Model
if args.TEST:
    from torchvision import transforms

    # Load checkpoint.
    print('==> Loading from checkpoint..')
    checkpoint = torch.load(args.load_checkpoint_path)
    net = model(args)
    net.load_state_dict(checkpoint['net'].state_dict())
    net.ensembleBlock.mask_pos = checkpoint['masks']
    net.ensembleBlock.float_mask = checkpoint['float_mask']
    stored_embeddings = checkpoint['stored_embeddings']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("the loaded model has been trained for # epoch", start_epoch)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = False
        stored_embeddings.cuda()

    epoch = -1
    if args.dataset == 'CIFAR-100':
        from datasets.folder import ImageFolderInstanceTest
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset_coarse = ImageFolderInstanceTest(root='./data/cifar100_coarse/train', transform=transform_test)
        testset_coarse = ImageFolderInstanceTest(root='./data/cifar100_coarse/test', transform=transform_test)
        trainset_fine = ImageFolderInstanceTest(root='./data/cifar100_fine/train', transform=transform_test)
        testset_fine = ImageFolderInstanceTest(root='./data/cifar100_fine/test', transform=transform_test)
    elif args.dataset == 'ImageNet-C16':
        from datasets.folder_imagenet import ImageFolderInstanceTest
        transform_imagenet_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        trainset_coarse = ImageFolderInstanceTest(root='./data/imagenetC16_coarse/train', transform=transform_imagenet_test)
        testset_coarse = ImageFolderInstanceTest(root='./data/imagenetC16_coarse/test', transform=transform_imagenet_test)
        trainset_fine = ImageFolderInstanceTest(root='./data/imagenetC16_fine/train', transform=transform_imagenet_test)
        testset_fine = ImageFolderInstanceTest(root='./data/imagenetC16_fine/test', transform=transform_imagenet_test)
    else:
        pass

    ndata = trainset_coarse.__len__()
    trainloader_coarse = torch.utils.data.DataLoader(trainset_coarse, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    testloader_coarse = torch.utils.data.DataLoader(testset_coarse, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    embeddings_coarse = LinearAverage(args.embedding_dim, ndata, args.temperature, args.memory_momentum)
    embeddings_coarse.to('cuda')

    trainloader_fine = torch.utils.data.DataLoader(trainset_fine, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    testloader_fine = torch.utils.data.DataLoader(testset_fine, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    embeddings_fine = LinearAverage(args.embedding_dim, ndata, args.temperature, args.memory_momentum)
    embeddings_fine.to('cuda')

    # for kNN acc
    acc = kNN(args, net, embeddings_fine, trainloader_fine, testloader_fine, 20, args.temperature, recompute_memory=True)

    # for mAP
    probs_list = get_probs(args, net, embeddings_coarse, trainloader_coarse, testloader_coarse, 20, args.temperature)
    mAP = compute_mAP(probs_list, epoch, net, embeddings_fine, trainloader_fine, testloader_fine, 50000, args.temperature)

