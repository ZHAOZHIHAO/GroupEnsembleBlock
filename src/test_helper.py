import torch
import time
from lib.utils import AverageMeter
import torch.nn.functional as F


def kNN(args, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            batchSize = inputs.size(0)
            out_projector, out_predictor = net(inputs)
            features = out_projector
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            out_projector, out_predictor = net(inputs)
            features = out_projector
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

            if args.TEST:
                print('Test [{}/{}]\t'
                      'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                      'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                      'Top1: {:.2f}  Top5: {:.2f}'.format(
                      total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))

    print(top1*100./total)
    return top1/total


IF_PART = False
part_idx_start = 1
part_idx_end = 4
def get_probs(args, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=1):
    if args.dataset == 'CIFAR-100':
        C_coarse= 20
    elif args.dataset == 'ImageNet-C16':
        C_coarse = 16
    else:
        pass
    net.eval()
    print("computing coarse-classes probs")
    net_time = AverageMeter()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    C = trainLabels.max() + 1
    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            # print(pathes[0])
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            out_projector, out_predictor = net(inputs)
            # print("out_knn_train.shape", out_knn_train.shape)
            if IF_PART:
                out_knn_train = out_knn_train[:, part_idx_start * 64:(part_idx_end) * 64]
                out_knn_train = F.normalize(out_knn_train, p=2, dim=-1)
            features = out_projector
            # print(features.shape)
            # features, out_for_pred = net(inputs)
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
        trainLabels_coarse = torch.LongTensor(temploader.dataset.targets).cuda()
        trainloader.dataset.transform = transform_bak

    probs_list = []
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            batchSize = inputs.size(0)
            out_projector, out_predictor, = net(inputs)
            if IF_PART:
                out_knn_train = out_knn_train[:, part_idx_start * 64:(part_idx_end) * 64]
                out_knn_train = F.normalize(out_knn_train, p=2, dim=-1)
            features = out_projector
            # features, out_for_pred = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            # add information about coarse class
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels_coarse.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C_coarse).zero_()
            retrieval_one_hot.scatter_(1, retrieval[:, :K].reshape(-1, 1), 1)
            yd_transform = yd[:, :K].clone().div_(sigma).exp_()
            probs = torch.sum(
                torch.mul(retrieval_one_hot.view(batchSize, -1, C_coarse), yd_transform.view(batchSize, -1, 1)), 1)
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
            probs_list.append(probs)
    return probs_list


def compute_mAP_cond_optimized_CIFAR100(probs_list, epoch, net, lemniscate, trainloader, testloader, K, sigma,
                               recompute_memory=1):
    print("compute mAP")
    net.eval()
    net_time = AverageMeter()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=80, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            out_projector, out_predictor = net(inputs)
            if IF_PART:
                out_knn_train = out_knn_train[:, part_idx_start * 64:(part_idx_end) * 64]
                out_knn_train = F.normalize(out_knn_train, p=2, dim=-1)

            features = out_projector
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
        trainloader.dataset.transform = transform_bak

    average_precision_list = []
    class_num = 20
    subclass_num = 5
    print("ongoing mAP computation, averaged till the current fine class")
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)
            out_projector, out_predictor = net(inputs)

            if IF_PART:
                out_knn_train = out_knn_train[:, part_idx_start * 64:(part_idx_end) * 64]
                out_knn_train = F.normalize(out_knn_train, p=2, dim=-1)
            features = out_projector
            # features, out_for_pred = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            probs = probs_list[batch_idx]
            PROT = 0.05
            for i in range(class_num):  # class num
                dist[:, i * 500 * subclass_num:(i + 1) * 500 * subclass_num] += torch.log(
                    (probs[:, i] + PROT) / (1 - probs[:, i] + PROT)).unsqueeze(-1)

            # compute AP for each test sample
            for i in range(inputs.shape[0]):
                for K in [5000, 10000, 50000]:
                    yd, yi = dist[i:i + 1, :].topk(K, dim=1, largest=True, sorted=True)
                    candidates = trainLabels.view(1, -1)  # .expand(batchSize, -1)
                    retrieval = torch.gather(candidates, 1, yi)
                    if torch.sum(retrieval == targets[i]) == 500:
                        break
                    else:
                        continue
                total = 0
                correct = 0
                precisions = []
                for j in range(retrieval.shape[1]):
                    total += 1
                    if retrieval[0, j] == targets[i]:
                        correct += 1
                        precisions.append(float(correct) / total)

                if len(precisions) != 0:
                    average_precision = sum(precisions) / len(precisions)
                else:
                    average_precision = 0
                average_precision_list.append(average_precision)
            print(batch_idx, sum(average_precision_list) / len(average_precision_list))
    mAP = sum(average_precision_list) / len(average_precision_list)
    return mAP


def compute_mAP_cond_optimized_ImageNetC16(probs_list, epoch, net, lemniscate, trainloader, testloader, K, sigma,
                               recompute_memory=1):
    C_coarse = 16
    subclass_num = 2
    print("compute_mAP_cond_optimized_ImageNetC16")
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=80, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            out_projector, out_predictor = net(inputs)
            if IF_PART:
                out_knn_train = out_knn_train[:, part_idx_start * 64:(part_idx_end) * 64]
                out_knn_train = F.normalize(out_knn_train, p=2, dim=-1)
            features = out_projector
            # features, out_for_pred = net(inputs)
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
        trainloader.dataset.transform = transform_bak

    end = time.time()
    average_precision_list = []
    print("ongoing mAP computation, averaged till the current fine class")
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)
            # targets[...] = batch_idx
            # targets = targets.cuda()
            batchSize = inputs.size(0)
            out_projector, out_predictor = net(inputs)
            if IF_PART:
                out_knn_train = out_knn_train[:, part_idx_start * 64:(part_idx_end) * 64]
                out_knn_train = F.normalize(out_knn_train, p=2, dim=-1)
            features = out_projector
            # features, out_for_pred = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            probs = probs_list[batch_idx]
            dist = torch.mm(features, trainFeatures)
            PROT = 0.05

            _, num_per_fine_class = torch.unique(torch.tensor(trainloader.dataset.targets), return_counts=True)
            # print(num_per_fine_class)
            num_per_coarse_class = [sum(num_per_fine_class[i:i + subclass_num]) for i in
                                    range(0, C_coarse * subclass_num, subclass_num)]
            # print(num_per_coarse_class)
            # print([i for i in range(0, C_coarse*subclass_num, subclass_num)])
            num_per_coarse_class = [i.item() for i in num_per_coarse_class]  # from list of tensor to list of int
            # print(num_per_coarse_class)
            # print(trainloader.dataset.targets)
            # print(num_per_class)
            for i in range(C_coarse):  # class num
                start = sum(num_per_coarse_class[:i])
                end = sum(num_per_coarse_class[:i + 1])
                # print(dist[:, start:end].shape)
                # print(probs[:, i].shape)
                dist[:, start:end] += torch.log((probs[:, i] + PROT) / (1 - probs[:, i] + PROT)).unsqueeze(-1)

            # print(retrieval.shape)
            # compute AP for each test sample
            for i in range(inputs.shape[0]):
                for K in [10000, 40516]:  # total training examples 40517
                    yd, yi = dist[i:i + 1, :].topk(K, dim=1, largest=True, sorted=True)
                    candidates = trainLabels.view(1, -1)  # .expand(batchSize, -1)
                    retrieval = torch.gather(candidates, 1, yi)
                    if torch.sum(retrieval == targets[i]) == 500:
                        break
                    else:
                        continue
                # print("retrieval.shape", retrieval.shape)
                total = 0
                correct = 0
                precisions = []
                for j in range(retrieval.shape[1]):
                    total += 1
                    if retrieval[0, j] == targets[i]:
                        correct += 1
                        precisions.append(float(correct) / total)
                    # if len(precisions) == 500: # number of test images per subclass
                    #     break#

                # print(len())
                if len(precisions) != 0:
                    average_precision = sum(precisions) / len(precisions)
                else:
                    average_precision = 0
                average_precision_list.append(average_precision)
            print(batch_idx, sum(average_precision_list) / len(average_precision_list))
    mAP = sum(average_precision_list) / len(average_precision_list)
    return mAP