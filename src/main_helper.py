import torch.nn.functional as F
import time
from lib.utils import AverageMeter


def adjust_learning_rate_warmup(optimizer, total_warm_up_epochs, epoch, start_lr):
    if epoch == 0:
        lr = 0.001
    else:
        lr = start_lr / total_warm_up_epochs * (epoch+1)
    print("warm start", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_memory_update_rate(lemniscate, epoch):
    if epoch >= 120:
        lemniscate.params[1] = 0.8
    if epoch >= 140:
        lemniscate.params[1] = 0.9


# https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py
def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()
    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


def compute_ssl_loss(ps, zs):
    num = len(ps)
    ssl_loss = 0
    count = 0
    for i in range(num-1):
        for j in range(i+1, num):
            ssl_loss += (D(ps[i], zs[j]) + D(ps[j], zs[i])) / 2
            count += 1
    ssl_loss /= count
    return ssl_loss


# Training
def trainCifar100(args, epoch, net, optimizer, trainloader, criterion, lemniscate, use_cuda):
    train_sncaloss = AverageMeter()
    train_sllloss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs1, inputs2, inputs3, inputs4, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        if use_cuda:
            targets, indexes = targets.cuda(), indexes.cuda()
            inputs1, inputs2, inputs3, inputs4 = inputs1.cuda(), inputs2.cuda(), inputs3.cuda(), inputs4.cuda()
        optimizer.zero_grad()

        out_projector1, out_predictor1 = net(inputs1)
        out_projector2, out_predictor2 = net(inputs2)
        out_projector3, out_predictor3 = net(inputs3)
        out_projector4, out_predictor4 = net(inputs4)

        snca_loss1 = criterion(lemniscate(out_projector1, indexes), indexes)
        snca_loss2 = criterion(lemniscate(out_projector2, indexes), indexes)
        snca_loss3 = criterion(lemniscate(out_projector3, indexes), indexes)
        snca_loss4 = criterion(lemniscate(out_projector4, indexes), indexes)
        snca_loss = (snca_loss1 + snca_loss2 + snca_loss3 + snca_loss4) / 4

        ssl_loss = compute_ssl_loss([out_predictor1, out_predictor2, out_predictor3, out_predictor4],
                                    [out_projector1, out_projector2, out_projector3, out_projector4])
        # check if collapse
        # print("", D(out_predictor1[1:], out_projector1[:-1]))
        # print("", D(out_predictor1[1:], out_projector2[:-1]))

        SSL_WEIGHT = 1.0
        loss = snca_loss + SSL_WEIGHT * ssl_loss

        loss.backward()
        optimizer.step()

        train_sncaloss.update(snca_loss.item(), inputs1.size(0))
        train_sllloss.update(ssl_loss.item(), inputs1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 60 == 0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Snca: {train_sncaloss.val:.4f} ({train_sncaloss.avg:.4f}) '
                  'Ssl: {train_sllloss.val:.4f} ({train_sllloss.avg:.4f}) '
                  .format(
                  epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time,
                  train_sllloss=train_sllloss, train_sncaloss=train_sncaloss))

    return


def trainImageNetC16(args, epoch, net, optimizer, trainloader, criterion, lemniscate, use_cuda):
    train_sncaloss = AverageMeter()
    train_sllloss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs1, inputs2, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        if use_cuda:
            targets, indexes = targets.cuda(), indexes.cuda()
            inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()
        optimizer.zero_grad()

        out_projector1, out_predictor1 = net(inputs1)
        out_projector2, out_predictor2 = net(inputs2)

        snca_loss1 = criterion(lemniscate(out_projector1, indexes), indexes)
        snca_loss2 = criterion(lemniscate(out_projector2, indexes), indexes)
        snca_loss = (snca_loss1 + snca_loss2) / 2

        ssl_loss = compute_ssl_loss([out_predictor1, out_predictor2],
                                    [out_projector1, out_projector2])

        SSL_WEIGHT = 1.0
        loss = snca_loss + SSL_WEIGHT * ssl_loss

        loss.backward()
        optimizer.step()

        train_sncaloss.update(snca_loss.item(), inputs1.size(0))
        train_sllloss.update(ssl_loss.item(), inputs1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 60 == 0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Snca: {train_sncaloss.val:.4f} ({train_sncaloss.avg:.4f}) '
                  'Ssl: {train_sllloss.val:.4f} ({train_sllloss.avg:.4f}) '
                  .format(
                  epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time,
                  train_sllloss=train_sllloss, train_sncaloss=train_sncaloss))
    return