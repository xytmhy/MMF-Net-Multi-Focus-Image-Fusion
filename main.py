import argparse
import os
import shutil
import time

import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import fusion_transforms
import models

import datasets
from multiscaleloss import *
import datetime
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='PyTorch Matte Model Based Cascade Boundary Aware Image Fusion Net Training on Simulated Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', default='/media/data2/mhy/data/FusionTMM/Fusion0816_Tri',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='fusion_data', choices=dataset_names,
                    help='dataset type : ' + ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.9, type=float,
                   help='test-val split proportion (between 0 (only test) and 1 (only train))')
parser.add_argument('--arch', '-a', metavar='ARCH', default='CBAfusionnet', choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder')
parser.add_argument('--milestones', default=[40, 60, 80], metavar='N', nargs='*', help='epochs at which learning rate is divided by 5')

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_EPE, save_path
    args = parser.parse_args()
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataset, save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(8):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    # Data loading code
    input_transform = transforms.Compose([
        fusion_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])
    target_transform = transforms.Compose([
        fusion_transforms.TargetToTensor(),
        transforms.Normalize(mean=[0], std=[255]),
    ])
    co_transform = fusion_transforms.Compose([
        # fusion_transforms.RandomTranslate(100),
        # deblur_transforms.RandomRotate(45),
        fusion_transforms.RandomCrop((256, 256)),
        fusion_transforms.RandomVerticalFlip(),
        fusion_transforms.RandomHorizontalFlip()
    ])

    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform,
        split=args.split_file if args.split_file else args.split_value
    )

    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set), len(train_set), len(test_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=2,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0, output_writers)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

    print('Total paprams: %.2fMB'%(sum(p.numel() for p in model.parameters())/(1024*1024)*4))

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        train_loss, train_EPE = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean EPE', train_EPE, epoch)

        # evaluate on validation set

        with torch.no_grad():
            EPE = validate(val_loader, model, epoch, output_writers)
        test_writer.add_scalar('mean EPE', EPE, epoch)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE
        }, is_best)


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = torch.cat(images, 1).to(device)
        target = target.to(device)

        input_image = images[:, :6]
        target_fusion = images[:, 6:9]
        target_Gmap = target

        # compute output
        # with torch.no_grad():
        output = model(input_image, target_fusion, target_Gmap)

        result_Gmap = output[:, :1]
        result_fusion = output[:, 1:4]

        loss = mapEPE(target_fusion, target_Gmap, result_Gmap, result_fusion, mean=True)
        fusion_EPE = realEPE(result_fusion, target_fusion)

        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        EPEs.update(fusion_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
                  .format(epoch, i, epoch_size, batch_time, data_time, losses, EPEs))
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, EPEs.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = torch.cat(images, 1).to(device)
        target = target.to(device)

        input_image = images[:, :6]
        target_fusion = images[:, 6:9]
        target_Gmap = target

        # compute output
        output = model(input_image, target_fusion, target_Gmap)

        result_Gmap = output[:, :1]
        result_fusion = output[:, 1:4]

        # record EPE
        fusion_EPE = realEPE(result_fusion, target_fusion)
        EPEs.update(fusion_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(output_writers):  # log first output of first batches
            if epoch == 0:
                # mean_values = torch.tensor([0.411,0.432,0.45], dtype=input.dtype).view(3,1,1)
                output_writers[i].add_image('GroundTruth', (target_fusion[0, :].cpu()).clamp(0, 1), 0)
                output_writers[i].add_image('Guidance GT', (target_Gmap[0, :].cpu()).clamp(0, 1), 0)
                output_writers[i].add_image('Inputs', (input_image[0, :3].cpu()).clamp(0, 1), 0)
                output_writers[i].add_image('Inputs', (input_image[0, 3:].cpu()).clamp(0, 1), 1)
            output_writers[i].add_image('Fusion Results', (result_fusion[0, :].cpu()).clamp(0, 1), epoch)
            output_writers[i].add_image('Guidance Maps', (result_Gmap[0, :].cpu()).clamp(0, 1), epoch)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, EPEs))

    print(' * EPE {:.6f}'.format(EPEs.avg))

    return EPEs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

if __name__ == '__main__':
    main()
