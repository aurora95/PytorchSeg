from __future__ import absolute_import

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from datasets import *
from data_utils import GeneratorEnqueuer, DataTransformer
from models import *
from utils import *
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('model_name',
                        help='the name of model definition function',
                        default=None, type=str)
    parser.add_argument('dataset', default=None, type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
    parser.add_argument('--epochs', dest='epochs', default=30, type=int)

    parser.add_argument('--base_lr', dest='base_lr', default=0.01, type=float)
    parser.add_argument('--lr_power', dest='lr_power', default=0.9, type=float)
    parser.add_argument('--resize_size', dest='resize_size', nargs=2, default=None, type=int)
    parser.add_argument('--pad_size', dest='pad_size', nargs=2, default=None, type=int)
    parser.add_argument('--crop_size', dest='crop_size', nargs=2, default=[480, 480], type=int)
    parser.add_argument('--crop_mode', dest='crop_mode', default='random', type=str)
    parser.add_argument('--ch_mean', dest='ch_mean', nargs=3, default=[0.485*255, 0.456*255, 0.406*255], type=float)
    parser.add_argument('--ch_std', dest='ch_std', nargs=3, default=[0.229*255, 0.224*255, 0.225*255], type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0001, type=float)

    parser.add_argument('--workers', dest='workers', default=4, type=int)
    parser.add_argument('--max_queue_size', dest='max_queue_size', default=16, type=int)
    parser.add_argument('--gpu', dest='gpus',
                        nargs='*',
                        help='GPU device id to use',
                        default=[0], type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train_batch(batch_x, batch_y, net, optimizer, loss_functions, metric_functions):
    optimizer.zero_grad()
    inputs, targets = Variable(batch_x), Variable(batch_y)
    outputs = net(inputs)
    losses =[ f(outputs, targets) for f in loss_functions]
    for loss in losses:
        loss.backward()
    optimizer.step()
    for m in metric_functions:
        m.update(preds=[outputs], labels=[targets])
    return losses

def train(net, args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(os.path.join(current_dir, 'results/')) == False:
        os.mkdir(os.path.join(current_dir, 'results/'))
    save_path = 'results/%s/'%args.dataset
    if os.path.exists(os.path.join(current_dir, save_path)) == False:
        os.mkdir(os.path.join(current_dir, save_path))
    save_path += '%s/'%args.model_name
    if os.path.exists(os.path.join(current_dir, save_path)) == False:
        os.mkdir(os.path.join(current_dir, save_path))
    logger = Logger(save_path + 'logs/')

    train_file_path, val_file_path, data_dir, label_dir = get_dataset_path(args.dataset)
    classes = get_dataset_classes(args.dataset)
    transformer = DataTransformer(ch_mean=args.ch_mean, ch_std=args.ch_std, resize_size=args.resize_size,
                 pad_size=args.pad_size, crop_mode=args.crop_mode, crop_size=args.crop_size,
                 zoom_range=[0.5, 2.0], horizontal_flip=True, color_jittering_range=20.,
                 fill_mode='constant', cval=0., label_cval=255, data_format='channels_first',
                 color_format='RGB', x_dtype=np.float32)
    dataloader = VOC12(data_list_file=train_file_path, data_source_dir=data_dir,
                       label_source_dir=label_dir, data_transformer=transformer,
                       batch_size=args.batch_size, shuffle=True)

    num_sample = dataloader.get_num_sample()
    num_steps = num_sample//args.batch_size
    if num_sample % args.batch_size > 0:
        num_steps += 1

    enqueuer = GeneratorEnqueuer(generator=dataloader)
    enqueuer.start(workers=args.workers, max_queue_size=args.max_queue_size)
    output_generator = enqueuer.get()

    net = net(classes)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=args.gpus)
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(net.parameters(), lr=args.base_lr,
                                momentum=0.9, weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = get_polyscheduler(optimizer, args.lr_power, args.epochs)
    loss_functions = [nn.CrossEntropyLoss(ignore_index=255)]
    metric_functions = [SparseAccuracy(ignore_label=255, name='Acc')]

    for epoch in range(args.epochs):
        scheduler.step()
        print('training epoch %d/%d, lr=%.4f:'%(epoch+1, args.epochs, optimizer.state_dict()['param_groups'][0]['lr']))
        for m in metric_functions:
            m.reset()
        train_loss = 0.
        for i in range(num_steps):
            batch_x, batch_y = next(output_generator)
            batch_x, batch_y = torch.Tensor(batch_x).cuda(), torch.LongTensor(np.squeeze(batch_y).astype(int)).cuda()
            losses = train_batch(batch_x, batch_y, net, optimizer, loss_functions, metric_functions)
            info = ''
            train_loss += sum([loss.cpu().data.numpy()[0] for loss in losses])
            info += '| loss: %.3f'%(train_loss/(i+1))
            for m in metric_functions:
                name, value = m.get()
                info += ' | %s: %.3f'%(name, value)
            progress_bar(i, num_steps, info)
        # write logs for this epoch
        logger.scalar_summary('loss', train_loss/num_steps, epoch)
        for m in metric_functions:
            name, value = m.get()
            logger.scalar_summary(name, value, epoch)
        torch.save(net.state_dict(), save_path+'checkpoint.params')
    enqueuer.stop()

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    net = globals()[model_name]
    train(net, args)
