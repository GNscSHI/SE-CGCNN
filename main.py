import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import pandas as pd
import numpy as np
import torch, gc
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.init as init

from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.6, type=float, metavar='N',
                    help='number of training data to be loaded (default 0.6)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.2')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.2')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='Adam', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

parser.add_argument('--mode', '-m', choices=('grad_norm', 'equal_weight'), default='grad_norm', help='weight determining strategy for mutiple tasks (default: grad_norm)')
parser.add_argument('--alpha', '-a', type=float, default=1.5, 
                    help='hyper-parameter of gradnorm, constraint on balancing the training speed')

parser.add_argument('--depth', type=float, default=0, 
                    help='threshold depth for surface atoms (default: 0)')

parser.add_argument('--fix-conv-param', action='store_true',
                    help='fix parameters in convolution layers for transfer learning')
parser.add_argument('--fine-tuning', action='store_true',
                    help='reinitialize parameters with pretrained model')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_loss = 1e10
else:
    best_mae_error = 0.
    
epoch_info = {'epoch':[], 'train_batch_num':[], 'train_loss':[],'valid_batch_num':[], 'valid_loss':[],
              'test_batch_num':[]}

def main():
    global args, best_loss, epoch_info

    # load data
    dataset = CIFData(*args.data_options, depth=args.depth)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)
    
    normalizers = {}
    
    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, targets_dict, _ = collate_pool(sample_data_list)

        for i in targets_dict.keys():
            exec(f'normalizers["normalizer{i+1}"] = Normalizer(targets_dict[i])')
            
    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=True if args.task == 'classification' else False,
                                i_tasks=targets_dict.keys())
    if args.cuda:
        model.cuda()
    print(f"cuda:{args.cuda}")
    
    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            for name, param in model.state_dict().items():
                if name in checkpoint['state_dict'] and name != 'weights':
                    param.copy_(checkpoint['state_dict'][name])
                else:
                    print(f"{name} isn't loaded from checkpoint")

            if not (args.fine_tuning or args.fix_conv_param):
                args.start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                optimizer.load_state_dict(checkpoint['optimizer'])
                for i in targets_dict.keys():
                    exec(f'normalizers["normalizer{i+1}"].load_state_dict(checkpoint["normalizer{i+1}"])')
                    
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        args.start_epoch = 0
        
        if args.fix_conv_param:
            for name, parameter in model.named_parameters():
                if ('embedding' in name) or ('convs' in name):
                    parameter.requires_grad = False
                else:
                    # Reinitialize
                    if len(parameter.shape) > 1:
                        init.xavier_uniform_(parameter)
                    else:
                        init.constant_(parameter, 0.0)

            args.start_epoch = 0
            print('conv_layer params are fixed')
    
    
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.9)
    
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, **normalizers)

        # evaluate on validation set
        total_loss = validate(val_loader, model, criterion, **normalizers)

        if total_loss != total_loss:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = total_loss < best_loss
            best_loss = min(total_loss, best_loss)
        else:
            is_best = total_loss > best_loss
            best_loss = max(total_loss, best_loss)

        save_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'args': vars(args)
        }
        for key, value in normalizers.items():
            save_dict[key] = value.state_dict()

        save_checkpoint(save_dict, is_best)
        
        # release memory
        gc.collect()
        torch.cuda.empty_cache()

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    test_loss = validate(test_loader, model, criterion, test=True, **normalizers)
    
    # save epoch_info
    pd.DataFrame(pd.DataFrame.from_dict(epoch_info, orient='index').values.T, columns=list(epoch_info.keys())).to_csv('epoch_info.csv')
    
    
def train(train_loader, model, criterion, optimizer, epoch, **normalizers):
    n_tasks = len(model.i_tasks)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = []
    for i in range(n_tasks):
        exec(f'losses.append(AverageMeter())')
    losses_total = AverageMeter()
    if args.task == 'regression':
        mae_errors = []
        for i in range(n_tasks):
            exec(f'mae_errors.append(AverageMeter())')

    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for batch, (input, targets, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = []
            for i, count in enumerate(model.i_tasks):
                exec(f'target_normed.append(list(normalizers.values())[i].norm(targets[count]))')
        else:
            target_normed = target.view(-1).long()
            
        target_var = []
        if args.cuda:
            for i in range(n_tasks):
                exec(f'target_var.append(Variable(target_normed[i].cuda(non_blocking=True)))')
        else:
            for i in range(n_tasks):
                exec(f'target_var.append(Variable(target_normed[i]))')
        
        # weight determining strategies
        if args.mode == 'equal_weight':
            # compute output
            outputs = model(*input_var)
            loss = []
            for i in range(n_tasks):
                exec(f'loss.append(criterion(outputs[i], target_var[i]))')
            total_loss = sum(loss)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        if args.mode == 'grad_norm':
            outputs = model(*input_var)
            loss = []
            for i in range(n_tasks):
                exec(f'loss.append(criterion(outputs[i], target_var[i]))')
            task_loss = torch.stack(loss)
            weighted_task_loss = torch.mul(model.weights, task_loss)
            total_loss = torch.sum(weighted_task_loss) # get the total loss
            
            if batch == 0:
                # set L(0)
                if torch.cuda.is_available():
                    initial_task_loss = task_loss.data.cpu()
                else:
                    initial_task_loss = task_loss.data
                initial_task_loss = initial_task_loss.numpy()
            
            optimizer.zero_grad()
            # do the backward pass to compute the gradients for the whole set of weights
            total_loss.backward(retain_graph=True)
            
            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            model.weights.grad.data = model.weights.grad.data * 0.0
            
            # get layer of shared weights
            W = model.get_last_shared_layer()

            # get the gradient norms for each of the tasks
            norms = []
            for t in range(len(task_loss)):
                # get the gradient of this task loss with respect to the shared parameters
                gygw = torch.autograd.grad(task_loss[t], W.parameters(), retain_graph=True)
                # compute the norm
                norms.append(torch.norm(torch.mul(model.weights[t], gygw[0])))
            norms = torch.stack(norms)


            # compute the inverse training rate r_i(t) 
            # \curl{L}_i 
            if torch.cuda.is_available():
                loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
            else:
                loss_ratio = task_loss.data.numpy() / initial_task_loss
            # r_i(t)
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)

            # compute the mean norm \tilde{G}_w(t) 
            if args.cuda:
                mean_norm = np.mean(norms.data.cpu().numpy())
            else:
                mean_norm = np.mean(norms.data.numpy())
                
            # compute the GradNorm loss 
            # this term has to remain constant
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
            if args.cuda:
                constant_term = constant_term.cuda()

            # this is the GradNorm loss itself
            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

            # compute the gradient for the weights
            model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]
  
            optimizer.step()
        
        
        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = []
            for i, count in enumerate(model.i_tasks):
                exec(f'mae_error.append(mae(list(normalizers.values())[i].denorm(outputs[i].data.cpu()), targets[count]))')
                exec(f'losses[i].update(loss[i].data.cpu(), targets[count].size(0))')
                exec(f'mae_errors[i].update(mae_error[i], targets[count].size(0))')

            losses_total.update(total_loss.data.cpu(), targets[0].size(0))
                
                
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch % args.print_freq == 0:
            string = [0]
            if args.task == 'regression':
                string[0] = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f}\t'.format(
                    epoch, batch, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses_total)
                )

                for i in range(n_tasks):
                    exec(f'string[0] += "Loss{i+1} {losses[i].val:.4f}\t"')
                    exec(f'string[0] += "MAE{i+1} {mae_errors[i].val:.4f}\t"')
                print(string[0])
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )
        
    if args.mode == 'grad_norm':
        # print data
        weight_list = model.weights.data.tolist()
        formatted_weights = [f"{w:.4f}" for w in weight_list]
        print(f"weights of {n_tasks} tasks: {formatted_weights}")
        
        # renormalize
        normalize_coeff = len(task_loss) / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

    # update epoch_info
    epoch_info['epoch'].append(epoch)
    epoch_info['train_batch_num'].append(len(train_loader))
    epoch_info['train_loss'].append(losses_total.avg)
    for i in range(n_tasks):
        try:
            exec(f'epoch_info["train_loss{i+1}"]') 
        except:
            exec(f'epoch_info["train_loss{i+1}"] = []')
            exec(f'epoch_info["train_mae{i+1}"] = []')
        exec(f'epoch_info["train_loss{i+1}"].append(losses[i].avg)')
        exec(f'epoch_info["train_mae{i+1}"].append(mae_errors[i].avg)')

            
def validate(val_loader, model, criterion, test=False, **normalizers):
    n_tasks = len(model.i_tasks)
    batch_time = AverageMeter()
    losses = []
    for i in range(n_tasks):
        exec(f'losses.append(AverageMeter())')
    losses_total = AverageMeter()
    if args.task == 'regression':
        mae_errors = []
        for i in range(n_tasks):
            exec(f'mae_errors.append(AverageMeter())')
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        for i in range(n_tasks):
            exec(f'test_targets.append([])')
            exec(f'test_preds.append([])')
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad(): ### save memory
        for batch, (input, targets, batch_cif_ids) in enumerate(val_loader):

            if args.cuda:
                with torch.no_grad():
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                with torch.no_grad():
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
            if args.task == 'regression':
                for i, count in enumerate(model.i_tasks):
                    exec(f'target{i+1}_normed = list(normalizers.values())[i].norm(targets[count])')
            else:
                target_normed = target.view(-1).long()
            if args.cuda:
                for i in range(n_tasks):
                    exec(f'target{i+1}_var = Variable(target{i+1}_normed.cuda(non_blocking=True))')
            else:
                for i in range(n_tasks):
                    exec(f'target{i+1}_var = Variable(target{i+1}_normed)')

            # compute output
            outputs = model(*input_var)
            loss = []
            for i in range(n_tasks):
                exec(f'loss.append(criterion(outputs[i], target{i+1}_var))')
            total_loss = sum(loss)
            
            # measure accuracy and record loss
            if args.task == 'regression':
                mae_error = []
                for i, count in enumerate(model.i_tasks):
                    exec(f'mae_error.append(mae(list(normalizers.values())[i].denorm(outputs[i].data.cpu()), targets[count]))')
                    exec(f'losses[i].update(loss[i].data.cpu(), targets[count].size(0))')
                    exec(f'mae_errors[i].update(mae_error[i], targets[count].size(0))')
                
                losses_total.update(total_loss.data.cpu(), targets[0].size(0))

                if test:
                    test_pred = []
                    for i, count in enumerate(model.i_tasks):
                        exec(f'test_pred.append(list(normalizers.values())[i].denorm(outputs[i].data.cpu()))')
                        exec(f'test_preds[i] += test_pred[i].view(-1).tolist()')
                        exec(f'test_targets[i] += targets[count].view(-1).tolist()')
                    
                    test_cif_ids += batch_cif_ids

            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))
                if test:
                    test_pred = torch.exp(output.data.cpu())
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch % args.print_freq == 0:
                string = [0]
                if args.task == 'regression':
                    string[0] = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f}\t'.format(
                        batch, len(val_loader), batch_time=batch_time, 
                        loss=losses_total))
                    
                    for i in range(n_tasks):
                        exec(f'string[0] += "Loss{i+1} {losses[i].val:.4f}\t"')
                        exec(f'string[0] += "MAE{i+1} {mae_errors[i].val:.4f}\t"')
                    print(string[0])
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        accu=accuracies, prec=precisions, recall=recalls,
                        f1=fscores, auc=auc_scores))
                
    # update epoch_info
    if not test:
        epoch_info['valid_batch_num'].append(len(val_loader))
        for i in range(n_tasks):
            try:
                exec(f'epoch_info["valid_loss{i+1}"]') 
            except:
                exec(f'epoch_info["valid_loss{i+1}"] = []')
                exec(f'epoch_info["valid_mae{i+1}"] = []')
            exec(f'epoch_info["valid_loss{i+1}"].append(losses[i].avg)')
            exec(f'epoch_info["valid_mae{i+1}"].append(mae_errors[i].avg)')
        epoch_info['valid_loss'].append(losses_total.avg)
        
    else:
        epoch_info['test_batch_num'].append(len(val_loader))
        for i in range(n_tasks):
            try:
                exec(f'epoch_info["test_loss{i+1}"]') 
            except:
                exec(f'epoch_info["test_loss{i+1}"] = []')
                exec(f'epoch_info["test_mae{i+1}"] = []')
            exec(f'epoch_info["test_loss{i+1}"].append(losses[i].avg)')
            exec(f'epoch_info["test_mae{i+1}"].append(mae_errors[i].avg)')
        epoch_info['test_loss'] = [losses_total.avg]
            
    if test:
        star_label = '**'
        import csv
        
        test_targets_preds = []
        for i in range(n_tasks):
            exec(f'test_targets_preds.append(test_targets[i])')
            exec(f'test_targets_preds.append(test_preds[i])')
            
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for values in zip(test_cif_ids, *test_targets_preds):
                writer.writerow((values))
    else:
        star_label = '*'
    if args.task == 'regression':
        string = [0] 
        string[0] = ' {star} '.format(star=star_label)
        mae_errors_here = [0]
        for i in range(n_tasks):
            exec(f'mae_errors_here[0] = mae_errors[i]')
            exec(f'string[0] += "MAE{i+1} {mae_errors_here[0].avg:.4f}\t "')
        print(string[0])
        return losses_total.avg  # A single indicator is used to determine the optimal model
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
