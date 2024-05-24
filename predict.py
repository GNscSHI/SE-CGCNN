import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import CIFData
from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--depth', type=float, default=2, 
                    help='threshold depth for surface atoms')
parser.add_argument('--consolidate', action='store_true')

args = parser.parse_args(sys.argv[1:])
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if model_args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, model_args, best_mae_error

    # load data
    dataset = CIFData(args.cifpath, depth=args.depth)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)
    
    sample_data_list = [dataset[i] for i in range(len(dataset))]
    _, targets_dict, _ = collate_pool(sample_data_list)
    
    normalizers = {}
    for i in range(len(targets_dict)):
        exec(f'normalizers["normalizer{i+1}"] = Normalizer(targets_dict[i])')
    
    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task == 'classification' else False,
                                i_tasks=targets_dict.keys())
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    
    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        if args.consolidate:
            # ini model param
            for n,p in model.named_parameters():
                n = n.replace('.', '__')
                model.register_buffer('{}_mean'.format(n), 0*p.data.clone())
                model.register_buffer('{}_fisher'
                                     .format(n), 0*p.data.clone())
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])
        for i in range(len(targets_dict)):
            exec(f'normalizers["normalizer{i+1}"].load_state_dict(checkpoint["normalizer{i+1}"])')
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_loss']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    validate(test_loader, model, criterion, test=True, **normalizers)


def validate(val_loader, model, criterion, test=False, **normalizers):
    n_tasks = len(model.i_tasks)
    batch_time = AverageMeter()
    losses = []
    for i in range(n_tasks):
        exec(f'losses.append(AverageMeter())')
    losses_total = AverageMeter()
    if model_args.task == 'regression':
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
    for batch, (input, targets, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
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
        if model_args.task == 'regression':
            for i, count in enumerate(model.i_tasks):
                exec(f'target{i+1}_normed = list(normalizers.values())[i].norm(targets[count])')
        else:
            target_normed = target.view(-1).long()
        with torch.no_grad():
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
        total_loss = sum(loss) / n_tasks
        
        if args.consolidate:
            # ewc loss is 0 if there's no consolidated parameters.
            ewc_loss = model.ewc_loss(cuda=args.cuda)
            loss = loss + ewc_loss.item()
        
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
        return losses_total.avg  
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


if __name__ == '__main__':
    main()
