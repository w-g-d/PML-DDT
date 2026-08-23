import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.amp import GradScaler, autocast
from torch.optim import lr_scheduler

from dataset.dataset import (
    COCO2014_handler_test,
    COCO2014_handler_train,
    NUS_WIDE_handler_test,
    NUS_WIDE_handler_train,
    VOC2007_handler_test,
    VOC2007_handler_train,
    generate_noisy_labels,
    get_COCO2014,
    get_NUS_WIDE,
    get_VOC2007,
)
from src.helper_functions.helper_functions import (
    CutoutPIL,
    ModelEma,
    add_weight_decay,
    mAP,
)
from src.loss_functions.losses import DynamicAsymmetricLoss, MeanConstraintLoss
from src.models import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_DEVICE_TYPE = DEVICE.type
AMP_ENABLED = AMP_DEVICE_TYPE == 'cuda'


def parse_args():
    parser = argparse.ArgumentParser(description='PML-DDT: Partial Multi-Label Learning Training')
    parser.add_argument('--data-path', metavar='DIR', default='./data',
                        help='root path of the datasets')
    parser.add_argument('--dataset-name', default='VOC', type=str,
                        choices=['VOC', 'COCO', 'NUS-WIDE'], help='dataset to train on')
    parser.add_argument('--lr', default=0.00007, type=float,
                        help='learning rate of the warm-up stage')
    parser.add_argument('--model-name', default='tresnet_l', type=str,
                        choices=['tresnet_m', 'tresnet_l', 'tresnet_xl'])
    parser.add_argument('--model-path', default=None, type=str,
                        help='path to ImageNet-pretrained TResNet weights (e.g. tresnet_l.pth)')
    parser.add_argument('--num-classes', default=20, type=int,
                        help='number of classes (VOC2007: 20, COCO2014: 80, NUS-WIDE: 81)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--image-size', default=224, type=int, metavar='N',
                        help='input image size (default: 224)')
    parser.add_argument('--thre', required=True, type=float, metavar='N', 
                        help='threshold for the binary precision/recall/F1 metrics')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                        help='mini-batch size (VOC2007: 64, COCO2014 and NUS-WIDE: 128)')
    parser.add_argument('--noise-rate', '--noise_rate', dest='noise_rate', type=float, default=0.2,
                        help='partial-label corruption rate, should be less than 1')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    # stage 1: warm-up
    parser.add_argument('--warmup-epochs', default=10, type=int,
                        help='number of warm-up epochs (stage 1)')
    parser.add_argument('--warmup-weight-decay', default=7e-7, type=float,
                        help='weight decay of the warm-up stage')
    parser.add_argument('--weight-strategy', default='linear', type=str,
                        choices=['linear', 'exp', 'log', 'pow_2', 'pow_0.5', 'cut', 'noweight'],
                        help='confidence weighting strategy used in the warm-up stage')

    # stage 2: self-evolution disambiguation
    parser.add_argument('--epochs', default=60, type=int,
                        help='maximum number of disambiguation epochs (stage 2)')
    parser.add_argument('--lr2', default=1e-4, type=float,
                        help='learning rate of the disambiguation stage')
    parser.add_argument('--weight-decay2', default=1e-4, type=float,
                        help='weight decay of the disambiguation stage')
    parser.add_argument('--loss', dest='loss_function', default='dynamic_asl_mc', type=str,
                        choices=['bce', 'dynamic_asl_mc'],
                        help='loss of stage 2: bce, asl, or asl + mean-constraint (asl_mc)')
    parser.add_argument('--init-upper', '--init_upper', dest='init_upper', default=0.9, type=float,
                        help='initial upper confidence bound for disambiguation')
    parser.add_argument('--init-lower', '--init_lower', dest='init_lower', default=0.0, type=float,
                        help='initial lower confidence bound for disambiguation')
    parser.add_argument('--final-upper', '--final_upper', dest='final_upper', type=float, default=0.5,
                        help='target/final upper threshold')
    parser.add_argument('--final-lower', '--final_lower', dest='final_lower', type=float, default=0.5,
                        help='target/final lower threshold')
    parser.add_argument('--threshold-update-epochs', type=int, default=60,
                        help='maximum epochs for threshold updating (VOC2007: 60, COCO2014 and NUS-WIDE: 30)')
    parser.add_argument('--gamma-neg-init', type=float, default=5.0,
                         help='Initial value of gamma negative')
    parser.add_argument('--gamma-neg-final', type=float, default=2.0,
                         help='Final value of gamma negative')
    parser.add_argument('--gamma-update-epochs', type=int, default=30,
                         help='Maximum epochs for gamma updating')
    parser.add_argument('--gamma-decay-power', type=float, default=1.0,
                         help='Decay power for gamma scheduling')
    args = parser.parse_args()
    args.do_bottleneck_head = False
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, save_path='best_model.pth'):
        """
        Args:
            patience (int): how many consecutive epochs without improvement to tolerate
            delta (float): minimum change to be considered an improvement
            save_path (str): path to save the best model
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, model, ema, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.save_checkpoint(model, ema)
        elif current_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model, ema)

    def save_checkpoint(self, model, ema):
        """Save the current best model and its EMA weights."""
        torch.save(model.state_dict(), self.save_path)
        dirname = os.path.dirname(self.save_path)
        ema_path = os.path.join(dirname, 'ema_' + os.path.basename(self.save_path))
        torch.save(ema.state_dict(), ema_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = os.path.join('checkpoints', args.dataset_name, str(args.noise_rate))
    os.makedirs(save_dir, exist_ok=True)

    # Setup model
    print('creating model...')
    model = create_model(args)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)

    if args.model_path:  # load ImageNet-pretrained weights (except the classification head)
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {'module.' + k: v for k, v in state['model'].items() if
                         ('module.' + k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)

    ema = ModelEma(model, 0.9997)  # 0.9997^641 = 0.82

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((args.image_size, args.image_size)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()])

    if args.dataset_name == 'VOC':
        # Expected directory structure:
        # ├── VOCtest_06-Nov-2007/
        # │   └── VOCdevkit/
        # │       └── VOC2007/
        # │           ├── Annotations/
        # │           ├── ImageSets/
        # │           ├── JPEGImages/
        # │           ├── SegmentationClass/
        # │           └── SegmentationObject/
        # └── VOCtrainval_06-Nov-2007/

        data_path_val = os.path.join(args.data_path, 'VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007')
        data_path_train = os.path.join(args.data_path, 'VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007')

        train_images, train_labels, test_images, test_labels = get_VOC2007(data_path_train, data_path_val)

        train_plabels = generate_noisy_labels(train_labels, noise_rate=args.noise_rate)

        train_dataset = VOC2007_handler_train(train_images, train_labels, train_plabels, data_path_train,
                                              transform_aug=train_transform)
        test_dataset = VOC2007_handler_test(test_images, test_labels, data_path_val, transform=test_transform)

    elif args.dataset_name == 'COCO':
        # Expected directory structure:
        # |-- train2014/
        # `-- val2014/
        #     `-- COCO_val2014_xxx.jpg
        data_path_val = os.path.join(args.data_path, 'COCO/val2014')
        data_path_train = os.path.join(args.data_path, 'COCO/train2014')

        train_images, train_labels, test_images, test_labels = get_COCO2014()
        train_plabels = generate_noisy_labels(train_labels, noise_rate=args.noise_rate)

        train_dataset = COCO2014_handler_train(train_images, train_labels, train_plabels, data_path_train,
                                               transform=train_transform)
        test_dataset = COCO2014_handler_test(test_images, test_labels, data_path_val, transform=test_transform)

    elif args.dataset_name == 'NUS-WIDE':
        # Expected directory structure:
        # ├── ImageList/
        # │   ├── TrainImagelist.txt
        # │   └── TestImagelist.txt
        # ├── Groundtruth/
        # │   ├── TrainTestLabels/
        # │   │   ├── Labels_airport_Test.txt
        # │   │   └── ...
        # │   ├── AllLabels/
        # ├── Flickr/
        # │   ├── actor
        #     │   ├── 0001_2124494179.jpg
        #         └── ...
        train_images, train_labels, test_images, test_labels = get_NUS_WIDE(args.data_path, cache=True)
        train_plabels = generate_noisy_labels(train_labels, noise_rate=args.noise_rate)

        train_dataset = NUS_WIDE_handler_train(train_images, train_labels, train_plabels, args.data_path,
                                               transform=train_transform)
        test_dataset = NUS_WIDE_handler_test(test_images, test_labels, args.data_path, transform=test_transform)

    print("len(train_dataset): ", len(train_dataset))
    print("len(val_dataset): ", len(test_dataset))

    # PyTorch data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    train(model, ema, train_loader, test_loader, args, train_plabels, save_dir)


def self_evolution_disambiguation(model, train_loader, partial_labels, lower, upper):
    """Refine the candidate (partial) labels with the current model predictions.

    Candidate labels whose confidence is below ``lower`` are discarded (set to 0),
    and labels with confidence inside the ambiguous interval [lower, upper] are
    marked as 0.5 (ignored by the loss weighting).
    """
    print('Starting disambiguation...')
    model.eval()
    sigmoid = torch.nn.Sigmoid()
    partial_labels = torch.from_numpy(partial_labels).to(DEVICE)
    confidences = torch.zeros_like(partial_labels)

    for images, _, _, indices in train_loader:
        # compute output
        with torch.no_grad(), autocast(AMP_DEVICE_TYPE, enabled=AMP_ENABLED):
            output_sigmoid = sigmoid(model(images.to(DEVICE)).float())
        confidences[indices] = output_sigmoid

    refined = partial_labels.clone()
    refined = torch.where((partial_labels == 1) & (confidences < lower),
                          torch.zeros_like(refined), refined)
    refined = torch.where((partial_labels == 1) & (confidences >= lower) & (confidences <= upper),
                          torch.full_like(refined, 0.5), refined)
    model.train()

    return refined


def train(model, ema, train_loader, val_loader, args, partial_labels, save_dir):
    sigmoid = torch.nn.Sigmoid()
    steps_per_epoch = len(train_loader)
    log_file = os.path.join(save_dir, 'validation_log.csv')

    # ------------------------- stage 1: warm-up -------------------------
    parameters = add_weight_decay(model, args.warmup_weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr,
                                 weight_decay=args.warmup_weight_decay)  # true wd, filter_bias_and_bn
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
                                        epochs=args.epochs, pct_start=0.2)
    scaler = GradScaler(AMP_DEVICE_TYPE, enabled=AMP_ENABLED)
    early_stopper_warmup = EarlyStopping(patience=1, delta=0,
                                         save_path=os.path.join(save_dir, 'best_warmup_model.pth'))

    for epoch in range(args.warmup_epochs):
        print(f'Warm-up epoch {epoch + 1}/{args.warmup_epochs}')

        for i, (images, _, partial_targets, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            partial_targets = partial_targets.to(DEVICE).float()

            with autocast(AMP_DEVICE_TYPE, enabled=AMP_ENABLED):  # mixed precision
                output = model(images).float()
                output_sigmoid = sigmoid(output)

            # confidence-weighted BCE
            weight = output_sigmoid.clone().detach()

            # You can apply this adjustment under heavy label noise (e.g., noise_rate >= 0.6)
            # weight[partial_targets == 0] = 1.0

            if args.weight_strategy == 'linear':
                pass
            elif args.weight_strategy == 'exp':
                weight = weight.exp() - 1
            elif args.weight_strategy == 'log':
                weight = (weight + 1).log()
            elif args.weight_strategy == 'pow_2':
                # conservative strategy: down-weight high-confidence predictions to
                # slow down the over-confident learning process
                weight = weight.pow(2)
            elif args.weight_strategy == 'pow_0.5':
                # aggressive strategy: up-weight high-confidence predictions to
                # speed up learning on these samples
                weight = weight.pow(0.5)
            elif args.weight_strategy == 'cut':
                # truncate weights above 0.8 so that over-confident predictions
                # do not dominate training
                weight = weight.clamp(max=0.8)
            elif args.weight_strategy == 'noweight':
                weight = torch.ones_like(weight)

            loss = F.binary_cross_entropy(output_sigmoid, partial_targets, weight)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            # store information
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.4f}'
                      .format(epoch, args.warmup_epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], loss.item()))

        model.eval()
        mAP_score_regular, mAP_score_ema = validate_multi(val_loader, model, ema, epoch,
                                                          args.thre, log_file)
        model.train()

        # check whether to stop the warm-up stage early
        early_stopper_warmup(mAP_score_regular, model, ema, epoch + 1)
        if early_stopper_warmup.early_stop:
            print(f"Early stopping at warm-up epoch {epoch + 1}. "
                  f"Best mAP: {early_stopper_warmup.best_score:.4f} "
                  f"at epoch {early_stopper_warmup.best_epoch}")
            break


    # -------------------- stage 2: self-evolution disambiguation --------------------
    if args.dataset_name != 'VOC':
        parameters = add_weight_decay(model, args.weight_decay2)
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr2,
                                    weight_decay=0)  # true wd, filter_bias_and_bn
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr2, steps_per_epoch=steps_per_epoch,
                                            epochs=args.epochs, pct_start=0.2)
        scaler = GradScaler(AMP_DEVICE_TYPE, enabled=AMP_ENABLED)

    early_stopper = EarlyStopping(patience=5, delta=0.0,
                                    save_path=os.path.join(save_dir, 'best_model.pth'))

    criterion_dynamic_asl = DynamicAsymmetricLoss(gamma_neg_init=args.gamma_neg_init,gamma_neg_final=args.gamma_neg_final,max_epoch=args.gamma_update_epochs,decay_power=args.gamma_decay_power)
    criterion_mc = MeanConstraintLoss(margin=0.5)

    # the ambiguous interval [lower, upper] is annealed towards [final_lower, final_upper]
    upper, lower = args.init_upper, args.init_lower
    final_upper, final_lower = args.final_upper, args.final_lower
    update_epochs = args.threshold_update_epochs

    eta1 = (upper - final_upper) / update_epochs
    eta2 = (final_lower - lower) / update_epochs

    highest_mAP_regular = 0
    highest_mAP_ema = 0
    highest_mAP = 0

    for epoch in range(args.epochs):
        refined_labels = self_evolution_disambiguation(model, train_loader, partial_labels,
                                                       lower, upper)

        # entries marked as 0.5 (ambiguous) are excluded from the loss
        weights = torch.ones_like(refined_labels)
        weights[refined_labels == 0.5] = 0

        for i, (images, _, partial_targets, indices) in enumerate(train_loader):
            images = images.to(DEVICE)
            partial_targets = partial_targets.to(DEVICE).float()

            weight = weights[indices]
            r = refined_labels[indices]

            with autocast(AMP_DEVICE_TYPE, enabled=AMP_ENABLED):  # mixed precision
                output = model(images).float()
                output_sigmoid = sigmoid(output)

            if args.loss_function == 'bce':
                loss = F.binary_cross_entropy(output_sigmoid, partial_targets)
            elif args.loss_function == 'dynamic_asl_mc':
                loss_dis = criterion_dynamic_asl(output_sigmoid, r, weight, epoch)
                loss_mc = criterion_mc(output_sigmoid, partial_targets)
                loss = loss_dis + loss_mc

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            # store information
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.4f}'
                      .format(epoch, args.epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], loss.item()))

        if epoch < update_epochs:
            upper -= eta1
            lower += eta2

        model.eval()
        mAP_score_regular, mAP_score_ema = validate_multi(val_loader, model, ema, epoch,
                                                          args.thre, log_file)
        model.train()

        # check whether to stop early
        early_stopper(mAP_score_ema, model, ema, epoch + 1)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}. "
                  f"Best mAP: {early_stopper.best_score:.4f} "
                  f"at epoch {early_stopper.best_epoch}")
            break

        if mAP_score_regular > highest_mAP_regular:
            highest_mAP_regular = mAP_score_regular
        if mAP_score_ema > highest_mAP_ema:
            highest_mAP_ema = mAP_score_ema
        if max(mAP_score_ema,mAP_score_regular) > highest_mAP:
            highest_mAP = max(mAP_score_ema,mAP_score_regular)

        print('current_mAP_ema = {:.2f}, highest_mAP_ema = {:.2f}\n'.format(mAP_score_ema, highest_mAP_ema))
        print('current_mAP_regular = {:.2f}, highest_mAP_regular = {:.2f}\n'.format(mAP_score_regular, highest_mAP_regular))
        print('highest_mAP = {:.2f}\n'.format(highest_mAP))

def validate_multi(val_loader, model, ema_model, epoch, threshold, log_file):
    print("starting validation")
    model.eval()
    ema_model.eval()
    sigmoid = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    tp, fp, fn = 0, 0, 0

    for images, target, _ in val_loader:
        # compute output
        with torch.no_grad(), autocast(AMP_DEVICE_TYPE, enabled=AMP_ENABLED):
            output_regular = sigmoid(model(images.to(DEVICE))).cpu()
            output_ema = sigmoid(ema_model.module(images.to(DEVICE))).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.detach())
        preds_ema.append(output_ema.detach())
        targets.append(target.detach())

        # binarize the EMA predictions with the given threshold
        pred = output_ema.data.gt(threshold).long()
        # accumulate per-class true/false positives and false negatives
        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)

    # per-class precision / recall / F1 (computed once over the whole validation set)
    p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0
           for i in range(len(tp))]
    r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0
           for i in range(len(tp))]
    f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0
           for i in range(len(tp))]
    mean_p_c = sum(p_c) / len(p_c)
    mean_r_c = sum(r_c) / len(r_c)
    mean_f_c = sum(f_c) / len(f_c)

    # overall precision / recall / F1
    p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
    r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
    f_o = 2 * p_o * r_o / (p_o + r_o)

    print(f'--------------------------------threshold: {threshold}------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))


    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))

    # save the log to a CSV file
    save_header = not os.path.exists(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if save_header:
            writer.writerow(['Epoch', 'mAP score regular', 'mAP score EMA',
                             'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1'])
        if epoch == 0:
            # write the timestamp of the current run
            writer.writerow([f'Run started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([epoch + 1, mAP_score_regular, mAP_score_ema,
                         mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o])

    return mAP_score_regular, mAP_score_ema


if __name__ == '__main__':
    main()
