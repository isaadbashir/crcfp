import time
from itertools import cycle
from math import ceil

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from base import BaseTrainer
from utils.helpers import colorize_mask, DeNormalize
from utils.metrics import eval_metrics, AverageMeter


class Trainer(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None, gpu=None, gt_loader=None, test=False):
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.iter_per_epoch = iter_per_epoch

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

    def _train_epoch(self, epoch):
        if self.gpu == 0:
            self.logger.info('\n')

        self.model.train()

        self.supervised_loader.train_sampler.set_epoch(epoch)

        if self.mode == 'semi':
            self.unsupervised_loader.train_sampler.set_epoch(epoch)

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=135)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader)))
            tbar = tqdm(range(self.iter_per_epoch), ncols=160)

        self._reset_metrics()

        for batch_idx in tbar:

            if self.mode == 'supervised':
                (input_l, target_l), (input_ul, target_ul) = next(dataloader), (None, None)
            else:
                # (input_l, target_l), (image_ul, label_ul, input_ul, target_ul, ul1, br1, ul2, br2, flip) = next(dataloader)
                (input_l, target_l), (image_ul, label_ul, input_ul, target_ul, ul1, br1, ul2, br2, flip) = next(dataloader)

            if self.mode == 'supervised':
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                self.optimizer.zero_grad()

                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1)
            else:

                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
                image_ul, label_ul = image_ul.cuda(non_blocking=True), label_ul.cuda(non_blocking=True)
                self.optimizer.zero_grad()

                kargs = {'gpu': self.gpu, 'ul1': ul1, 'br1': br1, 'ul2': ul2, 'br2': br2, 'flip': flip}
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, image_unlabelled = image_ul, x_ul=input_ul,
                                                             curr_iter=batch_idx, epoch=epoch - 1, **kargs)
                target_ul = label_ul

            total_loss.backward()
            self.optimizer.step()

            if self.gpu == 0:
                if batch_idx % 100 == 0:
                    self.logger.info("epoch: {} train_loss: {}".format(epoch, total_loss))

            if batch_idx == 0:
                for key in cur_losses:
                    if not hasattr(self, key):
                        setattr(self, key, AverageMeter())

            # self._update_losses has already implemented synchronized DDP
            self._update_losses(cur_losses)

            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            if self.gpu == 0:
                logs = self._log_values(cur_losses)

                if batch_idx % self.log_step == 0:
                    self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                    self._write_scalars_tb(logs)

                # if batch_idx % int(len(self.unsupervised_loader)*0.9) == 0:
                #     self._write_img_tb(input_l, target_l, input_ul, target_ul, outputs, epoch)

                descrip = 'T ({}) | '.format(epoch)
                for key in cur_losses:
                    descrip += key + ' {:.2f} '.format(getattr(self, key).average)
                descrip += 'm1 {:.2f} m2 {:.4f} dice_l {:.4f} dice_ul {:.4f}|'.format(self.mIoU_l, self.mIoU_ul, self.mDice_l, self.mDice_ul)
                tbar.set_description(descrip)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            self.lr_scheduler.step(epoch=epoch - 1)

        return logs if self.gpu == 0 else None

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            if self.gpu == 0:
                val_visual = []

            for batch_idx, (data, target) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                output = self.model(data)

                output = output[:, :, :H, :W]
                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)

                total_loss_val.update(loss.item())

                # eval_metrics has already implemented DDP synchronized
                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                if self.gpu == 0:
                    # LIST OF IMAGE TO VIZ (15 images)
                    if len(val_visual) < 15:
                        if isinstance(data, list):
                            data = data[0]
                        target_np = target.data.cpu().numpy()
                        output_np = output.data.max(1)[1].cpu().numpy()
                        val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                dice = (2 * IoU) / (IoU + 1)

                mIoU = IoU.mean()
                mDice = dice.mean()

                seg_metrics = {
                    "Pixel_Accuracy": np.round(pixAcc, 4),
                    "Mean_IoU": np.round(mIoU, 4),
                    "Mean_Dice": np.round(mDice, 4),
                    "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
                    "Class_Dice": dict(zip(range(self.num_classes), np.round(dice, 3)))

                }

                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.4f}, Mean IoU: {:.4f}, Mean Dice: {:.4f} |'.format(epoch,
                                                                                                                                    total_loss_val.average, pixAcc, mIoU, mDice))

            if self.gpu == 0:
                self._add_img_tb(val_visual, 'val')

                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-2]:
                    print(k, v)
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.mDice_l, self.mDice_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.mDice_l, self.class_iou_l, self.class_dice_l = seg_metrics_l.values()

        if 'unsup_pred' in outputs:
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)

            if self.gpu == 0:
                self._update_seg_metrics(*seg_metrics_ul, False)
                seg_metrics_ul = self._get_seg_metrics(False)
                self.pixel_acc_ul, self.mIoU_ul, self.mDice_ul, self.class_iou_ul, self.class_dice_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
            dice = (2 * IoU) / (IoU + 1)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
            dice = (2 * IoU) / (IoU + 1)

        mIoU = IoU.mean()
        mDice = dice.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Mean_Dice": np.round(mDice, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
            "Class_Dice": dict(zip(range(self.num_classes), np.round(dice, 3)))

        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        logs['dice_labeled'] = self.mDice_l

        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
            logs['dice_unlabeled'] = self.mDice_ul

        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k:
                self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        # current_rampup = self.model.module.unsup_loss_w.current_rampup
        # self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

        if self.mode == 'semi':
            outputs_ul_np = outputs['unsup_pred'].data.max(1)[1].cpu().numpy()
            targets_ul_np = target_ul.data.cpu().numpy()
            imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_ul, outputs_ul_np, targets_ul_np)]
            self._add_img_tb(imgs, 'unsupervised')
