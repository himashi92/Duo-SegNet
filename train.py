import argparse
import os
from datetime import datetime
from distutils.dir_util import copy_tree

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from data import image_loader
from loss import loss_sup, loss_adversarial_1, loss_adversarial_2, make_Dis_label, gt_label, loss_diff
from metrics import dice_coef
from model.UNet import Unet
from model.model_discriminator import PixelDiscriminator
from utils import get_logger, create_dir

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-2, help='learning rate')
parser.add_argument('--lr_dis', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--dataset', type=str, default='nuclei', help='dataset name')
parser.add_argument('--split', type=float, default=0.8, help='training data ratio')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--decay', default=3e-5, type=float)
parser.add_argument('--ratio', type=float, default=0.05, help='labeled data ratio')

opt = parser.parse_args()
CE = torch.nn.BCELoss()


class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.model_1 = Unet()
        self.model_2 = Unet()
        self.critic = PixelDiscriminator()
        self.best_mIoU, self.best_dice_coeff = 0, 0
        self._init_configure()
        self._init_logger()

    def _init_configure(self):
        with open('configs/config.yml') as fp:
            self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        log_dir = 'logs/' + opt.dataset + '/train/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.image_save_path_1 = log_dir + "/saved_images_1"
        self.image_save_path_2 = log_dir + "/saved_images_2"

        create_dir(self.image_save_path_1)
        create_dir(self.image_save_path_2)

        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        print('Generator Learning Rate: {} Critic Learning Rate'.format(opt.lr_gen,opt.lr_dis))

        self.model_1.cuda()
        self.model_2.cuda()

        params = list(self.model_1.parameters()) + list(self.model_2.parameters())
        dis_params = self.critic.parameters()
        optimizer = torch.optim.SGD(params, lr=opt.lr_gen, momentum=opt.momentum)
        dis_optimizer = torch.optim.RMSprop(dis_params, opt.lr_dis)

        image_root = self.cfg[opt.dataset]['image_dir']
        gt_root = self.cfg[opt.dataset]['mask_dir']

        self.logger.info("Split Percentage : {} Labeled Data Ratio : {}".format(opt.split, opt.ratio))
        train_loader_1, train_loader_2, unlabeled_train_loader, val_loader = image_loader(image_root, gt_root,
                                                                                          opt.batchsize, opt.trainsize,
                                                                                          opt.split, opt.ratio)
        self.logger.info(
            "train_loader_1 {} train_loader_2 {} unlabeled_train_loader {} val_loader {}".format(len(train_loader_1),
                                                                                                 len(train_loader_2),
                                                                                                 len(
                                                                                                     unlabeled_train_loader),
                                                                                                 len(val_loader)))
        print("Let's go!")

        for epoch in range(1, opt.epoch):

            running_loss = 0.0
            running_dice_val_1 = 0.0
            running_dice_val_2 = 0.0

            for i, data in enumerate(zip(train_loader_1, train_loader_2, unlabeled_train_loader)):
                inputs_S1, labels_S1 = data[0][0], data[0][1]
                inputs_S2, labels_S2 = data[1][0], data[1][1]
                inputs_U, labels_U = data[2][0], data[2][1]

                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()
                inputs_S2, labels_S2 = Variable(inputs_S2), Variable(labels_S2)
                inputs_S2, labels_S2 = inputs_S2.cuda(), labels_S2.cuda()
                inputs_U = Variable(inputs_U)
                inputs_U = inputs_U.cuda()

                optimizer.zero_grad()

                # Train Model 1
                prediction_1 = self.model_1(inputs_S1)
                prediction_1 = torch.sigmoid(prediction_1)

                u_prediction_1 = self.model_1(inputs_U)
                u_prediction_1 = torch.sigmoid(u_prediction_1)

                # Train Model 2
                prediction_2 = self.model_2(inputs_S2)
                prediction_2 = torch.sigmoid(prediction_2)

                u_prediction_2 = self.model_2(inputs_U)
                u_prediction_2 = torch.sigmoid(u_prediction_2)

                self.critic.cuda()

                Loss_sup = loss_sup(prediction_1, prediction_2, labels_S1, labels_S2)
                Loss_diff = loss_diff(u_prediction_1, u_prediction_2, opt.batchsize)

                prediction_1 = prediction_1.detach()
                prediction_2 = prediction_2.detach()

                D_fake_1 = F.interpolate(torch.sigmoid(self.critic(prediction_1)),
                                         (prediction_1.shape[2], prediction_1.shape[3]),
                                         mode='bilinear', align_corners=False)
                D_fake_2 = F.interpolate(torch.sigmoid(self.critic(prediction_2)),
                                         (prediction_2.shape[2], prediction_2.shape[3]),
                                         mode='bilinear', align_corners=False)

                D_fake3 = F.interpolate(torch.sigmoid(self.critic(u_prediction_1)),
                                         (u_prediction_1.shape[2], u_prediction_1.shape[3]),
                                         mode='bilinear', align_corners=False)

                D_fake4 = F.interpolate(torch.sigmoid(self.critic(u_prediction_2)),
                                        (u_prediction_2.shape[2], u_prediction_2.shape[3]),
                                        mode='bilinear', align_corners=False)

                ignore_mask_remain_1 = np.zeros(D_fake3.shape).astype(np.bool)
                ignore_mask_remain_2 = np.zeros(D_fake4.shape).astype(np.bool)

                Loss_adv_labeled = loss_adversarial_1(D_fake_1, D_fake_2, labels_S1, labels_S2)
                Loss_adv_unlabeled = CE(D_fake3, make_Dis_label(gt_label, ignore_mask_remain_1)) + CE(D_fake4, make_Dis_label(gt_label, ignore_mask_remain_2))
                Loss_adv1 = Loss_adv_labeled + Loss_adv_unlabeled
                seg_loss = Loss_sup + 0.4 * Loss_diff + 0.2 * Loss_adv1

                seg_loss.backward()
                running_loss += seg_loss.item()
                optimizer.step()

                # Train Critic
                dis_optimizer.zero_grad()
                prediction_1 = prediction_1.detach()
                prediction_2 = prediction_2.detach()

                D_fake_1 = F.interpolate(torch.sigmoid(self.critic(prediction_1)),
                                         (prediction_1.shape[2], prediction_1.shape[3]),
                                         mode='bilinear', align_corners=False)
                D_fake_2 = F.interpolate(torch.sigmoid(self.critic(prediction_2)),
                                         (prediction_2.shape[2], prediction_2.shape[3]),
                                         mode='bilinear', align_corners=False)
                D_real_1 = F.interpolate(torch.sigmoid(self.critic(labels_S1)),
                                         (labels_S1.shape[2], labels_S1.shape[3]),
                                         mode='bilinear', align_corners=False)
                D_real_2 = F.interpolate(torch.sigmoid(self.critic(labels_S2)),
                                         (labels_S2.shape[2], labels_S2.shape[3]),
                                         mode='bilinear', align_corners=False)

                Loss_adv2 = loss_adversarial_2(D_fake_1, D_real_1, D_fake_2, D_real_2, labels_S1, labels_S2)
                Loss_adv2.backward()
                dis_optimizer.step()

            epoch_loss = running_loss / (len(train_loader_1) + len(train_loader_2))
            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, opt.epoch, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    images = Variable(images)
                    gts = Variable(gts)
                    images = images.cuda()
                    gts = gts.cuda()

                    prediction_1 = self.model_1(images)
                    prediction_1 = torch.sigmoid(prediction_1)

                    prediction_2 = self.model_2(images)
                    prediction_2 = torch.sigmoid(prediction_2)

                dice_coe_1 = dice_coef(prediction_1, gts)
                running_dice_val_1 += dice_coe_1
                dice_coe_2 = dice_coef(prediction_2, gts)
                running_dice_val_2 += dice_coe_2

            epoch_dice_val_1 = running_dice_val_1 / len(val_loader)

            self.logger.info('Validation dice coeff model 1: {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Validation_1/DSC', epoch_dice_val_1, epoch)

            epoch_dice_val_2 = running_dice_val_2 / len(val_loader)

            self.logger.info('Validation dice coeff model 1: {}'.format(epoch_dice_val_2))
            self.writer.add_scalar('Validation_1/DSC', epoch_dice_val_2, epoch)

            mdice_coeff_1 = epoch_dice_val_1
            mdice_coeff_2 = epoch_dice_val_2

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True

                if not os.path.exists(self.image_save_path_1):
                    os.makedirs(self.image_save_path_1)

                copy_tree(self.image_save_path_1, self.save_path + '/best_model_predictions_1')
                self.patience = 0
            else:
                self.save_best_model_1 = False
                self.patience += 1

            if self.best_dice_coeff_2 < mdice_coeff_2:
                self.best_dice_coeff_2 = mdice_coeff_2
                self.save_best_model_2 = True

                if not os.path.exists(self.image_save_path_2):
                    os.makedirs(self.image_save_path_2)

                copy_tree(self.image_save_path_2, self.save_path + '/best_model_predictions_2')
                self.patience = 0
            else:
                self.save_best_model_2 = False
                self.patience += 1

            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                torch.save(self.model_1.state_dict(), Checkpoints_Path + '/Model_1.pth')
                torch.save(self.critic.state_dict(), Checkpoints_Path + '/Critic.pth')

            if self.save_best_model_2:
                torch.save(self.model_2.state_dict(), Checkpoints_Path + '/Model_2.pth')
                torch.save(self.critic.state_dict(), Checkpoints_Path + '/Critic.pth')

            self.logger.info(
                'current best dice coef model 1 {}, model 2 {}'.format(self.best_dice_coeff_1, self.best_dice_coeff_2))
            self.logger.info('current patience :{}'.format(self.patience))


if __name__ == '__main__':
    train_network = Network()
    train_network.run()
