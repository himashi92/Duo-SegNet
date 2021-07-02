import numpy as np
import torch
from torch.autograd import Variable

from metrics import dice_coef

CE = torch.nn.BCELoss()

# labels for adversarial training
pred_label = 0
gt_label = 1


def make_Dis_label(label, gts):
    D_label = np.ones(gts.shape) * label
    D_label = Variable(torch.FloatTensor(D_label)).cuda()

    return D_label


def calc_loss(pred, target, bce_weight=0.5):
    bce = CE(pred, target)
    dl = 1 - dice_coef(pred, target)
    loss = bce * bce_weight + dl * bce_weight

    return loss


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    loss1 = calc_loss(logit_S1, labels_S1)
    loss2 = calc_loss(logit_S2, labels_S2)

    return loss1 + loss2


def loss_diff(u_prediction_1, u_prediction_2, batch_size):
    a = CE(u_prediction_1, Variable(u_prediction_2, requires_grad=False))
    a = a.item()

    b = CE(u_prediction_2, Variable(u_prediction_1, requires_grad=False))
    b = b.item()

    loss_diff_avg = (a + b)
    return loss_diff_avg / batch_size


def loss_adversarial_1(D_fake_1, D_fake_2, labels_S1, labels_S2):
    D_loss_fake_0_1 = CE(D_fake_1, make_Dis_label(pred_label, labels_S1))
    D_loss_fake_0_2 = CE(D_fake_2, make_Dis_label(pred_label, labels_S2))

    loss = D_loss_fake_0_1 + D_loss_fake_0_2
    return loss


def loss_adversarial_2(D_fake_1, D_real_1, D_fake_2, D_real_2, labels_S1, labels_S2):
    D_loss_fake_0_1 = CE(D_fake_1, make_Dis_label(pred_label, labels_S1))
    D_loss_fake_0_2 = CE(D_fake_2, make_Dis_label(pred_label, labels_S2))

    D_loss_real_1 = CE(D_real_1, make_Dis_label(gt_label, labels_S1))
    D_loss_real_2 = CE(D_real_2, make_Dis_label(gt_label, labels_S2))

    loss = D_loss_fake_0_1 + D_loss_fake_0_2 + D_loss_real_1 + D_loss_real_2
    return loss
