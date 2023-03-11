import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.optim import Adam
from dataloader import get_data_loader, get_data_loader_cifar100_task0
from network import ResNet18
from resnet18_cifar import resnet18_cifar


class Model:
    def __init__(self,firsttask=50, class_per_task=5, ntasks=11, ascent_num_steps=50, ascent_step_size=0.001, radius=8.0,
                 gamma=2.0):
        self.net_old = None
        self.net_new = resnet18_cifar().cuda()
        print('model initialized')
        self.firsttask = firsttask
        self.class_per_task = class_per_task
        self.ntasks = ntasks
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        self.test_loaders = []
        self.writer = None
        self.gamma = gamma
        self.radius = radius

    def train(self, niter, lr=1e-3, beta=1, gamma=1):
        self.writer = tensorboard.SummaryWriter('log')
        print('tensorboard initialized')

        ############################################################
        # training on the first task
        ############################################################

        train_loader, test_loader = get_data_loader_cifar100_task0(0)
        print('get data of task %d' % 0)
        self.test_loaders.append(train_loader)
        opt = Adam(self.net_new.parameters(), lr=lr)
        for epoch in range(niter):

            self.net_new.train()

            losses, loss_Cs = [], []

            ########################################################

            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                x = x.to(torch.float)
                batch_size = len(x)
                # Randomly sample points around the training data
                # We will perform SGD on these to find the adversarial points
                x_adv = torch.randn(x.shape).cuda().detach().requires_grad_()
                x_adv_sampled = x_adv + x

                for step in range(self.ascent_num_steps):
                    with torch.enable_grad():

                        new_targets = torch.zeros(batch_size, 1).cuda()
                        new_targets = torch.squeeze(new_targets)
                        new_targets = new_targets.to(torch.float)
                        y_pred1 = self.net_new(x_adv_sampled)[:, :self.firsttask+1]
                        logits = y_pred1[:, -1:]
                        logits = torch.squeeze(logits, dim=1)
                        new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
                        grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                        grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                        grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                        grad_normalized = grad / grad_norm
                    with torch.no_grad():
                        x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

                    if (step + 1) % 10 == 0:
                        # Project the normal points to the set N_i(r)
                        h = x_adv_sampled - x
                        norm_h = torch.sqrt(torch.sum(h ** 2,
                                                      dim=tuple(range(1, h.dim()))))
                        alpha = torch.clamp(norm_h, self.radius,
                                            self.gamma * self.radius).cuda()
                        # Make use of broadcast to project h
                        proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
                        h = proj * h
                        x_adv_sampled = x + h  # These adv_points are now on the surface of hyper-sphere

                x_adv_sampled1 = torch.empty((0, 3, 32, 32)).cuda()
                new_targets_1 = torch.empty(0).cuda()
                a = np.zeros((len(y), self.firsttask+1))
                for i in range(len(y)):
                    a[i][y[i]] = 1
                    a[i][self.firsttask] = 1
                b = a * 0
                a = torch.from_numpy(a).cuda()
                b = torch.from_numpy(b).cuda()
                for i in range(len(new_targets)):
                    x_adv_sampled1 = torch.cat([x_adv_sampled1, x[i].reshape((1, 3, 32, 32))], dim=0)
                    x_adv_sampled1 = torch.cat([x_adv_sampled1, x_adv_sampled[i].reshape((1, 3, 32, 32))], dim=0)
                    new_targets_1 = torch.cat([new_targets_1, a[i].reshape(1, self.firsttask+1)], dim=0)
                    new_targets_1 = torch.cat([new_targets_1, b[i].reshape(1, self.firsttask+1)], dim=0)
                y_pred2 = self.net_new(x_adv_sampled1)[:, :self.firsttask+1]

                y_pred_lwm = torch.empty(0).cuda()
                y_pred_lwm_targets = torch.empty(0).cuda()
                for i in range(len(new_targets)):
                    y_pred_lwm = torch.cat([y_pred_lwm, y_pred2[2 * i: 2 * i + 1, : - 1].reshape((1, self.firsttask))], dim=0)
                    y_pred_lwm_targets = torch.cat([y_pred_lwm_targets, new_targets_1[2 * i: 2 * i + 1, : - 1].reshape((1, self.firsttask))], dim=0)

                loss_C = F.binary_cross_entropy_with_logits(y_pred_lwm, y_pred_lwm_targets).mean()

                loss_Drocc = F.binary_cross_entropy_with_logits(y_pred2[:, -1:], new_targets_1[:, -1:]).mean()

                # loss_C = F.binary_cross_entropy_with_logits(y_pred2, new_targets_1)
                # loss = loss_C
                loss = loss_C + (1/self.firsttask)*loss_Drocc
                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
                loss_Cs.append(loss_C.item())

            ########################################################

            loss = np.array(losses, dtype=np.float).mean()
            loss_C = np.array(loss_Cs, dtype=np.float).mean()

            print('| epoch %d, task %d, train_loss %.4f' % (epoch, 0, loss))

            self.writer.add_scalar('train_loss', loss, epoch)
            self.writer.add_scalar('train_loss_C', loss_C, epoch)

            if epoch % 2 == 0:
                self.test(epoch, self.firsttask + 1)

        ############################################################
        # training on the other tasks
        ############################################################

        for ntask in range(1, self.ntasks):
            # copying the old network
            self.net_new.feature = None
            self.net_old = copy.deepcopy(self.net_new)
            self.net_old.requires_grad_(False)

            # get data
            train_loader, test_loader = get_data_loader(ntask-1+self.firsttask//self.class_per_task)
            self.test_loaders.append(train_loader)

            # transferring
            self.transfer(ntask, niter, train_loader, lr, beta, gamma, ntask * (self.class_per_task+1) + (self.firsttask+1))

    def grad_cam_loss(self, feature_o, out_o, feature_n, out_n):
        batch = out_n.size()[0]
        index = out_n.argmax(dim=-1).view(-1, 1)
        onehot = torch.zeros_like(out_n)
        onehot.scatter_(-1, index, 1.)
        out_o, out_n = torch.sum(onehot * out_o), torch.sum(onehot * out_n)

        grads_o = torch.autograd.grad(out_o, feature_o)[0]
        grads_n = torch.autograd.grad(out_n, feature_n, create_graph=True)[0]
        weight_o = grads_o.mean(dim=(2, 3)).view(batch, -1, 1, 1)
        weight_n = grads_n.mean(dim=(2, 3)).view(batch, -1, 1, 1)

        cam_o = F.relu((grads_o * weight_o).sum(dim=1))
        cam_n = F.relu((grads_n * weight_n).sum(dim=1))

        # normalization
        cam_o = F.normalize(cam_o.view(batch, -1), p=2, dim=-1)
        cam_n = F.normalize(cam_n.view(batch, -1), p=2, dim=-1)

        loss_AD = (cam_o - cam_n).norm(p=1, dim=1).mean()
        return loss_AD

    def transfer(self, ntask, niter, train_loader, lr, beta, gamma, lim):
        opt = Adam(self.net_new.parameters(), lr=lr)

        for epoch in range(niter):

            self.net_new.train()
            self.net_old.train()

            losses, loss_Cs, loss_Ds, loss_ADs = [], [], [], []

            ########################################################

            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                x = x.to(torch.float)
                batch_size = len(x)
                # Randomly sample points around the training data
                # We will perform SGD on these to find the adversarial points
                x_adv = torch.randn(x.shape).cuda().detach().requires_grad_()
                x_adv_sampled = x_adv + x

                for step in range(self.ascent_num_steps):
                    with torch.enable_grad():

                        new_targets = torch.zeros(batch_size, 1).cuda()
                        new_targets = torch.squeeze(new_targets)
                        new_targets = new_targets.to(torch.float)
                        y_pred1 = self.net_new(x_adv_sampled)[:, :lim]
                        logits = y_pred1[:, -1:]
                        logits = torch.squeeze(logits, dim=1)
                        new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
                        grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                        grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                        grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                        grad_normalized = grad / grad_norm
                    with torch.no_grad():
                        x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

                    if (step + 1) % 10 == 0:
                        # Project the normal points to the set N_i(r)
                        h = x_adv_sampled - x
                        norm_h = torch.sqrt(torch.sum(h ** 2,
                                                      dim=tuple(range(1, h.dim()))))
                        alpha = torch.clamp(norm_h, self.radius,
                                            self.gamma * self.radius).cuda()
                        # Make use of broadcast to project h
                        proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
                        h = proj * h
                        x_adv_sampled = x + h  # These adv_points are now on the surface of hyper-sphere

                x_adv_sampled1 = torch.empty((0, 3, 32, 32)).cuda()
                new_targets_1 = torch.empty(0).cuda()
                a = np.zeros((len(y), self.class_per_task+1))
                for i in range(len(y)):
                    a[i][y[i] - self.class_per_task * (ntask - 1) - self.firsttask] = 1
                    a[i][self.class_per_task] = 1
                b = a * 0

                a = torch.from_numpy(a).cuda()
                b = torch.from_numpy(b).cuda()
                for i in range(len(new_targets)):
                    x_adv_sampled1 = torch.cat([x_adv_sampled1, x[i].reshape((1, 3, 32, 32))], dim=0)
                    x_adv_sampled1 = torch.cat([x_adv_sampled1, x_adv_sampled[i].reshape((1, 3, 32, 32))], dim=0)
                    new_targets_1 = torch.cat([new_targets_1, a[i].reshape(1, self.class_per_task+1)], dim=0)
                    new_targets_1 = torch.cat([new_targets_1, b[i].reshape(1, self.class_per_task+1)], dim=0)
                y_pred_old = self.net_old(x_adv_sampled1)[:, :lim - self.class_per_task - 1]
                y_pred_new = self.net_new(x_adv_sampled1)[:, :lim]

                y_pred_lwm = torch.empty(0).cuda()
                y_pred_lwm_targets = torch.empty(0).cuda()
                for i in range(len(new_targets)):
                    y_pred_lwm = torch.cat([y_pred_lwm, y_pred_new[2*i: 2*i+1, lim - self.class_per_task - 1: lim - 1].reshape((1, self.class_per_task))], dim=0)
                    y_pred_lwm_targets = torch.cat([y_pred_lwm_targets, new_targets_1[2 * i: 2 * i + 1, : - 1].reshape((1, self.class_per_task))], dim=0)

                loss_C = F.binary_cross_entropy_with_logits(y_pred_lwm, y_pred_lwm_targets).mean()

                loss_Drocc = F.binary_cross_entropy_with_logits(y_pred_new[:, -1:], new_targets_1[:, -1:]).mean()

                # loss_C = F.binary_cross_entropy_with_logits(y_pred_new[:, -self.class_per_task - 1:], new_targets_1).mean()

                loss_D = F.binary_cross_entropy_with_logits(y_pred_new[:, :-self.class_per_task-1], y_pred_old.detach().sigmoid())

                loss_AD = self.grad_cam_loss(self.net_old.feature, y_pred_old, self.net_new.feature, y_pred_new[:, :-self.class_per_task-1])

                loss = loss_C + (1/self.class_per_task)*loss_Drocc + loss_D * beta + loss_AD * gamma

                # loss = loss_C + loss_D * beta + loss_AD * gamma

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
                loss_Cs.append(loss_C.item())
                loss_Ds.append(loss_D.item())
                loss_ADs.append(loss_AD.item())

                torch.cuda.empty_cache()

            ########################################################

            loss = np.array(losses, dtype=np.float).mean()
            loss_C = np.array(loss_Cs, dtype=np.float).mean()
            loss_D = np.array(loss_Ds, dtype=np.float).mean()
            loss_AD = np.array(loss_ADs, dtype=np.float).mean()

            print('| epoch %d, task %d, train_loss %.4f, train_loss_C %.4f, train_loss_D %.4f, train_loss_AD %.4f' % (
            epoch, ntask, loss, loss_C, loss_D, loss_AD))

            self.writer.add_scalar('train_loss', loss, epoch + niter * ntask)
            self.writer.add_scalar('train_loss_C', loss_C, epoch + niter * ntask)
            self.writer.add_scalar('train_loss_D', loss_D, epoch + niter * ntask)
            self.writer.add_scalar('train_loss_AD', loss_AD, epoch + niter * ntask)

            if epoch % 2 == 0:
                self.test(epoch + niter * ntask, lim)
            os.makedirs("params", exist_ok=True)
            torch.save(self.net_new.state_dict(), 'params/lwm_drocc_100_5.pt')

    def test(self, total_epoch, lim):
        self.net_new.eval()
        with torch.no_grad():
            cor_num_lwf, total_num_lwf = 0, 0
            cor_num_drocc, total_num_drocc = 0, 0
            cor_num_all, total_num_all = 0, 0
            for ntask, test_loader in enumerate(self.test_loaders):
                correct_num_lwf, total_lwf = 0, 0
                correct_num_drocc, total_drocc = 0, 0
                correct_num_all, total_all = 0, 0
                for x, y in test_loader:
                    x, y = x.cuda(), y.numpy()
                    # ------------------------
                    # task incremental setting
                    # y_pred = self.net_new(x)[:, ntask * self.class_per_task:(ntask + 1) * self.class_per_task].cpu().numpy()
                    # y -=  ntask * self.class_per_task
                    # ------------------------
                    # class incremental setting
                    y_pred = self.net_new(x)[:, :lim].cpu().numpy()
                    # print(y_pred[0])
                    # print(y[0])
                    y_lwf = np.empty(shape=[len(y), 0])
                    y_drocc = np.empty(shape=[len(y), 0])
                    y_lwm_drocc = np.empty(shape=[len(y), 0])
                    for i in range(ntask + 1):
                        if i == 0:
                            y_lwf = np.append(y_lwf, y_pred[:, 0:self.firsttask], axis=1)
                            y_drocc = np.append(y_drocc, y_pred[:, self.firsttask:self.firsttask + 1], axis=1)
                            y_lwm_drocc = np.append(y_lwm_drocc, y_pred[:, 0:self.firsttask] * y_pred[:, self.firsttask:self.firsttask + 1], axis=1)
                        else:
                            y_lwf = np.append(y_lwf, y_pred[:, (self.firsttask + 1) + (self.class_per_task + 1) * (i-1):(self.firsttask + 1) + (self.class_per_task + 1) * (i-1) + self.class_per_task], axis=1)
                            y_drocc = np.append(y_drocc, y_pred[:, (self.firsttask + 1) + (self.class_per_task + 1) * (i-1) + self.class_per_task:(self.firsttask + 1) + (self.class_per_task + 1) * (i-1) + self.class_per_task + 1], axis=1)
                            y_lwm_drocc = np.append(y_lwm_drocc, y_pred[:, (self.firsttask + 1) + (self.class_per_task + 1) * (i-1):(self.firsttask + 1) + (self.class_per_task + 1) * (i-1) + self.class_per_task]
                                                    * y_pred[:, (self.firsttask + 1) + (self.class_per_task + 1) * (i-1) + self.class_per_task:(self.firsttask + 1) + (self.class_per_task + 1) * (i-1) + self.class_per_task + 1], axis=1)
                    y_all = y_lwm_drocc
                    # ------------------------
                    y_lwf = y_lwf.argmax(axis=-1)
                    correct_num_lwf += (y_lwf == y).sum()
                    total_lwf += y.shape[0]
                    # print(y_pred[0])
                    # print(y_drocc[0])
                    y_drocc = y_drocc.argmax(axis=-1)
                    for i in range(len(y_drocc)):
                        if (y_drocc[i] == (y[i] // self.firsttask)) and (y_drocc[i] == 0):
                            correct_num_drocc = correct_num_drocc + 1
                        elif y_drocc[i] == ((y[i] - self.firsttask) // self.class_per_task) + 1:
                            correct_num_drocc = correct_num_drocc + 1
                    # correct_num_drocc += (y_drocc == (y // self.class_per_task)).sum()
                    total_drocc += y.shape[0]

                    y_all = y_all.argmax(axis=-1)
                    correct_num_all += (y_all == y).sum()
                    total_all += y.shape[0]
                acc_lwf = correct_num_lwf / total_lwf * 100
                cor_num_lwf += correct_num_lwf
                total_num_lwf += total_lwf

                acc_drocc = correct_num_drocc / total_drocc * 100
                cor_num_drocc += correct_num_drocc
                total_num_drocc += total_drocc

                acc_all = correct_num_all / total_all * 100
                cor_num_all += correct_num_all
                total_num_all += total_all
                self.writer.add_scalar('test_acc_%d' % ntask, acc_lwf, total_epoch)
                print('test_acc_lwf_%d %.4f' % (ntask, acc_lwf))
                print('test_acc_drocc_%d %.4f' % (ntask, acc_drocc))
                print('test_acc_all_%d %.4f' % (ntask, acc_all))
            acc_lwf = cor_num_lwf / total_num_lwf * 100
            acc_drocc = cor_num_drocc / total_num_drocc * 100
            acc_all = cor_num_all / total_num_all * 100
            self.writer.add_scalar('test_acc_total', acc_lwf, total_epoch)
            print('test_acc_total_lwf %.4f' % acc_lwf)
            print('test_acc_total_drocc %.4f' % acc_drocc)
            print('test_acc_total_all %.4f' % acc_all)
            # return acc_all
