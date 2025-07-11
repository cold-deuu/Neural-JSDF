# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from scipy.io import loadmat, savemat
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import time
import yaml
import os
import gc
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from sdf.robot_sdf import RobotSdfCollisionNet

def free_gpu_memory(*objs):
    for obj in objs:
        try:
            del obj
        except NameError:
            pass
    gc.collect()
    torch.cuda.empty_cache()

def create_dataset():
    try:
        device = torch.device('cuda', 0)
        tensor_args = {'device': device, 'dtype': torch.float32}

        # TensorBoard writer
        writer = SummaryWriter(log_dir=f"runs/canadarm")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', 'data-sampling', 'datasets', 'points.mat')
        data_path = os.path.abspath(data_path)
        data = loadmat(data_path)['total_array']

        augment_data = 1
        for i in range(augment_data):
            data_path = os.path.join(script_dir, '..', 'data-sampling', 'datasets', 'points_augment_' + str(i+1) + '.mat')
            data_path = os.path.abspath(data_path)
            aug_data = loadmat(data_path)['total_array']
            data = np.concatenate([data, aug_data], axis=0)

        N = data.shape[0]
        batch_size   = data.shape[0] // 3
        L_train_test = int(0.9 * N)
        L_val        = N

        train_test_idx = np.arange(0, L_train_test)
        val_idx        = np.arange(L_train_test, L_val)

        perm = np.random.permutation(train_test_idx)

        n_tt = len(train_test_idx)
        n_train = int(0.8 * N)
        n_test  = n_tt - n_train

        train_idx = perm[:n_train]
        test_idx  = perm[n_train:n_train + n_test]

        x = torch.tensor(data[:, 0:10], **tensor_args)
        y = 100 * torch.tensor(data[:, 10:], **tensor_args)

        x_train, y_train = x[train_idx], y[train_idx]
        x_test,  y_test  = x[test_idx],  y[test_idx]
        x_val,   y_val   = x[val_idx],   y[val_idx]

        # y_train_labels = y_train.clone()
        # y_train_labels[y_train_labels <= 0] = -1
        # y_train_labels[y_train_labels > 0]  =  1

        # y_test_labels = y_test.clone()
        # y_test_labels[y_test_labels <= 0] = -1
        # y_test_labels[y_test_labels > 0]  =  1

        # y_val_labels = y_val.clone()
        # y_val_labels[y_val_labels <= 0] = -1
        # y_val_labels[y_val_labels > 0]  =  1

        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test  = x_test.to(device)
        y_test  = y_test.to(device)
        x_val   = x_val.to(device)
        y_val   = y_val.to(device)

        perm = torch.randperm(x_train.size(0), device=device)

        dof = x.shape[1]
        s = 256
        n_layers = 10
        skips = []

        if not skips:
            n_layers -= 1

        nn_model = RobotSdfCollisionNet(
            in_channels=dof,
            out_channels=y.shape[1],
            layers=[s] * n_layers,
            skips=skips
        )
        
        nn_model.model.to(**tensor_args)
        model = nn_model.model

        nelem = sum(param.nelement() for param in model.parameters())
        print(repr(model))
        print(f"Sum of parameters: {nelem}")

        optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)

        epochs = 30000

        # Linear warmup with linear decay
        # steps_per_epoch = (x.shape[0] + batch_size - 1) // batch_size
        # num_training_steps = epochs * steps_per_epoch
        # num_warmup_steps = int(0.1 * num_training_steps)

        # def lr_lambda(cs):
        #     step = cs + 1
        #     if step <= num_warmup_steps:
        #         return step / float(max(1, num_warmup_steps))
        #     return max(0.0, (num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Cosine annealing with warm restart
        num_cosine_periods = 15
        num_training_steps = int(epochs / num_cosine_periods)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_training_steps, T_mult=1, eta_min=0.0)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5000,
        #     threshold=0.01,
        #     threshold_mode='rel',
        #     cooldown=0,
        #     min_lr=0,
        #     eps=1e-04
        # )

        epochs = 30000
        min_val_loss = float('inf')
        scaler = torch.amp.GradScaler('cuda', enabled=True)

        idx_close_train = (y_train[:, -1] < 10)
        idx_close_val = (y_val[:, -1] < 10)

        for e in range(epochs):
            start_time = time.time()

            train_loss = 0.0
            for start in range(0, x_train.size(0), batch_size):
                end = start + batch_size
                idx = perm[start:end]
                xb, yb = x_train[idx], y_train[idx]

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    y_pred = model(xb)
                    loss   = F.mse_loss(y_pred, yb, reduction='mean')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item() * xb.size(0)

            train_loss /= x_train.size(0)

            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    y_pred_test = model(x_test)
                    test_loss = F.mse_loss(y_pred_test, y_test, reduction='mean')

                with torch.amp.autocast('cuda'):
                    y_pred = model(x_val)
                    val_loss = F.mse_loss(y_pred, y_val, reduction='mean')
                    # ee_loss_close = F.l1_loss(
                    #     y_pred[idx_close_val][:, -1],
                    #     y_val[idx_close_val][:, -1],
                    #     reduction='mean'
                    # )

                if val_loss < min_val_loss and e > 100:
                    print(f"Saving model at epoch {e}, Val Loss: {val_loss.item():.4f}")
                    torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'norm': {
                            'x': {'mean': torch.mean(x, dim=0) * 0.0, 'std': torch.std(x, dim=0) * 0.0 + 1.0},
                            'y': {'mean': torch.mean(y, dim=0) * 0.0, 'std': torch.std(y, dim=0) * 0.0 + 1.0}
                        },
                        'train_loss': train_loss,
                        'test_loss': test_loss.item(),
                        'val_loss': val_loss.item(),
                    }, 'canadarm_mesh.pt')
                    min_val_loss = val_loss

            # Logging
            writer.add_scalar('Loss/Train', train_loss, e)
            writer.add_scalar('Loss/Test', test_loss.item(), e)
            writer.add_scalar('Loss/Validation', val_loss.item(), e)
            # writer.add_scalar('Loss/EE_Close', ee_loss_close.item(), e)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], e)

            elapsed = time.time() - start_time
            print(
                f"Epoch {e}: Train Loss={train_loss:.4f}, "
                f"Test Loss={test_loss.item():.4f}, "
                f"Val Loss={val_loss.item():.4f}, "
                # f"(Close={ee_loss_close.item():.4f}), "
                f"Time={elapsed:.2f}s, LR={optimizer.param_groups[0]['lr']:.6f}"
            )

        free_gpu_memory(model, optimizer, scheduler, scaler, x_train, y_train, x_val, y_val, x_test, y_test)
    except KeyboardInterrupt as e:
        free_gpu_memory(model, optimizer, scheduler, scaler, x_train, y_train, x_val, y_val, x_test, y_test)


if __name__ == '__main__':
    create_dataset()
