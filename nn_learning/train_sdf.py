#
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
from scipy.io import loadmat, savemat
import torch
import torch.nn.functional as F
import numpy as np
import time
import yaml

import os
import matplotlib.pyplot as plt
from sdf.robot_sdf import RobotSdfCollisionNet


def create_dataset(robot_name):
    device = torch.device('cuda', 0)
    tensor_args = {'device': device, 'dtype': torch.float32}
    #    data = loadmat('../weights/table_beeger.mat')['data']
    data = loadmat('/home/user/Neural-JSDF/data-sampling/datasets/points.mat')['total_array']
    # data = data[:20000,:]
    # idx_good = (data[:,10:-1] > -0.03).all(1)
    # data = data[idx_good, :]
    # data = data[torch.randperm(data.shape[0])]
    print(data.shape)
    L1 = 0
    L2 = int(0.95 * data.shape[0])
    print(L1, L2)
    n_size = L2
    train_ratio = 0.8
    test_ratio = 0.1
    val_ratio = 1 - train_ratio - test_ratio

    # MLP : Train/ Test/ ValidationPart 로 구성되어있음
    idx_train = np.arange(0, int(n_size * train_ratio))
    idx_val = np.arange(idx_train[-1] + 1, int(n_size * (train_ratio + test_ratio)))
    idx_test = np.arange(idx_val[-1] + 1, int(n_size))

    # Input 은 Joint Posture (7) 그리고 Sample Point (3)
    x = torch.Tensor(data[L1:L2, 0:10]).to(device, dtype=torch.float16)
    # 100을 곱해서 cm Scle 에서 Output 결정
    y = 100 * torch.Tensor(data[L1:L2, 10:]).to(device, dtype=torch.float16)
    #y[y<0]*=5
    #y[y==0] = 1
    dof = x.shape[1]
    s = 256
    n_layers = 5
    skips = []
    fname = 'canadarm_mesh.pt'
    if skips == []:
        n_layers-=1

    # in channels : 7 + 3
    # out channels : n_meshes
    # Layers : [256 , 256, 256, 256, 256]
    nn_model = RobotSdfCollisionNet(in_channels=dof, out_channels=y.shape[1], layers=[s] * n_layers, skips=skips)
    #nn_model.load_weights('../scripts/sdf_convex_256x5_mesh.pt', tensor_args)
    #nn_model.load_weights('../scripts/gridsearch/5_sdf_convex_512x5.pt', tensor_args)
    #nn_model.load_weights('../scripts/raesdf_256x5_mesh.pt', tensor_args)

    nn_model.model.to(**tensor_args)

    model = nn_model.model
    
    # parameter 는 뭐지
    # print(model.parameters())
    # model.parameters() >> nn.Module.parameters()
    # Parameter 는 각 Layer 의 Weight 와 bias 로 구성되어있음
    # 이건 신경망의 학습 가능한 파라미터들(weights, bias)를 반환하는 함수
    # Sum of Parametrs 는 각 파라미터의 원소 수를 의미함
    # ex. Layer1(30,256) --> n_param = 256 * 30
    # ex. Layer2(256,256) --> n_param = 256 * 256
    # 지금 30, 256, 256, 256, 256, 9 
    # 따라서 Weight 1 : 256 * 30 bias 1 : 256
    # W2 : 256 256, b2 : 256
    # W3 : 256 256, b3 : 256
    # W4 : 256, 256, b4 : 256
    # W5 : 9, 256, b5 : 9
    # n1 = 256 * 30 + 256
    # n2 = 256 * 256 + 256
    # n3 = 256 * 256 + 256
    # n4 = 256 * 256 + 256
    # n5 = 9 * 256 + 9

    # print("N : ",n1 + n2 + n3 + n4 + n5)

    nelem = sum([param.nelement() for param in model.parameters()])

    print(repr(model))
    print("Sum of parameters:%d" % nelem)

    #time.sleep(2)
    # load training set:
    x_train = x[idx_train, :]
    y_train = y[idx_train, :]
    y_train_labels = y[idx_train, :]
    y_train_labels[y_train_labels<=0] = -1
    y_train_labels[y_train_labels>0] = 1

    y_val_labels = y[idx_val, :]
    y_val_labels[y_val_labels<=0] = -1
    y_val_labels[y_val_labels>0] = 1

    # scale dataset: (disabled because of nerf features!)
    mean_x = torch.mean(x, dim=0) * 0.0
    std_x = torch.std(x, dim=0) * 0.0 + 1.0
    mean_y = torch.mean(y, dim=0) * 0.0
    std_y = torch.std(y, dim=0) * 0.0 + 1.0

    x_val = x[idx_val, :]
    y_val = y[idx_val, :]
    x_test = x[idx_test, :]
    y_test = y[idx_test, :]

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                           threshold=0.01, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-04)

    # print(model)
    epochs = 100000
    min_loss = 2000.0
    # training:
    e_notsaved = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    idx_close_train = (y_train[:, -1] < 10)
    idx_close_val = (y_val[:, -1] < 10)

    for e in range(epochs):
        t0 = time.time()
        model.train()
        loss = []
        i = 0
        with torch.cuda.amp.autocast():
            y_pred_train = (model.forward(x_train))
            train_loss = F.mse_loss(y_pred_train, y_train, reduction='mean')

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        loss.append(train_loss.item())

        model.eval()
        with torch.cuda.amp.autocast():
            y_pred = model.forward(x_val)
            val_loss = F.mse_loss(y_pred, y_val, reduction='mean')
            ee_loss_close = F.l1_loss(y_pred[idx_close_val][:,-1], y_val[idx_close_val][:,-1], reduction='mean')
        if e == 0:
            min_loss = val_loss
        scheduler.step(val_loss)
        train_loss = np.mean(loss)
        e_notsaved += 1
        if (val_loss < min_loss and e > 100 and e_notsaved > 100):
            e_notsaved = 0
            print('saving model', val_loss.item())
            torch.save(
                {
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'norm': {'x': {'mean': mean_x, 'std': std_x},
                             'y': {'mean': mean_y, 'std': std_y}}
                },
                fname)
            # 이거 걍 Weight 를 저장하는게 나을수도? SCA 처럼?
                #'sdf_convex_256_mlp_nerf_skip.pt')
            min_loss = val_loss
            # print(y_pred[0, :])
            # print(y_val[0, :])
            # if e > 1500:
            #     break
        print(
            "Epoch: %d (Saved at %d), Train Loss: %4.3f, Validation Loss: %4.3f (%4.3f), Epoch time: %4.3f s, LR = %4.8f" % (
                e, e-e_notsaved+1, train_loss.item(), val_loss.item(), ee_loss_close.item(), time.time() - t0,
                optimizer.param_groups[0]["lr"]))

    # with torch.no_grad():
    #     x = x_test  # [y_test[:,0] > 0.0]
    #     y = y_test  # [y_test[:,0] > 0.0]
    
    #     # print(x.shape)
    #     y_pred = model.forward(x)
    #     y_pred = torch.mul(y_pred, std_y) + mean_y
    #     y_test = torch.mul(y, std_y) + mean_y
    #     print(y_test[y_test > 0.0])
    #     print(y_pred[y_test > 0.0])
    #     # print(y_pred.shape, y_test.shape)
    #     loss = F.l1_loss(y_pred, y_test, reduction='mean')
    #     print(torch.median(y_pred), torch.mean(y_pred))
    #     print(loss.item())


if __name__ == '__main__':
    create_dataset('franka')