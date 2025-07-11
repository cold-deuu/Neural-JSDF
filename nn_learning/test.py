import os
import torch
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_args = {'device': device, 'dtype': torch.float32}

dof = 10 # 7 + 3
output_dim = 8 # Panda link 0 ~ Panda link 8, hand 총 9개
model = RobotSdfCollisionNet(in_channels=dof, out_channels=output_dim, layers=[256]*4, skips=[]).model
model.to(**tensor_args)
model.eval()
# 학습된 weight 로드
# checkpoint = torch.load('/home/chan/git/Neural-JSDF/canadarm_mesh.pt', map_location=device)
script_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(script_dir, 'pt', 'best_canadarm_MLPRegressionDropout.pt')
data_path = os.path.join(script_dir, 'pt', 'best_canadarm_MLPWithResidualNorm.pt')
data_path = os.path.abspath(data_path)

checkpoint = torch.load(data_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Input : Homepose of Panda, Point (1.0, 0.0, 0.5)
x_input = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]]).to(**tensor_args)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data-sampling', 'datasets', 'test_points.mat')
data_path = os.path.abspath(data_path)
data = loadmat(data_path)['total_array']

x = torch.tensor(data[:, 0:10], **tensor_args)
y = torch.tensor(data[:, 10:], **tensor_args)

# Output : Distance betw links and point
model.eval()
with torch.no_grad():
    with torch.amp.autocast('cuda'):
        y_output = 0.01 * model(x)
        print(y_output.shape)

import random
idx = random.sample(range(x.shape[0]), 3)

for i in idx:
    print("x " + str(i) + " :", x[i,:].cpu().numpy())
    print("y true  :", y[i,:].cpu().numpy())
    print("y output:", y_output[i,:].cpu().numpy())
    print("y error :", torch.abs(y[i,:]-y_output[i,:]).cpu().numpy())
    print("y err % :", torch.div(100 * torch.abs(y[i,:]-y_output[i,:]), y[i,:]).cpu().numpy())
    print("y min   :", torch.min(y[i,:]).cpu().numpy())
    print("yout min:", torch.min(y_output[i,:]).cpu().numpy())
    print("yerr max:", torch.max(torch.abs(y[i,:]-y_output[i,:])).cpu().numpy())
