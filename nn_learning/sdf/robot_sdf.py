#implementation is based on https://github.com/NVlabs/storm

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, ReLU6, Tanh
#from .network_macros_mod import MLPRegression, scale_to_base, scale_to_net
from .network_macros_mod import *
from .network_test import *
#from .util_file import *
from functorch import vmap, jacrev
from functorch.compile import aot_function


class RobotSdfCollisionNet():
    """This class loads a network to predict the signed distance given a robot joint config."""
    def __init__(self, in_channels, out_channels, skips, layers):

        super().__init__()
        act_fn = ReLU
        in_channels = in_channels
        self.out_channels = out_channels
        head = 0.1
        dropout_ratio = 0.1
        mlp_layers = layers
        # self.model = MLPRegression(in_channels, self.out_channels, mlp_layers, skips, act_fn=act_fn, nerf=True)
        self.model = MLPRegressionDropout(in_channels, self.out_channels, mlp_layers, skips, act_fn=act_fn, dropout=dropout_ratio, nerf=True)
        # self.model = MLPWithResidual(in_channels, self.out_channels, nerf=True)
        # self.model = ResMLP(in_channels, self.out_channels, depth=5)
        # self.model = MLPWithResidualNorm(in_channels, self.out_channels, nerf=True)

        self.m = torch.zeros((500, 1)).to('cuda:0')
        self.m[:, 0] = 1
        self.order = list(range(out_channels))

    def set_link_order(self, order):
        self.order = order

    def load_weights(self, f_name, tensor_args):
        """Loads pretrained network weights if available.

        Args:
            f_name (str): file name, this is relative to weights folder in this repo.
            tensor_args (Dict): device and dtype for pytorch tensors
        """
        try:
            chk = torch.load(f_name)
            self.model.load_state_dict(chk["model_state_dict"])
            self.norm_dict = chk["norm"]
            for k in self.norm_dict.keys():
                self.norm_dict[k]['mean'] = self.norm_dict[k]['mean'].to(**tensor_args)
                self.norm_dict[k]['std'] = self.norm_dict[k]['std'].to(**tensor_args)
            print('Weights loaded!')
        except Exception as E:
            print('WARNING: Weights not loaded')
            print(E)
        self.model = self.model.to(**tensor_args)
        self.tensor_args = tensor_args
        self.model.eval()

    def compute_signed_distance(self, q):
        """Compute the signed distance given the joint config.

        Args:
            q (tensor): input batch of joint configs [b, n_joints]

        Returns:
            [tensor]: largest signed distance between any two non-consecutive links of the robot.
        """
        with torch.no_grad():
            q_scale = scale_to_net(q, self.norm_dict, 'x')
            dist = self.model.forward(q_scale)
            dist_scale = scale_to_base(dist, self.norm_dict, 'y')
        return dist_scale[:, self.order].detach()

    def compute_signed_distance_wgrad(self, q, idx = 'all'):
        minidxMask = torch.zeros(q.shape[0])
        if idx == 'all':
            idx = list(range(self.out_channels))
        if self.out_channels == 1:
            with torch.enable_grad():
                q.requires_grad = True
                q.grad = None
                q_scale = scale_to_net(q, self.norm_dict, 'x')
                dist = self.model.forward(q_scale)
                dist_scale = scale_to_base(dist, self.norm_dict, 'y').detach()
                m = torch.zeros((q.shape[0], dist.shape[1])).to(q.device)
                m[:, 0] = 1
                dist.backward(m)
                grads = q.grad.detach()
                # jac = torch.autograd.functional.jacobian(self.model, q_scale)
        else:
            with torch.enable_grad():
                #https://discuss.pytorch.org/t/derivative-of-model-outputs-w-r-t-input-features/95814/2
                q.requires_grad = True
                q.grad = None
                #removed scaling as we don't use it
                dist_scale = self.model.forward(q)
                dist_scale = dist_scale[:, self.order]
                minidxMask = torch.argmin(dist_scale, dim=1)
                grd = torch.zeros((q.shape[0], self.out_channels), device = q.device, dtype = q.dtype) # same shape as preds
                if type(idx) == list:
                    grads = torch.zeros((q.shape[0], q.shape[1], len(idx)))
                    for k, i in enumerate(idx):
                        grd *= 0
                        grd[:, i] = 1  # column of Jacobian to compute
                        dist_scale.backward(gradient=grd, retain_graph=True)
                        grads[:, :, k] = q.grad  # fill in one column of Jacobian
                        q.grad.zero_()  # .backward() accumulates gradients, so reset to zero
                else:
                    grads = torch.zeros((q.shape[0], q.shape[1], 1))
                    grd[list(range(q.shape[0])), minidxMask] = 1
                    dist_scale.backward(gradient=grd, retain_graph=False)
                    grads[:, :, 0] = q.grad  # fill in one column of Jacobian
                    #q.grad.zero_()  # .backward() accumulates gradients, so reset to zero
                    for param in self.model.parameters():
                        param.grad = None
        return dist_scale.detach(), grads.detach(), minidxMask.detach()

    def functorch_jacobian(self, points):
        """calculate a jacobian tensor along a batch of inputs. returns something of size
        `batch_size` x `output_dim` x `input_dim`"""
        return vmap(jacrev(self.model))(points)

    def pytorch_jacobian(self, points):
        """calculate a jacobian tensor along a batch of inputs. returns something of size
        `batch_size` x `output_dim` x `input_dim`"""
        def _func_sum(points):
            return self.model(points).sum(dim=0)
        return torch.autograd.functional.jacobian(_func_sum, points, create_graph=True, vectorize=True).permute(1, 0, 2)

    def functorch_jacobian2(self, points):
        """calculate a jacobian tensor along a batch of inputs. returns something of size
        `batch_size` x `output_dim` x `input_dim`"""
        def _func_sum(points):
            return self.model(points).sum(dim=0)
        return jacrev(_func_sum)(points).permute(1, 0, 2)

    def ts_compile(self, fx_g, inps):
        print("compiling")
        f = torch.jit.script(fx_g)
        f = torch.jit.freeze(f.eval())
        return f

    def ts_compiler(self, f):
        return aot_function(f, self.ts_compile, self.ts_compile)

