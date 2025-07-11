import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRegressionDropout(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, mlp_layers: list[int], skips: list[int] = [], 
                 act_fn: type[nn.Module] = nn.ReLU, dropout: float = 0.1, nerf: bool = True):
        super().__init__()
        if nerf:
            input_dims *= 3

        mlp_arr: list[list[int]] = []
        if len(skips) == 0:
            mlp_arr.append(mlp_layers.copy())
        mlp_arr[-1].append(output_dims)
        mlp_arr[0].insert(0, input_dims)

        # MLP Layer : [[30, 256, 256, 256, 256, 8]]
        # print(f"MLP Layer : {mlp_arr}")

        self.layers = nn.ModuleList()
        channels = mlp_arr[0]
        blocks: list[nn.Module] = []
        for i in range(len(channels) - 1):
            in_c, out_c = channels[i], channels[i+1]
            blocks.append(nn.Linear(in_c, out_c, bias=True))
            if i < len(channels) - 2:
                blocks.append(act_fn())
                blocks.append(nn.Dropout(dropout))
        self.layers.append(nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        return self.layers[0](x_enc)
    

class MLPWithResidual(nn.Module):
    def __init__(self, input_dims, output_dims, nerf=True):
        super().__init__()
        if nerf:
            input_dims *= 3

        self.proj = nn.Linear(input_dims, 256)

        self.block1_fc1 = nn.Linear(256, 256)
        self.block1_fc2 = nn.Linear(256, 256)
        self.block2_fc1 = nn.Linear(256, 256)
        self.block2_fc2 = nn.Linear(256, 256)

        self.out_fc = nn.Linear(256, output_dims)

        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)

        h = self.act(self.proj(x))

        res = h
        h = self.act(self.block1_fc1(h))
        h = self.act(self.block1_fc2(h))
        h = h + res

        res = h
        h = self.act(self.block2_fc1(h))
        h = self.act(self.block2_fc2(h))
        h = h + res

        return self.out_fc(h)


class ResMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        y = self.norm1(x)
        y = self.token_mix(y)
        x = x + y
        y = self.norm2(x)
        y = self.channel_mix(y)
        return x + y


class ResMLP(nn.Module):
    def __init__(self,
                 in_dim: int = 10,
                 out_dim: int = 8,
                 depth: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, in_dim)
        self.blocks = nn.ModuleList([
            ResMLPBlock(in_dim) for _ in range(depth)
        ])
        self.output_proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
    

class MLPWithResidualNorm(nn.Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 nerf: bool = True,
                 hidden_dim: int = 256,
                 num_blocks: int = 2):
        super().__init__()
        self.input_dims = input_dims * 3 if nerf else input_dims

        self.proj    = nn.Linear(self.input_dims, hidden_dim)
        self.norm0   = nn.LayerNorm(hidden_dim)
        self.act0    = nn.ReLU(inplace=True)

        self.blocks  = nn.ModuleList([
            ResidualBlock(hidden_dim)
            for _ in range(num_blocks)
        ])

        self.out_fc  = nn.Linear(hidden_dim, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x_enc = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        else:
            raise ValueError("Input tensor must be 2D (batch, features)")

        h = self.proj(x_enc)
        h = self.norm0(h)
        h = self.act0(h)

        for block in self.blocks:
            h = block(h)

        return self.out_fc(h)