import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import models
from models import register


# def init_wb(shape, is_weight=True):
#     if is_weight:
#         # if shape == torch.Size([1, 1]):
#         #     # Change the shape of the weight to (2, shape[1])
#         #     weight = torch.empty(shape[1], shape[0])
#         # else:
#         #     weight = torch.empty(shape[1], shape[0] - 1)
#         weight = torch.empty(shape[1], shape[0])
#         nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
#         return weight.t().detach()
#     else:
#         val = shape[0]
#         bias = torch.empty(val, 1)
#         nn.init.kaiming_uniform_(bias, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(bias)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 1e-5
#         nn.init.uniform_(bias, -bound, bound)
#         return bias.detach()

def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()



@register('trans_inr')
class TransInr(nn.Module):

    def __init__(self, tokenizer, hyponet, n_groups, transformer_encoder):
        super().__init__()
        dim = transformer_encoder['args']['dim']
        self.tokenizer = models.make(tokenizer, args={'dim': dim})
        self.hyponet = models.make(hyponet)
        self.transformer_encoder = models.make(transformer_encoder)
        
        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        self.user_inp = False #True: hypo_mlp; False: hypo_Shacira
        for name, shape in self.hyponet.param_shapes.items():
            if self.user_inp:
                self.base_params[name] = nn.Parameter(init_wb(shape))
            else:
                self.base_params[name] = nn.Parameter(self.hyponet.get_codebook())
            # print("Before shape=", shape)
            # if shape != torch.Size([47737, 1]):
            #     if len(shape) == 1:   
            #         self.base_params[name] = nn.Parameter(init_wb(shape, is_weight=False))
            #     else:
            #         self.base_params[name] = nn.Parameter(init_wb(shape, is_weight=True))
            # # print(self.base_params[name].shape)
            # else:
            #     self.base_params[name] = nn.Parameter(self.hyponet.get_codebook())
                # print(self.base_params[name])
            # shape = self.base_params[name].shape
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            # if shape == torch.Size([1,1]):
            #     self.wtoken_postfc[name] = nn.Sequential(
            #         nn.LayerNorm(dim),
            #         nn.Linear(dim, shape[0]),
            #     )
            # else:
            #     self.wtoken_postfc[name] = nn.Sequential(
            #         nn.LayerNorm(dim),
            #         nn.Linear(dim, shape[0] - 1),
            #     )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))

    def forward(self, data):
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1))
        trans_out = trans_out[:, -len(self.wtokens):, :]
        
        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            # print("wb shape=", wb.shape)
            w, b = wb[:, :-1, :], wb[:, -1:, :]
            # print("w shape=", w.shape)
            # print("b shape=", b.shape)

            l, r = self.wtoken_rng[name]
            # print("l=", l)
            # print("r=", r)
            x = self.wtoken_postfc[name](trans_out[:, l: r, :])
            # print("x shape=", x.shape)
            x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            # print("x shape=", x.shape)        
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)
            # print("w shape=", w.shape)
            wb = torch.cat([w, b], dim=1)
            # print("wb shape=", wb.shape)
            if wb.shape == torch.Size([B, 3, 1]) or wb.shape == torch.Size([B, 16, 1]):
                if wb.shape == torch.Size([B, 3, 1]):
                    wb = wb.reshape(B, 3)
                elif wb.shape == torch.Size([B, 16, 1]):
                    wb = wb.reshape(B, 16)
                # print("wb shape=", wb.shape)
            params[name] = wb
            # print("params[name] shape=", params[name].shape)

        self.hyponet.set_params(params)
        return self.hyponet
