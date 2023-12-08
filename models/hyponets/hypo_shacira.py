import os
import argparse
import logging
import numpy as np
import wandb
import torch
import wisp
import json
from wisp.app_utils import default_log_setup, args_to_log_format, copy_dir_msrsync, copy_dir, copy_file
import wisp.config_parser as config_parser
from wisp.framework import WispState, ImageState
from wisp.models.grids import BLASGrid, LatentGrid
from wisp.models.latent_decoders.basic_latent_decoder import DecoderIdentity
from wisp.models.latent_decoders.multi_latent_decoder import MultiLatentDecoder
from wisp.tracers import BaseTracer, PackedRFTracer
from wisp.models.nefs import BaseNeuralField, NeuralImage
from wisp.models.pipeline import Pipeline
from models import register


def load_grid() -> BLASGrid:
    """ Wisp's implementation of NeRF uses feature grids to improve the performance and quality (allowing therefore,
    interactivity).
    This function loads the feature grid to use within the neural pipeline.
    Grid choices are interesting to explore, so we leave the exact backbone type configurable,
    and show how grid instances may be explicitly constructed.
    Grids choices, for example, are: OctreeGrid, TriplanarGrid, HashGrid,LatentGrid, CodebookOctreeGrid
    See corresponding grid constructors for each of their arg details.
    """
    grid = None
    # Optimization: For octrees based grids, if dataset contains depth info, initialize only cells known to be occupied
    # WARNING: Not implemented/tested for grids other than HashGrid/LatentGrid 
    # conf_latent_decoder = {'ldecode_enabled': True, 'ldecode_type': 'single', 'use_sga': True, 'diff_sampling': True, 'ldecode_matrix': 'sq', 'latent_dim': 1, 'norm': 'max', 'norm_every': 10, 'use_shift': True, 'num_layers_dec': 0, 'hidden_dim_dec': 0, 'activation': 'none', 'final_activation': 'none', 'clamp_weights': 0.0, 'ldec_std': 0.1, 'num_decoders': 2, 'temperature': 0.1, 'decay_period': 0.9, 'alpha_std': 10.0}
    # conf_entropy_reg = {'num_prob_layers': 2, 'entropy_reg': 0.001, 'entropy_reg_end': 0.0001, 'entropy_reg_sched': 'cosine', 'noise_freq': 1}
    conf_latent_decoder = {'ldecode_enabled': False, 'ldecode_type': 'single', 'use_sga': True, 'diff_sampling': True, 'ldecode_matrix': 'sq', 'latent_dim': 1, 'norm': 'max', 'norm_every': 10, 'use_shift': True, 'num_layers_dec': 0, 'hidden_dim_dec': 0, 'activation': 'none', 'final_activation': 'none', 'clamp_weights': 0.0, 'ldec_std': 0.1, 'num_decoders': 2, 'temperature': 0.1, 'decay_period': 0.9, 'alpha_std': 10.0}
    conf_entropy_reg = {'num_prob_layers': 2, 'entropy_reg': 0.0, 'entropy_reg_end': 0.0, 'entropy_reg_sched': 'cosine', 'noise_freq': 1}
    # "geometric" - determines the resolution of the grid using geometric sequence initialization from InstantNGP,

    grid = LatentGrid.from_geometric(
        feature_dim=1,
        latent_dim=conf_latent_decoder["latent_dim"],
        num_lods=44,
        multiscale_type='cat',
        resolution_dim=2,
        feature_std=0.1,
        feature_bias=0.0,
        codebook_bitwidth=10,
        min_grid_res=16,
        max_grid_res=128,
        blas_level=7,
        init_grid='uniform',
        conf_latent_decoder=conf_latent_decoder,
        conf_entropy_reg=conf_entropy_reg
        )
    return grid


def load_neural_field() -> NeuralImage:
    """ Creates a "Neural Field" instance which converts input coordinates to some output signal.
    Here a NeuralRadianceField is created, which maps 2D coordinates -> RGB.
    The NeuralRadianceField uses spatial feature grids internally for faster feature interpolation and raymarching.
    """
    grid = load_grid()
    nef = NeuralImage(
        grid=grid,
        pos_embedder='none',
        position_input=False,
        pos_multires=10,
        activation_type='relu',
        layer_type='none',
        hidden_dim=16,
        num_layers=1,
    )
    return nef

def load_neural_pipeline(device) -> Pipeline:
    """ In Wisp, a Pipeline comprises of a neural field + a tracer (the latter is optional in this image case).
    Together, they form the complete pipeline required to render a neural primitive from input rays / coordinates.
    """
    nef = load_neural_field()
    pipeline = Pipeline(nef=nef, tracer=None)
    # if args.pretrained:
    #     if args.model_format == "full":
    #         pipeline = torch.load(args.pretrained)
    #     else:
    #         pipeline.load_state_dict(torch.load(args.pretrained))
    pipeline.to(device)
    return pipeline

#Write a class from the above functions note that the forward function is already there in the pipeline.py of the wisp library folder
@register('hypo_shacira')
class HypoShacira(torch.nn.Module):
    def __init__(self):
        super(HypoShacira, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = load_neural_pipeline(self.device)
        self.param_shapes = dict()
        i = 0
        # self.layers_dict = dict()
        for (name, param) in self.pipeline.named_parameters():
            # if param.shape == torch.Size([1, 1]) and (name != "nefgridlatent_declayers0scale" or name != "nefgridlatent_declayers0shift"):
            #     continue
            # if name == "nefgridlatent_decdiv":
            #     self.param_shapes[f'nefgridlatent_decdiv'] = param.shape
            #     print(f'nefgridlatent_decdiv', name, param.shape)
            # if name == "nefgridlatent_declayers0scale":
            #     self.param_shapes[f'nefgridlatent_declayers0scale'] = param.shape
            #     print(f'nefgridlatent_declayers0scale', name, param.shape)
            # if name == "nefgridlatent_declayers0shift":
            #     self.param_shapes[f'nefgridlatent_declayers0shift'] = param.shape
            #     print(f'nefgridlatent_declayers0shift', name, param.shape)
            print("Name: ", name, "Shape: ", param.shape)
            # if param.shape == torch.Size([47737, 1]):
            if param.shape == torch.Size([38300, 1]):
                self.param_shapes[f'wb{i}'] = param.shape
                # self.layers_dict[f'wb{i}'] = [name, param, param.shape]
                print(f'Selected for prediction wb{i}', name, param.shape)
            elif param.shape == torch.Size([16, 16]):
                # shape = param.shape
                self.param_shapes[f'wb{i}'] = param.shape
                # self.layers_dict[f'wb{i}'] = [name, param, param.shape]
                print(f'Selected for prediction wb{i}', name, param.shape)
            elif param.shape == torch.Size([3,16]):
                # shape = param.shape
                self.param_shapes[f'wb{i}'] = param.shape
                # self.layers_dict[f'wb{i}'] = [name, param, param.shape]
                print(f'Selected for prediction wb{i}', name, param.shape)
            elif param.shape == torch.Size([16,44]):
                # shape = param.shape
                self.param_shapes[f'wb{i}'] = param.shape
                # self.layers_dict[f'wb{i}'] = [name, param, param.shape]
                print(f'Selected for prediction wb{i}', name, param.shape)
            # if i == 0:
                # print("Inside the constructor hyposhacira: ", param.data)
            i+=1
        self.params = None
    def set_params(self, params):
        self.params = params
    def get_codebook(self):
        return self.pipeline.nef.grid.codebook
    # def get_layers_dict(self):
    #     return self.layers_dict
    def forward(self, x):
        # print("Xshape= ", x.shape)
        btch = x.shape[0]
        # print(btch)
        op = torch.empty(btch, 178**2, 3, device=self.device)
        for j in range(btch):
            i = 0
            for (name, param) in self.pipeline.named_parameters():
                # if param.shape == torch.Size([1, 1]) and ((name != "nefgridlatent_declayers0scale") or (name != "nefgridlatent_declayers0shift")):
                #     continue      
                # if param.shape == torch.Size([47737, 1]) or param.shape == torch.Size([16, 16]) or param.shape == torch.Size([3,16]): 
                if param.shape == torch.Size([38300, 1]) or param.shape == torch.Size([16, 16]) or param.shape == torch.Size([3,16]) or param.shape == torch.Size([16,44]):               
                    param.data = self.params[f'wb{i}'][j, ...].data
                # if name == "nefgridlatent_decdiv":
                #     param.data = self.params[f'nefgridlatent_decdiv'][j, ...].data
                # if name == "nefgridlatent_declayers0scale":
                #     param.data = self.params[f'nef.grid.latent_dec.layers0scale'][j, ...].data
                # if name == "nefgridlatent_declayers0shift":
                #     param.data = self.params[f'nefgridlatent_declayers0shift'][j, ...].data
                i+=1
                if i == 0 and j == 0:
                    print(param.data)
            # print(self.params[f'wb{0}'][0, ...].data)
            xin =  x[j,:,:].to(self.device)
            xin  =  xin.view(-1, 2)
            # print(xin.shape)
            # with torch.no_grad():
            op[j, ...] = torch.stack(self.pipeline(coords=xin, channels=["rgb"]), dim=0).to(self.device)

        # Reshape the output tensor to (batch_size, 178, 178, 3)
        oup = op.view(btch, 178, 178, 3)
        # print(oup.shape)        
        return oup             
