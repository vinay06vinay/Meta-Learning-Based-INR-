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
    conf_latent_decoder = {'ldecode_enabled': True, 'ldecode_type': 'single', 'use_sga': True, 'diff_sampling': True, 'ldecode_matrix': 'sq', 'latent_dim': 1, 'norm': 'max', 'norm_every': 10, 'use_shift': True, 'num_layers_dec': 0, 'hidden_dim_dec': 0, 'activation': 'none', 'final_activation': 'none', 'clamp_weights': 0.0, 'ldec_std': 0.1, 'num_decoders': 2, 'temperature': 0.1, 'decay_period': 0.9, 'alpha_std': 10.0}
    conf_entropy_reg = {'num_prob_layers': 2, 'entropy_reg': 0.001, 'entropy_reg_end': 0.0001, 'entropy_reg_sched': 'cosine', 'noise_freq': 1}
    # "geometric" - determines the resolution of the grid using geometric sequence initialization from InstantNGP,

    grid = LatentGrid.from_geometric(
        feature_dim=1,
        latent_dim=conf_latent_decoder["latent_dim"],
        num_lods=16,
        multiscale_type='cat',
        resolution_dim=2,
        feature_std=0.1,
        feature_bias=0.0,
        codebook_bitwidth=12,
        min_grid_res=16,
        max_grid_res=512,
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
        for i, (name, param) in enumerate(self.pipeline.named_parameters()):
            self.param_shapes[f'wb{i}'] = param.shape
        self.params = None
    def set_params(self, params):
        self.params = params
    def forward(self, x):
        for i, (name, param) in enumerate(self.pipeline.named_parameters()):
            param.data = self.params[f'wb{i}'].data
        return self.pipeline(x)             
