# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from models.multi_net import EncoderDiscriminatorNet
from settings.util import load_encoder_settings
import torch
import os


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg, opt):

    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]

    if opt.encoder_checkpoint_path:  # load encoder from provided checkpoint
        model = EncoderDiscriminatorNet(opt, None, None, None)
        model.encoder_from_checkpoint(torch.load(os.path.join(opt.root_data_dir, opt.encoder_checkpoint_path)))
        if opt.encoder_settings_json:  # add encoders if user gave more
            encoder_archs, encoder_settings = load_encoder_settings(opt.encoder_settings_json)
            model.add_encoders(opt, encoder_archs, encoder_settings)
    else:  # create new encoder
        encoder_archs, encoder_settings = load_encoder_settings(opt.encoder_settings_json)
        model = EncoderDiscriminatorNet(opt, None, None, encoder_archs,
                                        encoder_settings=encoder_settings)
    model.freeze_generator()
    model.discriminatorB = meta_arch(cfg)
    model.faster_rcnn = True

    return model
