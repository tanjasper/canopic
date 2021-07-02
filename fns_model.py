import numpy as np
import os

linear_arches = ['LinearWithForcedTranspose', 'LinearWithFixedFlexibleTranspose', 'LinearWithLearnedTranspose',
                 'LinearOnly']
conv_arches = ['ConvGenerator']
new_arches = ['GaborConv2d','OrientedGradients','OrientedGradientsNew', 'OrientedGradientsCCMax']


def setup_discriminators(opt):
    """
    TODO:
        Figure out how to get transpose for linear encoders:
            Since this has to do with interaction between encoder and discriminators, it makes most sense to
            have EncoderDiscriminatorNet worry about this
    """
    recog_params = {}
    detect_params = {}
    if opt.generator_arch == 'LinearOnly':
        raise Exception('setup_discriminators -- transpose not setup for LinearOnly encoders')
        recog_arch = 'BackProjectNetwork'
        recog_params['discriminator_arch'] = opt.face_recog_arch
        # recog_params['proj_mat'] = np.transpose(proj_mat)
        detect_arch = 'BackProjectNetwork'
        detect_params['discriminator_arch'] = opt.face_noface_arch
        # detect_params['proj_mat'] = np.transpose(proj_mat)
    else:
        recog_arch = opt.face_recog_arch
        detect_arch = opt.face_noface_arch
    return recog_arch, recog_params, detect_arch, detect_params


def parser_to_encoder_settings(opt):
    """Converts old-format parser arguments into new-format parameter dictionary

    Previously, parameters for encoders are passed in via parser through the command line.
    Now, we want encoder parameters to be in dictionaries called user_parameters.
    This function takes the old-format parser and outputs the new-format dictionary.
    """
    if opt.generator_arch in linear_arches:
        encoder_settings = {'noise_std': opt.noise_std, 'learn_noise': opt.learn_noise}
        if opt.proj_mat_init_path:
            encoder_settings['proj_mat_path'] = os.path.join(opt.root_data_dir, opt.proj_mat_init_path)
        # TODO: Need to have quantize_bits and random_backproject for LinearWithFixedFlexibleTranspose
        # TODO: Need to have with_transpose for LinearWithForcedTranspose
    if opt.generator_arch in conv_arches:
        encoder_settings = {'red_dim': opt.red_dim, 'mxp': opt.mxp, 'avg': opt.avg, 'learn_noise': opt.learn_noise,
                              'quantize_bits': opt.quantize_bits, 'quantize': opt.quantize, 'mode': opt.mode}
        if opt.proj_mat_init_path:
            encoder_settings['kernel_path'] = opt.proj_mat_init_path
    if opt.generator_arch in ['ConvWithForcedTranspose', 'ConvWithFixedFlexibleTranspose', 'ConvWithLearnedTranspose']:
        encoder_settings = {'noise_std': opt.noise_std, 'learn_noise': opt.learn_noise}
        if opt.proj_mat_init_path:
            encoder_settings['kernel_path'] = opt.proj_mat_init_path
    if opt.generator_arch in new_arches:
        encoder_settings = {'kernel_size': opt.gabor_kernel, 'mxp': opt.mxp, 'learn_noise': opt.learn_noise,
                              'quantize_bits': opt.quantize_bits, 'quantize': opt.quantize}
    if opt.generator_arch == 'Perm':
        encoder_settings = {'red_dim': opt.red_dim}
        if opt.proj_mat_init_path:
            encoder_settings['perm_mat_path'] = opt.proj_mat_init_path
    if opt.generator_arch == 'learnQuantize':
        encoder_paramseters = {'bits': opt.quantize_bits}
        if opt.proj_mat_init_path:
            encoder_settings['kernel_path'] = opt.proj_mat_init_path
    if opt.generator_arch == 'ConvPermQuantize':
        encoder_settings = {'quantize_bits': opt.quantize_bits, 'red_dim': opt.red_dim}
        if opt.proj_mat_init_path:
            encoder_settings['kernel_path'] = opt.proj_mat_init_path
    if opt.generator_arch == 'ConvMxPQuantize':
        encoder_settings = {'quantize_bits': opt.quantize_bits, 'red_dim': opt.red_dim, 'sig_scale': opt.sig_scale}
        if opt.proj_mat_init_path:
            encoder_settings['kernel_path'] = opt.proj_mat_init_path
    if opt.generator_arch == 'PosConvMxPQuantize':
        encoder_settings = {'quantize_bits': opt.quantize_bits, 'red_dim': opt.red_dim, 'sig_scale': opt.sig_scale}
        if opt.proj_mat_init_path:
            encoder_settings['kernel_path'] = opt.proj_mat_init_path
    if opt.generator_arch == 'ConvPerm':
        encoder_settings = {'red_dim': opt.red_dim}
        if opt.proj_mat_init_path:
            encoder_settings['kernel_path'] = opt.proj_mat_init_path
    return encoder_settings
