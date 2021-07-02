import torch
import torch.nn as nn
import models.discriminators as discriminators
import models.encoders as generators
import torch.nn.functional as F
from fns_model import parser_to_encoder_settings
import sys
sys.path.append('maskrcnn_benchmark_master')
from maskrcnn_benchmark.structures.image_list import to_image_list, ImageList


class EncoderDiscriminatorNet(nn.Module):
	"""A class that can take in an arbitrary number of encoders in series and 2 discriminators A and B

	The nn.Sequential class is used for the encoders as it allows a sequence of Pytorch modules.
	It is chosen over the nn.ModuleList since nn.Sequential comes with the forward operation
	"""
	def __init__(self, opt, discriminatorA_arch, discriminatorB_arch, encoder_archs, discriminatorA_settings=None,
				 discriminatorB_settings=None, encoder_settings=None, faster_rcnn=False):

		super(EncoderDiscriminatorNet, self).__init__()
		self.discriminatorA_arch = discriminatorA_arch
		self.discriminatorB_arch = discriminatorB_arch
		self.encoder_archs = encoder_archs
		self.discriminatorA_settings = discriminatorA_settings if discriminatorA_settings else {}
		self.discriminatorB_settings = discriminatorB_settings if discriminatorB_settings else {}
		self.encoder_settings = encoder_settings if encoder_settings else []
		self.faster_rcnn = faster_rcnn
		self.faster_rcnn_resize = 0

		# initialize encoders
		if encoder_archs:
			if type(encoder_archs) is not list:
				encoder_archs = [encoder_archs]
			# Set settings to empty dictionaries if not given
			if encoder_settings is None:
				encoder_settings = [{}] * len(encoder_archs)  # TODO: might need to pass in None for opt
			encoder_modules = []
			for i, arch in enumerate(encoder_archs):
				encoder_modules.append(generators.__dict__[arch](opt, **encoder_settings[i]))
			self.encoder = nn.Sequential(*encoder_modules)
			self.proj = self.encoder  # for compatibility with older code. TODO: deprecate this
		else:
			self.encoder = generators.PassThrough()

		# initialize discriminators
		if discriminatorA_arch:
			self.discriminatorA = discriminators.__dict__[discriminatorA_arch](**discriminatorA_settings)
		else:
			self.discriminatorA = None
		if discriminatorB_arch:
			self.discriminatorB = discriminators.__dict__[discriminatorB_arch](**discriminatorB_settings)
		else:
			self.discriminatorB = None

	def encoder_modules(self):
		"""Returns encoder modules in a list for easy access to each module"""
		return list(self.encoder.modules())[0]

	def forward(self, x, target=None):
		"""Forward operation for MultiNet

		Default is to do encoder --> discriminatorA
		This also allows faster-rcnn forwarding which is encoder --> FasterRCNNforward"""

		if self.faster_rcnn:
			images = to_image_list(x)
			images.tensors = self.encoder(images.tensors)
			if self.faster_rcnn_resize:
				im = x.tensors
				if min(im.shape[2], im.shape[3]) < self.faster_rcnn_resize:
					ratio = self.faster_rcnn_resize / min(im.shape[2], im.shape[3])
					x.tensors = F.interpolate(x.tensors, size=(round(ratio*im.shape[2]), round(ratio*im.shape[3])))
					for idx, im_size in enumerate(x.image_sizes):
						x.image_sizes[idx] = torch.Size([round(im_size[0] * ratio), round(im_size[1] * ratio)])
			return self.discriminatorB(images, target)
		else:
			x = self.encoder(x)
			if self.discriminatorA:
				x = self.discriminatorA(x)
			return x

	def freeze_encoder(self):
		for encoder_module in self.encoder_modules():
			encoder_module.freeze()

	def unfreeze_encoder(self):
		for encoder_module in self.encoder_modules():
			encoder_module.unfreeze()

	def freeze_discriminatorA(self):
		for param in self.discriminatorA.parameters():
			param.requires_grad = False

	def freeze_discriminatorB(self):
		for param in self.discriminatorB.parameters():
			param.requires_grad = False

	def unfreeze_discriminatorA(self):
		for param in self.discriminatorA.parameters():
			param.requires_grad = True

	def unfreeze_discriminatorB(self):
		for param in self.discriminatorB.parameters():
			param.requires_grad = True

	def freeze_generator(self):
		"""For cross-compatibility with older versions. Deprecate this soon."""
		self.freeze_encoder()

	def unfreeze_generator(self):
		"""For cross-compatibility with older versions. Deprecate this soon."""
		self.unfreeze_encoder()

	def save_debug_data(self, save_dir, savename_suffix=''):
		"""Save any data to be used for debugging by calling save_debug_data of each encoder"""
		for encoder_module in self.encoder_modules():
			encoder_module.save_debug_data(save_dir, savename_suffix=savename_suffix)

	def encoder_from_checkpoint(self, checkpoint):
		"""Replaces current encoders with that provided in the checkpoint

		It also updates self.encoder_archs and self.encoder_settings"""
		if 'opt' in checkpoint and 'generator_arch' in checkpoint['opt']:  # old version
			encoder_arch = checkpoint['opt'].generator_arch
			encoder_settings = parser_to_encoder_settings(checkpoint['opt'])
			self.encoder = nn.Sequential(generators.__dict__[encoder_arch](checkpoint['opt'], **encoder_settings,
																		   skip_init_loading=True))
			self.encoder_modules()[0].load_state_dict(checkpoint['generator_state_dict'])
			self.encoder_archs = [encoder_arch]
			self.encoder_settings = [encoder_settings]
		else:  # new version
			if 'opt' in checkpoint:
				opt = checkpoint['opt']
			else:
				opt = None
			encoder_modules = []
			self.encoder_archs = []
			self.encoder_settings = []
			for i, arch in enumerate(checkpoint['encoder_archs']):
				encoder_modules.append(generators.__dict__[arch](opt, **checkpoint['encoder_settings'][i],
																 skip_init_loading=True))
				self.encoder_archs.append(arch)
				self.encoder_settings.append(checkpoint['encoder_settings'][i])
			self.encoder = nn.Sequential(*encoder_modules)
			self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
		self.proj = self.encoder

	def checkpoint_info(self):
		"""Returns dictionary of data used for saving checkpoints"""
		return {'discriminatorA_arch': self.discriminatorA_arch,
				'discriminatorB_arch': self.discriminatorB_arch,
				'encoder_archs': self.encoder_archs,
				'discriminatorA_settings': self.discriminatorA_settings,
				'discriminatorB_settings': self.discriminatorB_settings,
				'encoder_settings': self.encoder_settings,
				'model_state_dict': self.state_dict(),
				'encoder_state_dict': self.encoder.state_dict()}

	def train_generator_update_encoder(self, epoch):
		for encoder in self.encoder_modules():
			encoder.train_generator_update(epoch)

	def add_encoders(self, opt, encoder_archs, encoder_settings):
		encoder_modules = []
		for module in self.encoder_modules():
			encoder_modules.append(module)
		for arch, setting in zip(encoder_archs, encoder_settings):
			self.encoder_archs.append(arch)
			self.encoder_settings.append(setting)
			encoder_modules.append(generators.__dict__[arch](opt, **setting))
		self.encoder = nn.Sequential(*encoder_modules)
		self.proj = self.encoder  # for compatibility with older code. TODO: deprecate this


def model_from_checkpoint(checkpoint_path):
	"""Given the path to a saved Torch file, return the appropriate EncoderDiscriminatorNet

	The Torch checkpoint must contain the following:
		- model_state_dict (state_dict): The state_dict of the entire EncoderDiscriminatorNet
		- encoder_archs (list): list of encoder architectures
		- encoder_settings (list): list of encoder settings
		- discriminatorA_arch (str): architecture of discriminator A
		- discriminatorA_settings (dict): settings for discriminator A
		- discriminatorB_arch (str): architecture of discriminator B
		- discriminatorB_settings (dict): settings of discriminator B
	If encoder_archs is missing, the encoder will be a PassThrough
	If discriminatorA(B)_arch is missing, discriminatorA(B) will be None
	"""
	print('Loading checkpoint from ' + checkpoint_path)
	checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(0))
	# Cross-compatibility with old checkpoints
	if 'generator_state_dict' in checkpoint:
		discriminatorA_arch = checkpoint['opt'].face_recog_arch
		discriminatorB_arch = checkpoint['opt'].face_noface_arch
		encoder_arch = checkpoint['opt'].generator_arch
		encoder_settings = parser_to_encoder_settings(checkpoint['opt'])
		encoder_settings['skip_init_loading'] = True
		model = EncoderDiscriminatorNet(checkpoint['opt'], discriminatorA_arch, discriminatorB_arch, encoder_arch,
										encoder_settings=encoder_settings)
		model.encoder_modules()[0].load_state_dict(checkpoint['generator_state_dict'])
		# remove all generator state dict info, leaving only DiscriminatorA and DiscriminatorB
		# this allows load_state_dict to actually work since then the keys will be matched
		for k in checkpoint['model_state_dict'].keys():
			del checkpoint['model_state_dict'][k]
		model.load_state_dict(checkpoint['model_state_dict'])
	else:
		discriminatorA_arch = checkpoint['discriminatorA_arch']
		discriminatorB_arch = checkpoint['discriminatorB_arch']
		encoder_arch = checkpoint['encoder_archs']
		discriminatorA_settings = checkpoint['discriminatorA_settings']
		discriminatorB_settings = checkpoint['discriminatorB_settings']
		encoder_settings = checkpoint['encoder_settings']
		model = EncoderDiscriminatorNet(checkpoint['opt'], discriminatorA_arch, discriminatorB_arch, encoder_arch,
										discriminatorA_settings=discriminatorA_settings,
										discriminatorB_settings=discriminatorB_settings,
										encoder_settings=encoder_settings)
		model.load_state_dict(checkpoint['model_state_dict'])
	return model


# general network with a projection layer
# Note: intermediate has not been tested yet... might not be working
# In meantime, discriminator_params do not include anything
# generator_params may include (with defaults listed):
#   proj_type=0
#   proj_mat=None
#   noise_std=0
#   learn_noise=False
class ProjNet(nn.Module):

	def __init__(self, discriminator_arch, projection_arch,
				 discriminator_params={}, projection_params={},
				 num_classes=10, intermediate=None, intermediate_params=None):

		self.has_intermediate = False

		super(ProjNet, self).__init__()
		self.proj = generators.__dict__[projection_arch](**projection_params)
		if intermediate is not None:
			self.intermediate = discriminators.__dict__[intermediate](**intermediate_params)
			self.has_intermediate = True
		self.discriminator = discriminators.__dict__[discriminator_arch](num_classes=num_classes, **discriminator_params)

	def forward(self, x):
		x = self.proj(x)
		if self.has_intermediate:
			x = self.intermediate(x)
		x = self.discriminator(x)
		return x

	def freeze_generator(self):
		self.proj.freeze()

	def unfreeze_generator(self):
		self.proj.unfreeze()

	def freeze_discriminator(self):
		self.discriminator.freeze()

	def unfreeze_discriminator(self):
		self.discriminator.unfreeze()


class ProjNet_multiop(nn.Module):

	def __init__(self):
		raise NameError('ProjNet_multiop is deprecated. Please use ProjNet')


# general network with a projection layer and two discriminators
# Note: intermediate has not been tested yet... might not be working
# In meantime, discriminator_params do not include anything
# generator_params may include (with defaults listed):
#   proj_type=0
#   proj_mat=None
#   noise_std=0
#   learn_noise=False
class ProjNet2Discriminators(nn.Module):

	def __init__(self, discriminatorA_arch, discriminatorB_arch, projection_arch,
				 discriminatorA_params={}, discriminatorB_params={}, projection_params={},
				 intermediate=None, intermediate_params=None, num_face_classes=100):
		self.has_intermediate = False
		super(ProjNet2Discriminators, self).__init__()
		self.proj = generators.__dict__[projection_arch](**projection_params)
		if intermediate is not None:
			self.intermediate = discriminators.__dict__[intermediate](**intermediate_params)
			self.has_intermediate = True
		self.discriminatorA = discriminators.__dict__[discriminatorA_arch](num_classes=num_face_classes, **discriminatorA_params)  # hacker will recognize faces
		self.discriminatorB = discriminators.__dict__[discriminatorB_arch](num_classes=2, **discriminatorB_params) # useful task is face vs. no-face

	def forward(self, x):
		x = self.proj(x)
		if self.has_intermediate:
			x = self.intermediate(x)
		x = self.discriminatorA(x)
		return x

	def freeze_discriminatorA(self):
		for param in self.discriminatorA.parameters():
			param.requires_grad = False

	def freeze_discriminatorB(self):
		for param in self.discriminatorB.parameters():
			param.requires_grad = False

	def freeze_generator(self):
		for param in self.proj.parameters():
			param.requires_grad = False

	def unfreeze_discriminatorA(self):
		for param in self.discriminatorA.parameters():
			param.requires_grad = True

	def unfreeze_discriminatorB(self):
		for param in self.discriminatorB.parameters():
			param.requires_grad = True

	def unfreeze_generator(self):
		for param in self.proj.parameters():
			param.requires_grad = True
		if type(self.proj).__name__ == 'LinearWithForcedTranspose':
			if not self.proj.learn_noise:  # i = 1 is the noise parameter for LinearWithForcedTranspose
				self.proj.proj.noise.requires_grad = False
		if type(self.proj).__name__ == 'LinearWithFlexibleTranspose':
			if not self.proj.learn_noise:
				self.proj.noise.requires_grad = False
			if self.proj.proj_type == 1:  # do not learn transpose
				self.proj.projT.weight.requires_grad = False

	def unfreeze_transpose(self):
		self.proj.projT.weight.requires_grad = True


## Deprecated


class ProjNet2Discriminators_multiop(nn.Module):

	def __init__(self, discriminatorA_arch, discriminatorB_arch, projection_arch,
				 discriminatorA_params={}, discriminatorB_params={}, projection_params={},
				 intermediate=None, intermediate_params=None, num_face_classes=100):
		raise NameError('ProjNet2Discriminators_multiop is deprecated. Use ProjNet2Discriminators (exactly equal')

