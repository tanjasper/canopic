import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models
import torch

class ConvBNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
		padding = (kernel_size - 1) // 2
		super(ConvBNReLU, self).__init__(
			nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU6(inplace=True)
		)
		
def _make_divisible(v, divisor, min_value=None):
	"""
	This function is taken from the original tf repo.
	It ensures that all layers have a channel number that is divisible by 8
	It can be seen here:
	https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
	:param v:
	:param divisor:
	:param min_value:
	:return:
	"""
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v
class InvertedResidualMobileNet(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidualMobileNet, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = int(round(inp * expand_ratio))
		self.use_res_connect = self.stride == 1 and inp == oup

		layers = []
		if expand_ratio != 1:
			# pw
			layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
		layers.extend([
			# dw
			ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
			# pw-linear
			nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
			nn.BatchNorm2d(oup),
		])
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


def channel_shuffle(x, groups):
	# type: (torch.Tensor, int) -> torch.Tensor
	batchsize, num_channels, height, width = x.data.size()
	channels_per_group = num_channels // groups

	# reshape
	x = x.view(batchsize, groups,
			   channels_per_group, height, width)

	x = torch.transpose(x, 1, 2).contiguous()

	# flatten
	x = x.view(batchsize, -1, height, width)

	return x


class InvertedResidualShuffleNet(nn.Module):
	def __init__(self, inp, oup, stride):
		super(InvertedResidualShuffleNet, self).__init__()

		if not (1 <= stride <= 3):
			raise ValueError('illegal stride value')
		self.stride = stride

		branch_features = oup // 2
		assert (self.stride != 1) or (inp == branch_features << 1)

		if self.stride > 1:
			self.branch1 = nn.Sequential(
				self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
				nn.BatchNorm2d(inp),
				nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(branch_features),
				nn.ReLU(inplace=True),
			)
		else:
			self.branch1 = nn.Sequential()

		self.branch2 = nn.Sequential(
			nn.Conv2d(inp if (self.stride > 1) else branch_features,
					  branch_features, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(branch_features),
			nn.ReLU(inplace=True),
			self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
			nn.BatchNorm2d(branch_features),
			nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(branch_features),
			nn.ReLU(inplace=True),
		)

	@staticmethod
	def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
		return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

	def forward(self, x):
		if self.stride == 1:
			x1, x2 = x.chunk(2, dim=1)
			out = torch.cat((x1, self.branch2(x2)), dim=1)
		else:
			out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

		out = channel_shuffle(out, 2)

		return out

class MobileNetV2_IN(nn.Module):
	def __init__(self, N=8,num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
		"""
		MobileNet V2 main class
		Args:
			num_classes (int): Number of classes
			width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
			inverted_residual_setting: Network structure
			round_nearest (int): Round the number of channels in each layer to be a multiple of this number
			Set to 1 to turn off rounding
		"""
		super(MobileNetV2_IN, self).__init__()
		block = InvertedResidualMobileNet
		input_channel = 32
		last_channel = 1280

		if inverted_residual_setting is None:
			inverted_residual_setting = [
				# t, c, n, s
				[1, 16, 1, 1],
				[6, 24, 2, 2],
				[6, 32, 3, 2],
				[6, 64, 4, 2],
				[6, 96, 3, 1],
				[6, 160, 3, 2],
				[6, 320, 1, 1],
			]

		# only check the first element, assuming user knows t,c,n,s are required
		if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
			raise ValueError("inverted_residual_setting should be non-empty "
							 "or a 4-element list, got {}".format(inverted_residual_setting))

		# building first layer
		input_channel = _make_divisible(input_channel * width_mult, round_nearest)
		self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
		features = [ConvBNReLU(N, input_channel, stride=2)]
		# building inverted residual blocks
		for t, c, n, s in inverted_residual_setting:
			output_channel = _make_divisible(c * width_mult, round_nearest)
			for i in range(n):
				stride = s if i == 0 else 1
				features.append(block(input_channel, output_channel, stride, expand_ratio=t))
				input_channel = output_channel
		# building last several layers
		features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
		# make it nn.Sequential
		self.features = nn.Sequential(*features)

		# building classifier
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(self.last_channel, num_classes),
		)
		self.bn0 = nn.InstanceNorm2d(N)

		# weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		x = self.bn0(x)
		x = self.features(x)
		x = x.mean([2, 3])
		x = self.classifier(x)
		return x


class MobileNetV2_BN(nn.Module):
	def __init__(self, N=8,num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
		"""
		MobileNet V2 main class
		Args:
			num_classes (int): Number of classes
			width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
			inverted_residual_setting: Network structure
			round_nearest (int): Round the number of channels in each layer to be a multiple of this number
			Set to 1 to turn off rounding
		"""
		super(MobileNetV2_BN, self).__init__()
		block = InvertedResidualMobileNet
		input_channel = 32
		last_channel = 1280

		if inverted_residual_setting is None:
			inverted_residual_setting = [
				# t, c, n, s
				[1, 16, 1, 1],
				[6, 24, 2, 2],
				[6, 32, 3, 2],
				[6, 64, 4, 2],
				[6, 96, 3, 1],
				[6, 160, 3, 2],
				[6, 320, 1, 1],
			]

		# only check the first element, assuming user knows t,c,n,s are required
		if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
			raise ValueError("inverted_residual_setting should be non-empty "
							 "or a 4-element list, got {}".format(inverted_residual_setting))

		# building first layer
		input_channel = _make_divisible(input_channel * width_mult, round_nearest)
		self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
		features = [ConvBNReLU(N, input_channel, stride=2)]
		# building inverted residual blocks
		for t, c, n, s in inverted_residual_setting:
			output_channel = _make_divisible(c * width_mult, round_nearest)
			for i in range(n):
				stride = s if i == 0 else 1
				features.append(block(input_channel, output_channel, stride, expand_ratio=t))
				input_channel = output_channel
		# building last several layers
		features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
		# make it nn.Sequential
		self.features = nn.Sequential(*features)

		# building classifier
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(self.last_channel, num_classes),
		)
		self.bn0 = nn.BatchNorm2d(N)

		# weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		x = self.bn0(x)
		x = self.features(x)
		x = x.mean([2, 3])
		x = self.classifier(x)
		return x


class ShuffleNetV2_BN(nn.Module):
	def __init__(self, stages_repeats, stages_out_channels, N=8, num_classes=101):
		super(ShuffleNetV2_BN, self).__init__()

		if len(stages_repeats) != 3:
			raise ValueError('expected stages_repeats as list of 3 positive ints')
		if len(stages_out_channels) != 5:
			raise ValueError('expected stages_out_channels as list of 5 positive ints')
		self._stage_out_channels = stages_out_channels

		input_channels = N
		output_channels = self._stage_out_channels[0]
		self.conv1 = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace=True),
		)
		input_channels = output_channels

		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
		for name, repeats, output_channels in zip(
				stage_names, stages_repeats, self._stage_out_channels[1:]):
			seq = [InvertedResidualShuffleNet(input_channels, output_channels, 2)]
			for i in range(repeats - 1):
				seq.append(InvertedResidualShuffleNet(output_channels, output_channels, 1))
			setattr(self, name, nn.Sequential(*seq))
			input_channels = output_channels

		output_channels = self._stage_out_channels[-1]
		self.conv5 = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace=True),
		)

		self.fc = nn.Linear(output_channels, num_classes)
		self.bn0 = nn.BatchNorm2d(N)
	def forward(self, x):
		x = self.bn0(x)
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.conv5(x)
		x = x.mean([2, 3])  # globalpool
		x = self.fc(x)
		return x


class ShuffleNetV2_IN(nn.Module):
	def __init__(self, stages_repeats, stages_out_channels, N=8, num_classes=101):
		super(ShuffleNetV2_IN, self).__init__()

		if len(stages_repeats) != 3:
			raise ValueError('expected stages_repeats as list of 3 positive ints')
		if len(stages_out_channels) != 5:
			raise ValueError('expected stages_out_channels as list of 5 positive ints')
		self._stage_out_channels = stages_out_channels

		input_channels = N
		output_channels = self._stage_out_channels[0]
		self.conv1 = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace=True),
		)
		input_channels = output_channels

		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
		for name, repeats, output_channels in zip(
				stage_names, stages_repeats, self._stage_out_channels[1:]):
			seq = [InvertedResidualShuffleNet(input_channels, output_channels, 2)]
			for i in range(repeats - 1):
				seq.append(InvertedResidualShuffleNet(output_channels, output_channels, 1))
			setattr(self, name, nn.Sequential(*seq))
			input_channels = output_channels

		output_channels = self._stage_out_channels[-1]
		self.conv5 = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace=True),
		)

		self.fc = nn.Linear(output_channels, num_classes)
		self.bn0 = nn.InstanceNorm2d(N)
	def forward(self, x):
		x = self.bn0(x)
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.conv5(x)
		x = x.mean([2, 3])  # globalpool
		x = self.fc(x)
		return x



def mobilenetv2_bn(N=3,num_classes=100,width_mult=0.75):
	model = MobileNetV2_BN(num_classes=num_classes,N=N,width_mult=0.75)
	return model


def mobilenetv2_in(N=3,num_classes=100,width_mult=0.75):
	model = MobileNetV2_IN(num_classes=num_classes,N=N,width_mult=0.75)
	return model

def shufflenetv2_bn(N=3,num_classes=100):
	model = ShuffleNetV2_BN([4, 8, 4], [24, 48, 96, 192, 1024],num_classes=num_classes,N=N)
	return model

def shufflenetv2_in(N=3,num_classes=100):
	model = ShuffleNetV2_IN([4, 8, 4], [24, 48, 96, 192, 1024],num_classes=num_classes,N=N)
	return model

