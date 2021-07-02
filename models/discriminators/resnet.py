"""Networks based on ResNet (modified for 64x64 images)"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models
import torch
from models.discriminators.base_discriminator import BaseDiscriminator


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out



class BasicBlock2(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock2, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ELU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out




class BasicBlock3(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock3, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet64(BaseDiscriminator):

	def __init__(self, block, layers, num_classes=10, zero_init_residual=False, in_channels=3, init_kernel=3, init_stride=1):
		super(ResNet64, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)
		self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


class ResNet64_BN(BaseDiscriminator):

	def __init__(self, block, layers, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
		super(ResNet64_BN, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)
		self.bn0 = nn.BatchNorm2d(3)
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x



class ResNet64_1(BaseDiscriminator):

	def __init__(self, block, layers, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
		super(ResNet64_1, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)
		self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


class ResNet64_96(BaseDiscriminator):

	def __init__(self, block, layers, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
		super(ResNet64_96, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)
		#self.bn0 = nn.BatchNorm2d(96)
		self.conv1 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ELU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.dr = nn.Dropout(p=0.5)
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)



	def forward(self, x):
		#x = self.bn0(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		#x = self.dr(x)
		return x


class ResNet64_192(BaseDiscriminator):

	def __init__(self, block, layers, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
		super(ResNet64_192, self).__init__()
		self.inplanes = 256
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)
		#self.bn0 = nn.BatchNorm2d(192)
		self.conv1 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(256)
		self.relu = nn.ELU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		self.layer1 = self._make_layer(block, 256, layers[0])
		self.layer2 = self._make_layer(block, 256, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.dr = nn.Dropout(p=0.5)
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)



	def forward(self, x):
		# x = self.bn0(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		# x = self.dr(x)
		return x


class FC(nn.Module):
	def __init__(self,inp_dim,num_classes=100):
		super(FC, self).__init__()
		self.fc = nn.Linear(512 * inp_dim, num_classes)

	def forward(self, x):
		x = self.fc(x)
		return x




# class ResNet64_24(nn.Module):
#
#     def __init__(self, block, layers, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
#         super(ResNet64_24, self).__init__()
#         self.inplanes = 64
#         # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#         #                       bias=False)
#         self.conv1 = nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.zero_init_residual = zero_init_residual
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def _initialize_weights(self):
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if self.zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x


class ResNet64_N(BaseDiscriminator):

	def __init__(self, block, layers, N=8, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
		super(ResNet64_N, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)

		self.conv1 = nn.Conv2d(N, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)



	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


class ResNet64_N_BN(BaseDiscriminator):

	def __init__(self, block, layers, N=8, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
		super(ResNet64_N_BN, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)
		self.bn0 = nn.BatchNorm2d(N)
		self.conv1 = nn.Conv2d(N, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def forward(self, x):
		x = self.bn0(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


class ResNet64_N_IN(BaseDiscriminator):

	def __init__(self, block, layers, N=8, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
		super(ResNet64_N_IN, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)
		self.bn0 = nn.InstanceNorm2d(N)
		self.conv1 = nn.Conv2d(N, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def forward(self, x):
		x = self.bn0(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


class ResNet64_SmallerBN(BaseDiscriminator):

	def __init__(self, block, layers, num_classes=10, zero_init_residual=False, init_kernel=3, init_stride=1):
		super(ResNet64_SmallerBN, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		#                       bias=False)
		self.bn0 = nn.BatchNorm2d(3)
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 64 --> 64
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 --> 32
		#self.layer1 = self._make_layer(block, 64, layers[0])
		#self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 32 --> 16
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 16 --> 8
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 8 --> 4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # take average of 4x4 pixels to give 512 channels of 1x1 values
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.zero_init_residual = zero_init_residual

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _initialize_weights(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if self.zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def forward(self, x):
		x = self.bn0(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		#x = self.layer1(x)
		#x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


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
		block = InvertedResidual
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


def resnet18_64(num_classes=100):
	model = ResNet64(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
	return model


def resnet18_64_BN(num_classes=100):
	model = ResNet64_BN(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
	return model


def resnet_SmallerBN(num_classes=100):
	model = ResNet64_SmallerBN(BasicBlock3, [2, 2, 2, 2], num_classes=num_classes)
	return model


# def pretrained_resnet18_64(num_classes=100):
#     model = ResNet64(BasicBlock, [2, 2, 2, 2], 1000)
#     model.load_state_dict(torch.load('resnet18-5c106cde.pth'))
#     model.fc = FC(BasicBlock.expansion, num_classes)
#     return model


def resnet18_64_96(num_classes=100):
	model = ResNet64_96(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
	return model

# def resnet18_64_24(num_classes=100):
#     model = ResNet64_24(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
#     return model


def resnet18_64_N(N=12,num_classes=100):
	model = ResNet64_N(BasicBlock, [2, 2, 2, 2], N=N,num_classes=num_classes)
	return model


def resnet18_64_N_BN(N=12,num_classes=100):
	model = ResNet64_N_BN(BasicBlock, [2, 2, 2, 2], N=N,num_classes=num_classes)
	return model


def resnet18_64_N_IN(N=12,num_classes=100):
	model = ResNet64_N_IN(BasicBlock, [2, 2, 2, 2], N=N,num_classes=num_classes)
	return model


def resnet18_64_192(num_classes=100):
	model = ResNet64_192(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
	return model


def resnet18_64_1(num_classes=100):
	model = ResNet64_1(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
	return model


def resnet34_64(in_channels=3, num_classes=100):
	model = ResNet64(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)
	return model


def resnet50_64(num_classes=100):
	model = ResNet64(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
	return model
