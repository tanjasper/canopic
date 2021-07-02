# TO DO: for generator, plot and save losses instead of "accuracies"

import torchvision.models as models
import argparse
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import random
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.multi_net import model_from_checkpoint, EncoderDiscriminatorNet
from settings.util import load_encoder_settings, load_discriminator_settings
import fns_model
import losses_face as losses
from fns_all import *
from fns_face import prepare_face_data
import os
import data_fns
import warnings
import sys
import pprint
from datetime import datetime
from pytz import timezone

# some parameters
parser = argparse.ArgumentParser()

# architecture and losses:
parser.add_argument('--generator_loss', default='NegativeEntropyLoss')
parser.add_argument('--generator_loss_topk', default=100, type=int)
parser.add_argument('--input_dim_recog', default=109, type=int)
parser.add_argument('--input_dim_detect', default=95, type=int)
# data:
parser.add_argument('--tr_recog_filenames', default='most_popular_100_training_filenames.txt')
parser.add_argument('--val_recog_filenames', default='most_popular_100_val_filenames.txt')
parser.add_argument('--recog_imdir', default='data/vggface2')
parser.add_argument('--tr_detect_face_filenames', default='most_popular_100_training_filenames.txt')  # 'test400_30val_tr_filenames_tr30.txt'
parser.add_argument('--val_detect_face_filenames', default='most_popular_100_val_filenames.txt')  # 'test400_30val_tr_filenames_val30.txt'
parser.add_argument('--detect_face_imdir', default='data/vggface2')
parser.add_argument('--tr_detect_noface_filenames', default='training_faceless_filenames_random_60k.txt')
parser.add_argument('--val_detect_noface_filenames', default='val_faceless_filenames_2000_setA.txt')
parser.add_argument('--detect_noface_imdir', default='data/ILSVRC2012/images')
parser.add_argument('--root_data_dir', default='data/results')
parser.add_argument('--num_recog_classes', default=100, type=int)
# training:
parser.add_argument('--skip_pretrain', dest='skip_pretrain', action='store_true', help='skip pre-training of recognition and detection NNs (e.g. if they have been pretrained already')
parser.add_argument('--lrA', default=1e-2, type=float)
parser.add_argument('--lrB', default=1e-2, type=float)
parser.add_argument('--lr_generator', default=1e-2, type=float)
parser.add_argument('--lr_generator_detect', default=1e-2, type=float)
parser.add_argument('--max_discriminatorA_epochs', default=100, type=int)
parser.add_argument('--max_discriminatorB_epochs', default=10, type=int)
parser.add_argument('--maxA_pretrain_epochs', default=0, type=int)
parser.add_argument('--maxB_pretrain_epochs', default=0, type=int)
parser.add_argument('--wait_epochsA', default=10, type=int)
parser.add_argument('--wait_epochsB', default=5, type=int)
parser.add_argument('--generator_epochs', default=10, type=int)
parser.add_argument('--num_lrsA', default=3, type=int)
parser.add_argument('--num_lrsB', default=3, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--threshA', default=1, type=float, help='keep training discriminatorA until it hits this acc')
parser.add_argument('--threshB', default=1, type=float, help='keep training discriminatorB until it hits this acc')
parser.add_argument('--reinitialize_acc', default=0.1, type=float, help='reinitialize discriminatorA if it hits below this acc; from 0 to 1')
# data transformation:
parser.add_argument('--train_resize_recog', default=256, type=int, help='resize input training images to be of this size')
parser.add_argument('--train_pad', default=40, type=int, help='# of pad pixels per dim to add to training image')
parser.add_argument('--train_padding_mode', default='constant')
parser.add_argument('--train_min_scale', default=0.2, type=float)
parser.add_argument('--val_resize_recog', default=83, type=int, help='resize input training images to be of this size')
parser.add_argument('--val_pad', default=13, type=int, help='# of pad pixels per dim to add to val recog image')
parser.add_argument('--val_padding_mode', default='constant')
parser.add_argument('--train_resize_detect', default=256, type=int)
parser.add_argument('--train_centercrop_detect', default=256, type=int)
parser.add_argument('--val_resize_detect', default=95, type=int)
# neural net settings
parser.add_argument('--encoder_settings_json', type=str)
parser.add_argument('--discriminatorA_settings_json', default='json_templates/train_encoder/discriminators/resnet18IN_100class_in3_settings.json', type=str)
parser.add_argument('--discriminatorB_settings_json', default='json_templates/train_encoder/discriminators/resnet18IN_2class_in3_settings.json', type=str)
# other stuff:
parser.add_argument('--results_dir', default='two_discriminators')
parser.add_argument('--checkpoint_modelonly', dest='checkpoint_modelonly', action='store_true')
parser.add_argument('--proj_mat_init_path')
parser.add_argument('--num_threads', default=4, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--gpu_idx', default=0, type=int)
parser.add_argument('--print_freq', default=50, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--checkpoint_path', default='')
parser.add_argument('--save_dir', default='')
parser.add_argument('--measurement_rate', default=0.1, type=float)
parser.add_argument('--generator_path', default='')
parser.add_argument('--save_freq', default=50, type=int)
parser.add_argument('--seed', default=1313, type=int)
parser.add_argument('--entropy_weight', default=50, type=float)
parser.add_argument('--bce_weight', default=1, type=float)
parser.add_argument('--tv_reg', default=1e5, type=float)
parser.add_argument('--l1_reg', default=0.1, type=float)
parser.add_argument('--im_dim', default=64, type=int)
parser.add_argument('--change_embedding', dest='change_embedding', action='store_true')
parser.add_argument('--type_optimizer', default='Adam')
parser.add_argument('--debug', dest='debug', action='store_true')

parser.set_defaults(checkpoint_modelonly=False, skip_pretrain=False, change_embedding=False, learn_detect=False, debug=True)

def main():
	pretrained = False
	# (1) Set up:
	# call up parameters
	opt = parser.parse_args()

	start_epoch = 0
	# ignore Imagenet warnings
	warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)  # ignore imagenet EXIF warnings
	# random seed
	if opt.seed is not None:
		random.seed(opt.seed)
		torch.manual_seed(opt.seed)
		cudnn.deterministic = True
	# set up GPU
	torch.cuda.set_device(opt.gpu_idx)
	# create save directory if it doesn't exist
	save_dir = os.path.join(opt.root_data_dir, opt.save_dir, opt.results_dir)
	os.makedirs(save_dir, exist_ok=True)
	# let every print go to both stdout and to a logfile
	sys.stdout = Logger(save_dir)
	# print out setup info:
	print('======== Log ========')
	print(datetime.now(timezone('EST')))
	print('\n')
	print("Command ran:\n%s\n\n" % " ".join([x for x in sys.argv]))
	print("Opt:")
	pprint.pprint(vars(opt))
	print("\n")

	# (1) Prepare data
	train_recog_loader, train_detect_loader, train_detect_face_loader, train_detect_noface_loader, val_recog_loader, \
	val_detect_loader, val_detect_face_loader, val_detect_noface_loader = prepare_face_data(opt)

	# (2) Create EncoderDiscriminatorNet model
	if opt.checkpoint_path:  # load from checkpoint if given
		if opt.checkpoint_path == 'latest':
			checkpoint_path = os.path.join(save_dir, 'latest_network.tar')
		else:
			checkpoint_path = os.path.join(opt.root_data_dir, opt.checkpoint_path)
		model = model_from_checkpoint(checkpoint_path)
	else:
		# copy settings json files over to the save_dir to keep record
		shutil.copyfile(opt.encoder_settings_json, os.path.join(save_dir, 'encoder_settings.json'))
		shutil.copyfile(opt.discriminatorA_settings_json, os.path.join(save_dir, 'discriminatorA_settings.json'))
		shutil.copyfile(opt.discriminatorB_settings_json, os.path.join(save_dir, 'discriminatorB_settings.json'))
		# load each of the settings from the json files and create the model
		encoder_archs, encoder_settings = load_encoder_settings(opt.encoder_settings_json)
		discriminatorA_arch, discriminatorA_settings = load_discriminator_settings(opt.discriminatorA_settings_json)
		discriminatorB_arch, discriminatorB_settings = load_discriminator_settings(opt.discriminatorB_settings_json)
		model = EncoderDiscriminatorNet(opt, discriminatorA_arch, discriminatorB_arch, encoder_archs,
										discriminatorA_settings=discriminatorA_settings,
										discriminatorB_settings=discriminatorB_settings,
										encoder_settings=encoder_settings)
	model = model.cuda(opt.gpu_idx)
	print('# of parameters:' + str(sum(p.numel() for p in model.discriminatorA.parameters() if p.requires_grad)))

	# (3) Criterion and optimizers
	criterion = nn.CrossEntropyLoss().cuda(opt.gpu_idx)
	if opt.type_optimizer == 'Adam':
		optimizerA = torch.optim.Adam(model.discriminatorA.parameters(), opt.lrA, weight_decay=opt.weight_decay)
		optimizerB = torch.optim.Adam([
			{'params': model.discriminatorB.parameters()},
			{'params': model.encoder.parameters(), 'lr': opt.lr_generator_detect, 'weight_decay': 0}
		], lr=opt.lrB, weight_decay=opt.weight_decay)  # include both discriminatorB and projection
		optimizerOnlyB = torch.optim.Adam(model.discriminatorB.parameters(), opt.lrB, weight_decay=opt.weight_decay)
	else:
		optimizerA = torch.optim.SGD(model.discriminatorA.parameters(), opt.lrA, momentum=opt.momentum,
									 weight_decay=opt.weight_decay)
		optimizerB = torch.optim.SGD([
			{'params': model.discriminatorB.parameters()},
			{'params': model.encoder.parameters(), 'lr': opt.lr_generator_detect, 'weight_decay': 0}
		], lr=opt.lrB, momentum=opt.momentum,
			weight_decay=opt.weight_decay)  # include both discriminatorB and projection
		optimizerOnlyB = torch.optim.SGD(model.discriminatorB.parameters(), opt.lrB, momentum=opt.momentum,
										 weight_decay=opt.weight_decay)
	# loss for generator
	criterionG = losses.__dict__[opt.generator_loss](topk=opt.generator_loss_topk, num_classes=opt.num_recog_classes)
	optimizerG = torch.optim.SGD(model.encoder.parameters(), opt.lr_generator, momentum=opt.momentum)
	bd1 = 0
	bd2 = 0
	bd3 = 0
	bd4 = 0

	# (4) Define loss tracker
	accs = {'recog_tr_accs': [],
			'recog_tr_losses': [],
			'recog_val_accs': [],
			'recog_val_losses': [],
			'gen_tr_losses': [],
			'gen_val_losses': [],
			'gen_val_losses_post_det': [],
			'post_gen_val_accs': [],
			'post_det_val_accs': [],
			'recog_best_val_acc': 0,
			'recog_worst_val_acc': 1,
			'detect_tr_accs': [],
			'detect_tr_losses': [],
			'detect_val_accs': [],
			'detect_val_losses': [],
			'detect_best_val_acc': 0}
	checkpoint_info = {}

	# (5) if not opt.checkpoint_modelonly, then load optimizer parameters as well, but override them with user input
	if opt.checkpoint_path and not opt.checkpoint_modelonly:
		checkpoint = torch.load(checkpoint_path)
		optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
		optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
		if 'optimizerG_state_dict' in checkpoint:
			optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
		accs = checkpoint['accs']
		start_epoch = checkpoint['last_finished_epoch'] + 1
		if 'checkpoint_info' in checkpoint:
			checkpoint_info = checkpoint['checkpoint_info']
		# override checkpoint parameters with user-given parameters
		given_args = sys.argv
		if '--lrA' in given_args:
			optimizerA.param_groups[0]['lr'] = float(given_args[given_args.index('--lrA')+1])
		if '--lrB' in given_args:
			optimizerB.param_groups[0]['lr'] = float(given_args[given_args.index('--lrB')+1])
		if '--lr_generator_detect' in given_args:
			optimizerB.param_groups[1]['lr'] = float(given_args[given_args.index('--lr_generator_detect')+1])
		if '--lr_generator' in given_args:
			optimizerG.param_groups[0]['lr'] = float(given_args[given_args.index('--lr_generator')+1])

	# (6) If we did not load from a checkpoint, save this starting point
	if not opt.checkpoint_path:
		save_checkpoint({**{
			'optimizerA_state_dict': optimizerA.state_dict(),
			'accs': accs,
			'opt': opt,
			'checkpoint_info': checkpoint_info
		}, **model.checkpoint_info()}, False, False, False, save_dir, -1, opt)
		print('Done saving checkpoint')

	# (7) Pre-training training:
	if not opt.skip_pretrain:

		print('Starting pretraining -- recognition')

		# first, train recognition network
		if opt.maxA_pretrain_epochs and (not 'finished_recog' in checkpoint_info or not checkpoint_info['finished_recog']):
			model.freeze_generator()
			model.freeze_discriminatorB()
			model.unfreeze_discriminatorA()
			train_params = {
				'model': model,
				'criterion': criterion,
				'optimizer': optimizerA,
				'epoch': 0,
				'lr': opt.lrB,
				'wait_epochs': opt.wait_epochsB
			}
			tr_acc_recog, tr_loss_recog, val_acc_recog, val_loss_recog = train_with_decreasing_lr(
				train_recog_loader, train_params, opt, forward='A', max_epochs=opt.maxA_pretrain_epochs,
				num_lrs=opt.num_lrsA, thresh=opt.threshA, val_loader=val_recog_loader)
			checkpoint_info['finished_recog'] = True

		print('Saving checkpoint')
		save_dict = {**{
			'optimizerA_state_dict': optimizerA.state_dict(),
			'optimizerB_state_dict': optimizerB.state_dict(),
			'optimizerG_state_dict': optimizerG.state_dict(),
			'accs': accs,
			'last_finished_epoch': 0,
			'opt': opt,
			'checkpoint_info': checkpoint_info
		}, **model.checkpoint_info()}
		save_checkpoint(save_dict, False, False, False, save_dir, -2, opt)
		print('Done saving checkpoint')
		val_acc_recog, val_loss_recog = validate(val_recog_loader, model, criterion, opt, forward='A', retval='both')
		print('Final Recog Accuracy:' + str(val_acc_recog))
		print('Starting pretraining -- detection')

		# next, train detection network (without training generator)
		if opt.maxB_pretrain_epochs and (not 'finished_detect' in checkpoint_info or not checkpoint_info['finished_detect']):
			model.freeze_discriminatorA()
			model.unfreeze_discriminatorB()
			model.freeze_generator()
			# train discriminatorB
			train_params = {
				'model': model,
				'criterion': criterion,
				'optimizer': optimizerB,
				'epoch': 0,
				'lr': opt.lrB,
				'wait_epochs': opt.wait_epochsB
			}
			tr_acc_detect, tr_loss_detect, val_acc_detect, val_loss_detect = train_with_decreasing_lr(
				[train_detect_face_loader, train_detect_noface_loader], train_params, opt, forward='B',
				max_epochs=opt.maxB_pretrain_epochs, num_lrs=opt.num_lrsB, thresh=opt.threshB,
				val_loader=val_detect_loader)
			checkpoint_info['finished_detect'] = True
		val_acc_detect, val_loss_detect = validate(val_detect_loader, model, criterion, opt, forward='B', retval='both')
		print('Final Detect Accuracy:'+str(val_acc_detect))

		pretrained = True

		print('Done pretraining')
		print('Saving checkpoint')
		save_dict = {**{
			'optimizerA_state_dict': optimizerA.state_dict(),
			'optimizerB_state_dict': optimizerB.state_dict(),
			'optimizerG_state_dict': optimizerG.state_dict(),
			'accs': accs,
			'last_finished_epoch': 0,
			'opt': opt,
			'checkpoint_info': checkpoint_info
		}, **model.checkpoint_info()}
		save_checkpoint(save_dict, False, False, False, save_dir, -2, opt)
		print('Done saving checkpoint')

		if opt.change_embedding:
			print(model.proj.proj.weight.shape)
			krn = np.load(os.path.join(opt.root_data_dir, 'inits/conv4_1_11_bars.npy'))
			krnl = torch.tensor(krn, requires_grad=True)
			model.proj.proj.weight.data.copy_(krnl)
			model.proj.proj.weight.requires_grad = True
			print(model.proj.proj.weight.requires_grad)   

	# (8) Iterative training:
	just_reinitialized = False
	for epoch in range(start_epoch, opt.num_epochs):
		# Trains recog with an impostor class for a fixed num of epochs --> gen with BCE and entropy for a fixed num of epochs --> detect for fixed num of epochs
		# IMPORTANT: USE filename that has face + ilsvrc instead of face only for recog

		# Prepare model for training encoder
		model.freeze_discriminatorA()
		model.freeze_discriminatorB()
		model.unfreeze_generator()
		model.train_generator_update_encoder(epoch)  # perform any epoch-specific updates for the encoder
		train_params = {
			'model': model,
			'criterionG': criterionG,
			'criterion': criterion,
			'optimizerG': optimizerG,
			'epoch': epoch,
			'lr': opt.lr_generator,
		}

		# Perform training
		tr_acc_detect, tr_loss_detect, tr_loss_privacy, val_acc_detect, val_loss_detect = train_fix_epochs_onlygenerator_sepdata(
			[train_recog_loader,train_detect_loader], train_params, opt,
			max_epochs=opt.generator_epochs, num_lrs=opt.num_lrsB, thresh=opt.threshB,
			val_loader=val_detect_loader)
		# tr_acc_detect = tr_acc_detect if tr_acc_detect != 0 else accs['detect_tr_accs'][-1]
		# tr_loss_detect = tr_loss_detect if tr_loss_detect != 0 else accs['detect_tr_losses'][-1]

		# validate discriminatorA again
		val_acc_privacy, gen_postdetect_val_loss = validate(val_recog_loader, model, criterionG, opt, forward='A',
															retval='both')

		# train discriminatorA
		model.freeze_generator()
		model.freeze_discriminatorB()
		model.unfreeze_discriminatorA()
		train_params = {
			'model': model,
			'criterion': criterion,
			'optimizer': optimizerA,
			'epoch': epoch,
			'lr': opt.lrA,
			'wait_epochs': opt.wait_epochsA
		}
		tr_acc_recog, tr_loss_recog, val_acc_recog, val_loss_recog = train_with_decreasing_lr(
			train_recog_loader, train_params, opt, forward='A', max_epochs=opt.max_discriminatorA_epochs,
			num_lrs=opt.num_lrsA, thresh=opt.threshA, val_loader=val_recog_loader)
		val_acc_recog, val_loss_recog = validate(val_recog_loader, model, criterion, opt, forward='A',
												 retval='both')
		if val_acc_recog*0.01 < opt.reinitialize_acc and not just_reinitialized:
			model.discriminatorA._initialize_weights()
			tr_acc_recog, tr_loss_recog, val_acc_recog, val_loss_recog = train_with_decreasing_lr(
				train_recog_loader, train_params, opt, forward='A', max_epochs=opt.max_discriminatorA_epochs,
				num_lrs=opt.num_lrsA, thresh=opt.threshA, val_loader=val_recog_loader)
			just_reinitialized = True
		else:
			just_reinitialized = False

		val_acc_recog, val_loss_recog = validate(val_recog_loader, model, criterion, opt, forward='A',
												 retval='both')

		# if training was not performed, record previous training accuracy/loss
		tr_acc_recog = tr_acc_recog if tr_acc_recog != 0 or epoch == 0 else accs['recog_tr_accs'][-1]
		tr_loss_recog = tr_loss_recog if tr_loss_recog != 0 or epoch == 0 else accs['recog_tr_losses'][-1]

		# validate discriminatorA
		val_acc_privacy_predetect, gen_val_loss = validate(val_recog_loader, model, criterionG, opt, forward='A',
														   retval='both')

		#train detector only
		model.freeze_generator()
		model.freeze_discriminatorA()
		model.unfreeze_discriminatorB()
		train_params = {
			'model': model,
			'criterion': criterion,
			'optimizer': optimizerOnlyB,
			'epoch': epoch,
			'lr': opt.lrB,
			'wait_epochs': opt.wait_epochsB
		}

		tr_acc_detect, tr_loss_detect, val_acc_detect, val_loss_detect = train_with_decreasing_lr(
			train_detect_loader, train_params, opt, forward='B', max_epochs=opt.max_discriminatorB_epochs,
			num_lrs=opt.num_lrsB, thresh=opt.threshB, val_loader=val_detect_loader)

		val_acc_detect, val_loss_detect = validate(val_detect_loader, model, criterion, opt, forward='B',
												   retval='both')

		if opt.debug:
			savename_suffix = 'postdiscB_epoch%d' % epoch
			model.save_debug_data(save_dir, savename_suffix=savename_suffix)

		# Record losses and accuracies
		accs['recog_tr_accs'].append(tr_acc_recog)
		accs['recog_val_accs'].append(val_acc_recog)
		accs['recog_tr_losses'].append(tr_loss_recog)
		accs['recog_val_losses'].append(val_loss_recog)
		accs['gen_tr_losses'].append(tr_loss_privacy)
		accs['detect_tr_accs'].append(tr_acc_detect)
		accs['detect_tr_losses'].append(tr_loss_detect)
		accs['detect_val_accs'].append(val_acc_detect)
		accs['detect_val_losses'].append(val_loss_detect)
		# adjust bests and worsts
		is_best = val_acc_recog > accs['recog_best_val_acc']
		accs['recog_best_val_acc'] = max(val_acc_recog, accs['recog_best_val_acc'])
		is_worst = val_acc_recog < accs['recog_worst_val_acc']
		accs['recog_worst_val_acc'] = min(val_acc_recog, accs['recog_worst_val_acc'])
		is_best_detect = val_acc_detect > accs['detect_best_val_acc']
		accs['detect_best_val_acc'] = max(val_acc_detect, accs['detect_best_val_acc'])
		np.save(os.path.join(save_dir,'accuracies.npy'),accs)
		print('Saving checkpoint')
		save_dict = {**{
			'optimizerA_state_dict': optimizerA.state_dict(),
			'optimizerB_state_dict': optimizerB.state_dict(),
			'optimizerG_state_dict': optimizerG.state_dict(),
			'accs': accs,
			'last_finished_epoch': epoch,
			'opt': opt,
			'checkpoint_info': checkpoint_info
		}, **model.checkpoint_info()}
		save_checkpoint(save_dict, is_best, is_best_detect, is_worst, save_dir, epoch, opt)
		print('Done saving checkpoint')

		dict_save = {**{
			'optimizerA_state_dict': optimizerA.state_dict(), 'optimizerB_state_dict': optimizerB.state_dict(),
			'optimizerG_state_dict': optimizerG.state_dict(),
			'accs': accs,
			'last_finished_epoch': epoch,
			'opt': opt,
			'checkpoint_info': checkpoint_info
		}, **model.checkpoint_info()}
		if val_acc_detect > bd1:
			bd1 = val_acc_detect
			print('saving best at a det acc of:'+str(val_acc_detect))
			torch.save(dict_save, os.path.join(save_dir, 'best_network.tar'))
		if val_acc_detect < bd1 and val_acc_detect >= bd2:
			bd2 = val_acc_detect
			print('saving second best at a det acc of:'+str(val_acc_detect))
			torch.save(dict_save, os.path.join(save_dir, 'second_best_network.tar'))
		if val_acc_detect < bd2 and val_acc_detect >= bd3:
			bd3 = val_acc_detect
			print('saving third best at a det acc of:'+str(val_acc_detect))
			torch.save(dict_save, os.path.join(save_dir, 'third_best_network.tar'))
		if val_acc_detect < bd3 and val_acc_detect >= bd4:
			bd4 = val_acc_detect
			print('saving fourth best at a det acc of:'+str(val_acc_detect))
			torch.save(dict_save, os.path.join(save_dir, 'fourth_best_network.tar'))

		# plot and print
		plot_accs = [accs['recog_tr_accs'], accs['recog_val_accs'], accs['detect_tr_accs'], accs['detect_val_accs']]
		plot_legend = ['recog tr', 'recog val', 'detect tr', 'detect val']
		save_plot(accs['detect_best_val_acc'], accs['recog_best_val_acc'], plot_accs, os.path.join(save_dir, 'acc_plots.pdf'), plot_legend, scale=0.01, lim01=True)
		plot_accs = [accs['recog_tr_losses'], accs['recog_val_losses'], accs['gen_tr_losses'],
					 accs['detect_tr_losses'], accs['detect_val_losses']]
		plot_legend = ['recog tr loss', 'recog val loss', 'gen loss', 'detect tr loss', 'detect val loss']
		save_plot(accs['detect_best_val_acc'], accs['recog_best_val_acc'],plot_accs, os.path.join(save_dir, 'loss_plots.pdf'), plot_legend, scale=0.01)


# This function returns 8 data-loaders: 4 training and 4 validation
# There are 3 types of data: recognition images, face images for detection, and no-face images for detection
# Each have their own respective preprocessing transformations
# The 4 data loaders for both training and validation are:
#   (1) x_recog_loader -- data loader for face recognition
#   (2) x_detect_loader -- data loader that gives both face and no-face images for detection (not used for training)
#   (3) x_detect_face_loader -- data loader that gives only the face images for detection
#   (4) x_detect_noface_loader -- data loader that gives only the no-face images for detection
def prepare_face_data(opt):
	im_dim = opt.im_dim
	# (1) define transforms
	train_recog_transforms = transforms.Compose([
		transforms.RandomResizedCrop(im_dim),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
	])
	val_recog_transforms = transforms.Compose([
		transforms.Resize(round(im_dim / 0.875)),
		transforms.CenterCrop(im_dim),
		transforms.ToTensor(),
	])
	train_detect_face_transforms = transforms.Compose([
		transforms.RandomResizedCrop(im_dim),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
	])
	val_detect_face_transforms = transforms.Compose([
		transforms.Resize(opt.val_resize_detect),
		transforms.CenterCrop(opt.input_dim_detect),
		transforms.ToTensor(),
	])
	val_detect_face_transforms2 = transforms.Compose([
		transforms.Resize(round(im_dim / 0.875)),
		transforms.CenterCrop(im_dim),
		transforms.ToTensor(),
	])
	train_detect_noface_transforms = transforms.Compose([
		transforms.RandomResizedCrop(im_dim, scale=(0.2, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
	])
	val_detect_noface_transforms = transforms.Compose([
		transforms.CenterCrop(im_dim),
		transforms.ToTensor(),
	])

	# (2) parse filenames
	tr_recog_filenames = os.path.join(opt.recog_imdir, 'filenames', opt.tr_recog_filenames)
	val_recog_filenames = os.path.join(opt.recog_imdir, 'filenames', opt.val_recog_filenames)
	tr_detect_face_filenames = os.path.join(opt.detect_face_imdir, 'filenames', opt.tr_detect_face_filenames)
	val_detect_face_filenames = os.path.join(opt.detect_face_imdir, 'filenames', opt.val_detect_face_filenames)
	tr_detect_noface_filenames = os.path.join(opt.detect_noface_imdir, 'filenames', opt.tr_detect_noface_filenames)
	val_detect_noface_filenames = os.path.join(opt.detect_noface_imdir, 'filenames', opt.val_detect_noface_filenames)

	# (3) define datasets
	train_detect_dataset = data_fns.DatasetFromMultipleFilenames(
		[opt.detect_face_imdir, opt.detect_noface_imdir], [tr_detect_face_filenames, tr_detect_noface_filenames],
		train_detect_face_transforms)

	train_recog_dataset = data_fns.DatasetFromFilenames(opt.recog_imdir, tr_recog_filenames, train_recog_transforms)

	train_detect_face_dataset = data_fns.DatasetFromFilenames(
		opt.detect_face_imdir, tr_detect_face_filenames, train_detect_face_transforms)

	train_detect_noface_dataset = data_fns.DatasetFromFilenames(
		opt.detect_noface_imdir, tr_detect_noface_filenames, train_detect_noface_transforms)

	val_detect_dataset = data_fns.DatasetFromMultipleFilenames(
		[opt.detect_face_imdir, opt.detect_noface_imdir], [val_detect_face_filenames, val_detect_noface_filenames],
		val_detect_face_transforms2)

	val_recog_dataset = data_fns.DatasetFromFilenames(opt.recog_imdir, val_recog_filenames, val_recog_transforms)

	val_detect_face_dataset = data_fns.DatasetFromFilenames(opt.detect_face_imdir, val_detect_face_filenames,
															val_detect_face_transforms)

	val_detect_noface_dataset = data_fns.DatasetFromFilenames(opt.detect_noface_imdir, val_detect_noface_filenames,
															  val_detect_noface_transforms)

	# (4) random samplers
	tr_recog_weights_per_class = torch.from_numpy(train_recog_dataset.weights)
	tr_weights = tr_recog_weights_per_class[train_recog_dataset.labels]
	# tr_recog_sampler -- weighted sampler to handle imbalanced dataset
	tr_recog_sampler = torch.utils.data.WeightedRandomSampler(weights=tr_weights, num_samples=len(tr_weights))
	# noface_sampler -- make noface loader yield same # of images as face loader for detection
	noface_sampler = data_fns.Pytorch1RandomSampler(train_detect_noface_dataset, replacement=True,
													num_samples=len(train_detect_face_dataset))

	# (5) data loaders
	train_detect_loader = torch.utils.data.DataLoader(
		train_detect_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
	train_recog_loader = torch.utils.data.DataLoader(
		train_recog_dataset, batch_size=opt.batch_size, sampler=tr_recog_sampler, num_workers=opt.num_threads, pin_memory=True)
	train_detect_face_loader = torch.utils.data.DataLoader(
		train_detect_face_dataset, batch_size=int(opt.batch_size/2), shuffle=True, num_workers=opt.num_threads, pin_memory=True)
	train_detect_noface_loader = torch.utils.data.DataLoader(
		train_detect_noface_dataset, batch_size=int(opt.batch_size/2), sampler=noface_sampler, num_workers=opt.num_threads, pin_memory=True)
	val_recog_loader = torch.utils.data.DataLoader(
		val_recog_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads, pin_memory=True)
	val_detect_loader = torch.utils.data.DataLoader(
		val_detect_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads, pin_memory=True)
	val_detect_face_loader = torch.utils.data.DataLoader(
		val_detect_face_dataset, batch_size=int(opt.batch_size/2), shuffle=False, num_workers=opt.num_threads, pin_memory=True)
	val_detect_noface_loader = torch.utils.data.DataLoader(
		val_detect_noface_dataset, batch_size=int(opt.batch_size/2), shuffle=False, num_workers=opt.num_threads, pin_memory=True)

	return train_recog_loader, train_detect_loader, train_detect_face_loader, train_detect_noface_loader, \
		   val_recog_loader, val_detect_loader, val_detect_face_loader, val_detect_noface_loader


def save_checkpoint(save_dict, is_best, is_best_detect, is_worst, save_dir, epoch, opt):
	if epoch == -1:
		torch.save(save_dict, os.path.join(save_dir, 'disc_network_init.tar'))
		return
	elif epoch == -2:
		torch.save(save_dict, os.path.join(save_dir, 'disc_network_pretrain.tar'))
		return
	# first, save the network as the latest network
	print('Saving the latest')
	torch.save(save_dict, os.path.join(save_dir, 'latest_network.tar'))
	print('Saved')
	if epoch % opt.save_freq == 0:
		torch.save(save_dict, os.path.join(save_dir, 'disc_network_epoch%d.tar' % (epoch)))
	# if it is the best network, replace old best network
	if is_best:
		torch.save(save_dict, os.path.join(save_dir, 'best_recog_network.tar'))
	if is_best_detect:
		torch.save(save_dict, os.path.join(save_dir, 'best_detect_network.tar'))
	if is_worst:
		torch.save(save_dict, os.path.join(save_dir, 'worst_recog_network.tar'))


def save_plot(best_det, best_rec, accs, savename, legend, scale=1, stride=1, lim01=False):

	f = plt.figure()

	# print each plot_val
	for i in range(len(accs)):
		xp = range(len(accs[i][0::stride]))
		plt.plot(xp, [z * scale for z in accs[i][0::stride]])
	plt.legend(legend)
	if lim01:
		plt.ylim(0, 1)
	else:
		plt.ylim(-0.8, 0.8)
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Detect:'+str(best_det)+', Recog:'+str(best_rec))
	# plt.ylim(-1, 1)
	plt.show()
	f.savefig(savename, bbox_inches='tight')
	plt.close(f)


class Logger(object):
	def __init__(self, save_dir):
		self.terminal = sys.stdout
		self.log = open(os.path.join(save_dir, "log.txt"), "a+")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		#this flush method is needed for python 3 compatibility.
		#this handles the flush command by doing nothing.
		#you might want to specify some extra behavior here.
		pass


if __name__ == '__main__':
	main()



















