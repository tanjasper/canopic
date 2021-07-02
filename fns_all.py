import time
import torch
import os
import shutil
import numpy as np


def train_step(x, target, models, criterion, optimizer):

	for model in models:
		x = model(x)
	loss = criterion(x, target)
	acc = accuracy(x, target)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss.item(), acc[0].item()


def train_with_decreasing_lr(train_loader, train_params, opt, forward='', num_lrs=3, thresh=1, max_epochs=1000,
							 val_loader=None, best_val_model=True, return_multi_accs=False, decrease_lr=True,
							 save_checkpoint_path='', save_checkpoint_freq=50, load_checkpoint_path=''):
	"""General training function with the following features: decreasing lr, best val model, and early stopping

	Allows general training of EncoderDiscriminatorNet
	Can take in arbitrary criterion and optimizer that works for EncoderDiscriminatorNet, thus allowing
		one to choose which parameters and which type of training to perform
	Always returns 4 objects representing training loss, training accuracy, validation loss, validation
		accuracy even if validation wasn't performed (will return 0 or [] for the latter 2). Programs using
		this function should always expect 4 values even if not passing in any val_loader.

	Training features available:
	Decreasing lr:
		Learning rate is decreased by a factor of 10 if the validation accuracy has not increased in
		train_params['wait_epochs'] number of epochs. If this occurs after num_lrs learning rates, training
		is stopped. Set decrease_lr=True and set appropriate train_params['wait_epochs'] and num_lrs to use this.
	Best val model:
		The model is reverted back to the epoch that it had the highest validation accuracy for. This is similar
		to early stopping, but training is performed all the way (it is not stopped early) Set best_val_model=True
		to use this.
	Early stopping:
		Stop training once the validation accuracy hits a certain threshold. Set a thresh parameter < 1 to use this.

	Note: if neither Decreasing LR or Early Stopping are employed, then validation will not need to be performed
	during training. Thus, scalars will be returned for validation results even if return_multi_accs=True

	Args:
		train_loader: data loader for training
		train_params: dictionary containing keys 'model', 'criterion', 'optimizer', 'epoch', 'lr', 'wait_epochs'
		opt: parser TODO: what is this used for?
		forward: 'A' or 'B', depending on which discriminator we wish to train
		num_lrs: number of learning rates to decrease through
		thresh: validation accuracy (from 0 to 1) after which training stops (deactivated by default, i.e. set to 1)
		max_epochs: maximum total number of epochs to run training for
		val_loader: validation data loader
		best_val_model (bool): choose the model that has the best validation accuracy (else, the last model)
		return_multi_accs (bool): whether to return accuracies across all epochs (or just the best accuracy)
		decrease_lr (bool): whether to perform decreasing learning rates or not (num_lrs ignored if this is False)
		save_checkpoint_path (str): where to save checkpoints (if unspecified, do not save checkpoints)
		save_checkpoint_freq (int): how often in iterations to save checkpoint
		load_checkpoint_path (str): path to checkpoint to load (if given)

	TODO:
		Change the name of this function?

	Returns:
		float: training accuracy
		float: training loss
		float: validation accuracy
		float: validation loss
	"""

	def prepare_retval():
		"""Helper function to choose among the many possible return values"""
		# if a test loader was given, then test accuracy must be returned
		if return_multi_accs:  # return list of values
			if perform_inner_val:  # validation was performed in each training epoch
				return tr_accs, tr_losses, val_accs, val_losses
			else:
				return tr_accs, tr_losses, final_val_acc, final_val_loss
		if best_val_model:  # if best_val_model, return accuracies at the best validation epoch
			if best_val_epoch == -1:
				return 0, 0, final_val_acc, final_val_loss
			else:
				return tr_losses[best_val_epoch], tr_accs[best_val_epoch], val_losses[best_val_epoch], val_accs[best_val_epoch]
		else:  # without early stopping, simply return the last accuracy
			return tr_accs[-1], tr_losses[-1], final_val_acc, final_val_loss

	# parse train_parameters
	model = train_params['model']
	criterion = train_params['criterion']
	optimizer = train_params['optimizer']
	epoch = train_params['epoch']
	lr = train_params['lr']
	wait_epochs = train_params['wait_epochs'] if decrease_lr else max_epochs  # max_epochs for safety, shouldn't matter

	# condition on when validation should be performed within training epochs
	if thresh < 1 or best_val_model or decrease_lr:
		perform_inner_val = True
	else:
		perform_inner_val = False

	# Ensure that if validation is required that a val_loader was provided (according to conditions a few lines up)
	if perform_inner_val:
		assert val_loader is not None, "train_with_decreasing_lr -- missing val_loader when it is required"

	# variables for decreasing learning rate
	start_epoch = 0  # set as variable to override with checkpoint if given
	this_lr_accs = []
	curr_lr_round = 1

	# tracking of results
	best_val_epoch = -1
	tr_losses = []
	tr_accs = []
	val_losses = []
	val_accs = []
	best_val_acc = 0
	final_val_acc = 0
	final_val_loss = 0
	model_state_dict = None

	# Load checkpoint if given
	if load_checkpoint_path:
		print('Train with decreasing lr -- loading checkpoint from ' + load_checkpoint_path)
		load_checkpoint = torch.load(load_checkpoint_path)
		model.load_state_dict(load_checkpoint['model_state_dict'])
		start_epoch = load_checkpoint['epoch'] + 1
		this_lr_accs = load_checkpoint['this_lr_accs']
		curr_lr_round = load_checkpoint['curr_lr_round']
		tr_accs = load_checkpoint['tr_accs']
		val_accs = load_checkpoint['val_accs']
		decrease_learning_rate(lr, optimizer, curr_lr_round - 1)
		if 'best_model_state_dict' in load_checkpoint:
			best_model_state_dict = load_checkpoint['best_model_state_dict']
		else:
			best_model_state_dict = load_checkpoint['model_state_dict']

	# If early stopping (thresh < 1), validate first to see if we even need to train:
	if thresh < 1:
		val_acc, val_loss = validate(val_loader, model, criterion, opt, forward=forward, retval='both')
		if val_acc > thresh * 100:
			print('Threshold achieved without training. Returning model as is.')
			final_val_acc, final_val_loss = (val_acc, val_loss)
			return prepare_retval()

	# Perform training
	for j in range(start_epoch, max_epochs):

		# Train step
		print('Training discriminator %s: iteration %d, lr round: %d' % (forward, j, curr_lr_round))
		tr_acc, tr_loss = train(train_loader, model, criterion, optimizer, epoch, opt, forward=forward, retval='both')
		tr_accs.append(tr_acc)
		tr_losses.append(tr_loss)

		# Validation step (if needed). Includes various training features that have to do with validation.
		if perform_inner_val:
			val_acc, val_loss = validate(val_loader, model, criterion, opt, forward=forward, retval='both')
			val_accs.append(val_acc)
			val_losses.append(val_loss)
			# Check if this is the best validation accuracy
			if best_val_model:
				if val_acc >= best_val_acc:
					best_val_acc = val_acc
					best_val_epoch = j
					best_model_state_dict = model.state_dict()
			# Check early stopping
			if val_acc > thresh * 100:
				print('Threshold achieved. Stopping training now.')
				return prepare_retval()
			# Check decreasing lr (condition: accuracy has not increased for wait_epochs epochs)
			if decrease_lr:
				this_lr_accs.append(val_acc)
				if len(this_lr_accs) >= wait_epochs and max(this_lr_accs[-wait_epochs:]) == this_lr_accs[-wait_epochs]:
					if curr_lr_round == num_lrs:
						print('Giving up on training discriminator A after %d iterations' % j)
						return prepare_retval()
					else:
						print('Decreasing lr (starting round %d)' % (curr_lr_round + 1))
						curr_lr_round += 1
						decrease_learning_rate(lr, optimizer, curr_lr_round - 1)
						this_lr_accs = []
		if j == max_epochs-1:
			print('Reached max_epochs of %d for training' % max_epochs)
		if j > 0 and j % save_checkpoint_freq == 0 and save_checkpoint_path:
			torch.save({'model_state_dict': model.state_dict(), 'epoch': j, 'this_lr_accs': this_lr_accs,
						'curr_lr_round': curr_lr_round, 'tr_accs': tr_accs, 'val_accs': val_accs,
						'best_model_state_dict': best_model_state_dict},
						os.path.join(save_checkpoint_path, 'train_latest_checkpoint.tar'))

	# Perform final validation if user gave a validation loader but no validation was performed during training
	if not perform_inner_val and val_loader:
		final_val_acc, final_val_loss = validate(val_loader, model, criterion, opt, forward=forward, retval='both')

	# If user wants model at the best validation epoch, reload that model
	if best_val_model:
		model.load_state_dict(best_model_state_dict)

	return prepare_retval()


def train_fix_epochs_onlygenerator(train_loader, train_params, opt, num_lrs=3, thresh=0, max_epochs=1000, val_loader=None):

	# parse train_parameters
	model = train_params['model']
	criterion = train_params['criterion']
	criterionG = train_params['criterionG']
	optimizerG = train_params['optimizerG']
	epoch = train_params['epoch']
	lr = train_params['lr']
	# wait_epochs = train_params['wait_epochs']

	# Ensure that if a thresh is given, then so is a val_loader
	if thresh > 0:
		assert val_loader is not None, "in train_with_decreasing_lr, threshold is given but missing val_loader"

	# train discriminatorA
	this_lr_accs = []
	curr_lr_round = 1
	tr_acc = 0
	tr_loss = 0
	val_acc = 0
	val_loss = 0
	for j in range(max_epochs):
		# val_accA, val_lossA = validate(val_loader, model, criterion, opt, forward='A', retval='both')
		val_accB, val_lossB = validate(val_loader, model, criterion, opt, forward='B', retval='both')
		tr_accB, tr_lossA, tr_lossB = train_onlygenerator(train_loader, model, criterion, criterionG, optimizerG, epoch, opt, retval='both')
		this_lr_accs.append(tr_acc)
	print('Reached max_epochs of %d for training' % max_epochs)
	return tr_accB, tr_lossA, tr_lossB, val_accB, val_lossB



def train_fix_epochs_onlygenerator_sepdata(train_loader, train_params, opt, num_lrs=3, thresh=0, max_epochs=1000, val_loader=None):

	# parse train_parameters
	model = train_params['model']
	criterion = train_params['criterion']
	criterionG = train_params['criterionG']
	optimizerG = train_params['optimizerG']
	epoch = train_params['epoch']
	lr = train_params['lr']
	# wait_epochs = train_params['wait_epochs']

	# Ensure that if a thresh is given, then so is a val_loader
	if thresh > 0:
		assert val_loader is not None, "in train_with_decreasing_lr, threshold is given but missing val_loader"

	# train discriminatorA
	this_lr_accs = []
	curr_lr_round = 1
	tr_acc = 0
	tr_loss = 0
	val_acc = 0
	val_loss = 0
	for j in range(max_epochs):
		# val_accA, val_lossA = validate(val_loader, model, criterion, opt, forward='A', retval='both')
		val_accB, val_lossB = validate(val_loader, model, criterion, opt, forward='B', retval='both')
		tr_accB, tr_lossA, tr_lossB = train_onlygenerator_sepdata(train_loader, model, criterion, criterionG, optimizerG, epoch, opt, retval='both')
		this_lr_accs.append(tr_acc)
	print('Reached max_epochs of %d for training' % max_epochs)
	return tr_accB, tr_lossA, tr_lossB, val_accB, val_lossB



def train_fix_epochs_onlygenerator_multiop(train_loader, train_params, opt, num_lrs=3, thresh=0, max_epochs=1000, val_loader=None):

	# parse train_parameters
	model = train_params['model']
	criterion = train_params['criterion']
	criterionG = train_params['criterionG']
	optimizerG = train_params['optimizerG']
	epoch = train_params['epoch']
	lr = train_params['lr']
	# wait_epochs = train_params['wait_epochs']

	# Ensure that if a thresh is given, then so is a val_loader
	if thresh > 0:
		assert val_loader is not None, "in train_with_decreasing_lr, threshold is given but missing val_loader"

	# train discriminatorA
	this_lr_accs = []
	curr_lr_round = 1
	tr_acc = 0
	tr_loss = 0
	val_acc = 0
	val_loss = 0
	for j in range(max_epochs):
		# val_accA, val_lossA = validate(val_loader, model, criterion, opt, forward='A', retval='both')
		val_accB, val_lossB = validate_multiop(val_loader, model, criterion, opt, forward='B', retval='both')
		tr_accB, tr_lossA, tr_lossB = train_onlygenerator_multiop(train_loader, model, criterion, criterionG, optimizerG, epoch, opt, retval='both')
		this_lr_accs.append(tr_acc)
	print('Reached max_epochs of %d for training' % max_epochs)
	return tr_accB, tr_lossA, tr_lossB, val_accB, val_lossB


def train_fix_epochs_genloss(train_loader, train_params, opt, forward='', num_lrs=3, thresh=0, max_epochs=1000, val_loader=None):

	# parse train_parameters
	model = train_params['model']
	criterion = train_params['criterion']
	optimizer = train_params['optimizer']
	criterionG = train_params['criterionG']
	optimizerG = train_params['optimizerG']
	epoch = train_params['epoch']
	lr = train_params['lr']
	wait_epochs = train_params['wait_epochs']

	# Ensure that if a thresh is given, then so is a val_loader
	if thresh > 0:
		assert val_loader is not None, "in train_with_decreasing_lr, threshold is given but missing val_loader"

	# train discriminatorA
	this_lr_accs = []
	curr_lr_round = 1
	tr_acc = 0
	tr_loss = 0
	val_acc = 0
	val_loss = 0
	for j in range(max_epochs):

		# If threshold is nonzero, check if we have already achieved it:
		val_acc, val_loss = validate(val_loader, model, criterion, opt, forward=forward, retval='both')
		if val_acc > thresh*100:
			print('Threshold achieved after %d iterations' % (j-1))
			return tr_acc, tr_loss, val_acc, val_loss
		print('Training discriminator %s: iteration %d' % (forward, j))
		tr_acc, tr_loss = train_genloss(train_loader, model, criterion, criterionG, optimizerG, optimizer, epoch, opt, forward=forward, retval='both')
		this_lr_accs.append(tr_acc)
	print('Reached max_epochs of %d for training' % max_epochs)
	return tr_acc, tr_loss, val_acc, val_loss

def train_fix_epochs_genloss_new(train_loader, train_params, opt, forward='', num_lrs=3, thresh=0, max_epochs=1000, val_loader=None):

	# parse train_parameters
	model = train_params['model']
	criterion = train_params['criterion']
	optimizer = train_params['optimizer']
	criterionG = train_params['criterionG']
	optimizerG = train_params['optimizerG']
	epoch = train_params['epoch']
	lr = train_params['lr']
	wait_epochs = train_params['wait_epochs']

	# Ensure that if a thresh is given, then so is a val_loader
	if thresh > 0:
		assert val_loader is not None, "in train_with_decreasing_lr, threshold is given but missing val_loader"

	# train discriminatorA
	this_lr_accs = []
	curr_lr_round = 1
	tr_acc = 0
	tr_loss = 0
	val_acc = 0
	val_loss = 0
	gn_loss = 0
	for j in range(max_epochs):

		# If threshold is nonzero, check if we have already achieved it:
		val_acc, val_loss = validate(val_loader, model, criterion, opt, forward=forward, retval='both')
		if val_acc > thresh*100:
			print('Threshold achieved after %d iterations' % (j-1))
			return tr_acc, tr_loss, val_acc, val_loss
		print('Training discriminator %s: iteration %d' % (forward, j))
		tr_acc, tr_loss, gn_loss = train_genloss_new(train_loader, model, criterion, criterionG, optimizerG, optimizer, epoch, opt, forward=forward, retval='both')
		this_lr_accs.append(tr_acc)
	print('Reached max_epochs of %d for training' % max_epochs)
	return tr_acc, tr_loss, val_acc, val_loss, gn_loss


def train_alternate(loader_recog, loader_face, loader_noface, model, criterion, optimizer, epoch, opt, retval='both', train_recognition=True):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses_recog = AverageMeter()
	losses_privacy = AverageMeter()
	losses_detect = AverageMeter()
	top1_recog = AverageMeter()
	top1_privacy = AverageMeter()
	top1_detect = AverageMeter()

	# parse criterions and optimizers
	criterion_recog = criterion['recog']
	criterion_privacy = criterion['privacy']
	criterion_detect = criterion['detect']
	optimizer_recog = optimizer['recog']
	optimizer_privacy = optimizer['privacy']
	optimizer_detect = optimizer['detect']

	# switch to train mode
	model.train()

	end = time.time()
	for i, data in enumerate(zip(loader_recog, loader_face, loader_noface)):

		input_recog = data[0][0]
		target_recog = data[0][1]
		input_detect = torch.cat((data[1][0], data[2][0]))
		target_detect = torch.Tensor(data[1][0].shape[0]*[0] + data[2][0].shape[0]*[1]).type(torch.int64)

		# measure data loading time
		data_time.update(time.time() - end)

		input_recog = input_recog.cuda(opt.gpu_idx, non_blocking=True)
		target_recog = target_recog.cuda(opt.gpu_idx, non_blocking=True)
		input_detect = input_detect.cuda(opt.gpu_idx, non_blocking=True)
		target_detect = target_detect.cuda(opt.gpu_idx, non_blocking=True)

		# Train for recognition
		if train_recognition:
			model.freeze_generator()
			model.freeze_discriminatorB()
			model.unfreeze_discriminatorA()
			loss, acc = train_step(input_recog, target_recog, [model.proj, model.discriminatorA], criterion_recog,
								   optimizer_recog)
			losses_recog.update(loss, input_recog.size(0))
			top1_recog.update(acc, input_recog.size(0))

		# Train for privacy
		model.freeze_discriminatorA()
		model.freeze_discriminatorB()
		model.unfreeze_generator()
		loss, acc = train_step(input_recog, target_recog, [model.proj, model.discriminatorA], criterion_privacy,
							   optimizer_privacy)
		loss = loss
		losses_privacy.update(loss, input_recog.size(0))
		top1_privacy.update(acc, input_recog.size(0))

		# Train for detection
		model.freeze_discriminatorA()
		model.unfreeze_discriminatorB()
		model.unfreeze_generator()
		loss, acc = train_step(input_detect, target_detect, [model.proj, model.discriminatorB], criterion_detect,
							   optimizer_detect)
		losses_detect.update(loss, input_detect.size(0))
		top1_detect.update(acc, input_detect.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % opt.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Recog Loss {loss_recog.val:.4f} ({loss_recog.avg:.4f})\t'
				  'Recog Acc {top1_recog.val:.3f} ({top1_recog.avg:.3f})\t'
				  'Priv Loss {loss_privacy.val:.4f} ({loss_privacy.avg:.4f})\t'
				  'Priv Acc {top1_privacy.val:.3f} ({top1_privacy.avg:.3f})\t'
				  'Detect Loss {loss_detect.val:.4f} ({loss_detect.avg:.4f})\t'
				  'Detect Acc {top1_detect.val:.3f} ({top1_detect.avg:.3f})'.format(
				   epoch, i, len(loader_face), batch_time=batch_time, data_time=data_time, loss_recog=losses_recog,
				   top1_recog=top1_recog, loss_privacy=losses_privacy, top1_privacy=top1_privacy,
				   loss_detect=losses_detect, top1_detect=top1_detect))

	loss_dict = {'recog': losses_recog.avg, 'privacy': losses_privacy.avg, 'detect': losses_detect.avg}
	acc_dict = {'recog': top1_recog.avg, 'privacy': top1_privacy.avg, 'detect': top1_detect.avg}
	if retval == 'loss':
		return loss_dict
	elif retval == 'acc':
		return acc_dict
	elif retval == 'both':
		return [loss_dict, acc_dict]



def train_genloss(train_loader, model, criterion, criterionG, optimizerG, optimizer, epoch, opt, forward='', negative=False, retval='acc'):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	lossesA = AverageMeter()
	lossesB = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	# if we are given a list of train loaders, then we wish to simultaneously take inputs from both data loaders,
	#   each loader being one class
	if isinstance(train_loader, list):

		for i, data in enumerate(zip(train_loader[0], train_loader[1])):

			# measure data loading time
			data_time.update(time.time() - end)

			# these two lines are the only thing different for list of train_loaders... everything else is same
			input = torch.cat((data[0][0], data[1][0]))
			target = torch.Tensor(data[0][0].shape[0] * [0] + data[1][0].shape[0] * [1]).type(torch.int64)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			
			x = model.proj(input)
			outputA = model.discriminatorA(x)
			outputB = model.discriminatorB(x)

			lossB = criterion(outputB, target)
			lossA = criterionG(outputA, target)
			totloss = lossA + lossB



			# measure accuracy and record loss
			acc = accuracy(outputB, target)
			lossesB.update(lossB.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))
			lossesA.update(lossA.item(), input.size(0))
			# compute gradient and do SGD step
			optimizer.zero_grad()
			optimizerG.zero_grad()
			totloss.backward()
			optimizer.step()
			optimizerG.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Detection Loss {lossB.val:.4f} ({lossB.avg:.4f})\t'
					  'Gen Loss {lossA.val:.4f} ({lossA.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch, i, len(train_loader[0]), batch_time=batch_time,
					data_time=data_time, lossB=lossesB, lossA=lossesA, top1=top1))



	# else only single data loader
	else:

		for i, (input, target) in enumerate(train_loader):

			# measure data loading time
			data_time.update(time.time() - end)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			# compute output
			if forward == 'A':
				x = model.proj(input)
				output = model.discriminatorA(x)
			elif forward == 'B':
				x = model.proj(input)
				output = model.discriminatorB(x)
			else:
				output = model(input)
			loss = criterion(output, target)
			if negative:
				loss = -loss

			# measure accuracy and record loss
			acc = accuracy(output, target)
			losses.update(loss.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))

			# check for detection saying everything is one class
			if forward == 'B':
			   if (output.max(dim=1)[1].sum().item() == 0 and target.sum().item() > 5) \
					   or (output.max(dim=1)[1].sum().item() == 32 and target.sum().item() < 27):
				   pass
				   #import pdb; pdb.set_trace()
			   import math
			   if loss.item() == math.nan:
				   pass
				   #import pdb; pdb.set_trace()

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time,
					   data_time=data_time, loss=losses, top1=top1))
				#print('\n Output of network:', str(output))

	if retval == 'loss':
		return lossesB.avg
	elif retval == 'acc':
		return top1.avg
	elif retval == 'both':
		return top1.avg, lossesB.avg



def train_genloss_new(train_loader, model, criterion, criterionG, optimizerG, optimizer, epoch, opt, forward='', negative=False, retval='acc'):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	lossesA = AverageMeter()
	lossesB = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	# if we are given a list of train loaders, then we wish to simultaneously take inputs from both data loaders,
	#   each loader being one class
	if isinstance(train_loader, list):

		for i, data in enumerate(zip(train_loader[0], train_loader[1])):

			# measure data loading time
			data_time.update(time.time() - end)

			# these two lines are the only thing different for list of train_loaders... everything else is same
			input = torch.cat((data[0][0], data[1][0]))
			target = torch.Tensor(data[0][0].shape[0] * [0] + data[1][0].shape[0] * [1]).type(torch.int64)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			
			x = model.proj(input)
			outputA = model.discriminatorA(x)
			outputB = model.discriminatorB(x)

			lossB = criterion(outputB, target)
			lossA = criterionG(outputA, target)
			totloss = lossA + lossB



			# measure accuracy and record loss
			acc = accuracy(outputB, target)
			lossesB.update(lossB.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))
			lossesA.update(lossA.item(), input.size(0))
			# compute gradient and do SGD step
			optimizer.zero_grad()
			optimizerG.zero_grad()
			totloss.backward()
			optimizer.step()
			optimizerG.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Detection Loss {lossB.val:.4f} ({lossB.avg:.4f})\t'
					  'Gen Loss {lossA.val:.4f} ({lossA.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch, i, len(train_loader[0]), batch_time=batch_time,
					data_time=data_time, lossB=lossesB, lossA=lossesA, top1=top1))



	# else only single data loader
	else:

		for i, (input, target) in enumerate(train_loader):

			# measure data loading time
			data_time.update(time.time() - end)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			# compute output
			if forward == 'A':
				x = model.proj(input)
				output = model.discriminatorA(x)
			elif forward == 'B':
				x = model.proj(input)
				output = model.discriminatorB(x)
			else:
				output = model(input)
			loss = criterion(output, target)
			if negative:
				loss = -loss

			# measure accuracy and record loss
			acc = accuracy(output, target)
			losses.update(loss.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))

			# check for detection saying everything is one class
			if forward == 'B':
			   if (output.max(dim=1)[1].sum().item() == 0 and target.sum().item() > 5) \
					   or (output.max(dim=1)[1].sum().item() == 32 and target.sum().item() < 27):
				   pass
				   #import pdb; pdb.set_trace()
			   import math
			   if loss.item() == math.nan:
				   pass
				   #import pdb; pdb.set_trace()

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time,
					   data_time=data_time, loss=losses, top1=top1))
				#print('\n Output of network:', str(output))

	if retval == 'loss':
		return lossesB.avg
	elif retval == 'acc':
		return top1.avg
	elif retval == 'both':
		return top1.avg, lossesB.avg, lossesA.avg


def train(train_loader, model, criterion, optimizer, epoch, opt, forward='', negative=False, retval='acc'):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	# if we are given a list of train loaders, then we wish to simultaneously take inputs from both data loaders,
	#   each loader being one class
	if isinstance(train_loader, list):

		for i, data in enumerate(zip(train_loader[0], train_loader[1])):

			# measure data loading time
			data_time.update(time.time() - end)

			# these two lines are the only thing different for list of train_loaders... everything else is same
			input = torch.cat((data[0][0], data[1][0]))
			target = torch.Tensor(data[0][0].shape[0] * [0] + data[1][0].shape[0] * [1]).type(torch.int64)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			# compute output
			if forward == 'A':
				x = model.proj(input)
				output = model.discriminatorA(x)
			elif forward == 'B':
				x = model.proj(input)
				output = model.discriminatorB(x)
			else:
				output = model(input)
			# print(output.shape)
			# print(target.shape)
			# TODO: fix the following hack (googlenet has three branches and the loss is a weighted sum of the three)
			if forward == 'A' and model.discriminatorA_arch == 'Googlenet_64' and model.discriminatorA.aux_logits:
				loss = criterion(output[0], target) + 0.3 * (
							criterion(output[1], target) + criterion(output[2], target))
				output = output[0]  # for calculating accuracy
			elif forward == 'B' and model.discriminatorB_arch == 'Googlenet_64' and model.discriminatorB.aux_logits:
				loss = criterion(output[0], target) + 0.3 * (
							criterion(output[1], target) + criterion(output[2], target))
				output = output[0]  # for calculating accuracy
			else:
				loss = criterion(output, target)
			if negative:
				loss = -loss
			# measure accuracy and record loss
			acc = accuracy(output, target)
			losses.update(loss.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch, i, len(train_loader[0]), batch_time=batch_time,
					data_time=data_time, loss=losses, top1=top1))



	# else only single data loader
	else:

		for i, (input, target) in enumerate(train_loader):

			# measure data loading time
			data_time.update(time.time() - end)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			# compute output
			if forward == 'A':
				x = model.proj(input)
				output = model.discriminatorA(x)
			elif forward == 'B':
				x = model.proj(input)
				output = model.discriminatorB(x)
			else:
				output = model(input)
			# TODO: fix the following hack (googlenet has three branches and the loss is a weighted sum of the three)
			if forward == 'A' and model.discriminatorA_arch == 'Googlenet_64' and model.discriminatorA.aux_logits:
				loss = criterion(output[0], target) + 0.3 * (criterion(output[1], target) + criterion(output[2], target))
				output = output[0]  # for calculating accuracy
			elif forward == 'B' and model.discriminatorB_arch == 'Googlenet_64' and model.discriminatorB.aux_logits:
				loss = criterion(output[0], target) + 0.3 * (criterion(output[1], target) + criterion(output[2], target))
				output = output[0]  # for calculating accuracy
			else:
				loss = criterion(output, target)
			# a = list(model.proj.parameters())[0].clone()
			if negative:
				loss = -loss

			# measure accuracy and record loss
			acc = accuracy(output, target)
			losses.update(loss.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))

			# check for detection saying everything is one class
			if forward == 'B':
			   if (output.max(dim=1)[1].sum().item() == 0 and target.sum().item() > 5) \
					   or (output.max(dim=1)[1].sum().item() == 32 and target.sum().item() < 27):
				   pass
				   #import pdb; pdb.set_trace()
			   import math
			   if loss.item() == math.nan:
				   pass
				   #import pdb; pdb.set_trace()

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# b = list(model.proj.parameters())[0].clone()
			# print(torch.equal(a.data, b.data))
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time,
					   data_time=data_time, loss=losses, top1=top1))
				#print('\n Output of network:', str(output))

	if retval == 'loss':
		return losses.avg
	elif retval == 'acc':
		return top1.avg
	elif retval == 'both':
		return top1.avg, losses.avg


def TV_X(x):
	w_x = x.size()[1]
	# print(torch.pow((x[:, 1:] - x[:, :w_x - 1]), 2).sum())
	return torch.pow((x[:, 1:] - x[:, :w_x - 1]), 2).sum()


def train_onlygenerator(train_loader, model, criterion, criterionG, optimizerG, epoch, opt, retval='both', negative=False):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	lossesA = AverageMeter()
	lossesB = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	# if we are given a list of train loaders, then we wish to simultaneously take inputs from both data loaders,
	#   each loader being one class
	if isinstance(train_loader, list):

		for i, data in enumerate(zip(train_loader[0], train_loader[1])):

			# let train_loader[0] be the face identity data loader
			# let train_loader[1] be the face detection (both faces and no faces) data loader
			# imagesA = data[0][0]
			# data[0][0] should correspond to face identity images
			# data[0][1] should correspond to face identity labels
			# data[1][0] -- face detection images
			# data[1][1] -- face detection labels
			# xA = model.proj(data[0][0])
			# outputA = model.discriminatorA(xA)
			# lossA = ...(outputA)
			# xB = model.proj(data[1][0])
			# outputB = model.discriminatorB(xB)
			# lossB = ...(outputB, data[1][1])
			# tot_loss = lossA + lossB

			# measure data loading time
			data_time.update(time.time() - end)

			# these two lines are the only thing different for list of train_loaders... everything else is same
			input = torch.cat((data[0][0], data[1][0]))
			target = torch.Tensor(data[0][0].shape[0] * [0] + data[1][0].shape[0] * [1]).type(torch.int64)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			# compute output
			x = model.proj(input)
			outputA = model.discriminatorA(x)
			outputB = model.discriminatorB(x)
			a = list(model.proj.parameters())[0].clone()
			# print(output.shape)
			# print(target.shape)
			lossB = opt.bce_weight*criterion(outputB, target)
			# print(opt.entropy_weight)
			lossA = opt.entropy_weight*criterionG(outputA, target)
			# print(x.max())
			# print(lossB)
			if opt.bce_weight == 0:
				# print('no detection')
				tot_loss = lossA
			else:
				tot_loss = lossA + lossB
			# measure accuracy and record loss
			acc = accuracy(outputB, target)
			lossesB.update(lossB.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))
			lossesA.update(lossA.item(), input.size(0))

			# compute gradient and do SGD step
			optimizerG.zero_grad()
			tot_loss.backward()
			# print(model.proj.bias.grad)
			optimizerG.step()
			b = list(model.proj.parameters())[0].clone()

			# if a!=b:
			#     print('Generator Parameters are getting updated \n')
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Detection Loss {lossB.val:.4f} ({lossB.avg:.4f})\t'
					  'Gen Loss {lossA.val:.4f} ({lossA.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch, i, len(train_loader[0]), batch_time=batch_time,
					data_time=data_time, lossB=lossesB, lossA=lossesA, top1=top1))



	# else only single data loader
	else:

		for i, (input, target) in enumerate(train_loader):

			# measure data loading time
			data_time.update(time.time() - end)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			# compute output
			x= model.proj(input)
			outputA = model.discriminatorA(x)
			outputB = model.discriminatorB(x)
			a = list(model.proj.parameters())[0].clone()
			# print(output.shape)
			# print(target.shape)
			lossB = opt.bce_weight*criterion(outputB, target)
			# print(opt.entropy_weight)
			lossA = opt.entropy_weight*criterionG(outputA, target)
			if opt.bce_weight == 0:
				tot_loss = lossA
			else:
				tot_loss = lossA + lossB
			# measure accuracy and record loss
			acc = accuracy(outputB, target)
			lossesB.update(lossB.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))
			lossesA.update(lossA.item(), input.size(0))

			# compute gradient and do SGD step
			optimizerG.zero_grad()
			tot_loss.backward()
			optimizerG.step()
			b = list(model.proj.parameters())[0].clone()
			print(a==b)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Detection Loss {lossB.val:.4f} ({lossB.avg:.4f})\t'
					  'Gen Loss {lossA.val:.4f} ({lossA.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch, i, len(train_loader[0]), batch_time=batch_time,
					data_time=data_time, lossB=lossesB, lossA=lossesA, top1=top1))
				#print('\n Output of network:', str(output))

	if retval == 'loss':
		return lossesB.avg
	elif retval == 'acc':
		return top1.avg
	elif retval == 'both':
		return top1.avg, lossesB.avg, lossesA.avg

def train_onlygenerator_sepdata(train_loader, model, criterion, criterionG, optimizerG, epoch, opt, retval='both', negative=False):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	lossesA = AverageMeter()
	lossesB = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()
	end = time.time()

	# if we are given a list of train loaders, then we wish to simultaneously take inputs from both data loaders,
	#   each loader being one class
	for i, data in enumerate(zip(train_loader[0], train_loader[1])):

		# measure data loading time
		data_time.update(time.time() - end)

		# these two lines are the only thing different for list of train_loaders... everything else is same
		inputA = data[0][0]
		# print(inputA.shape)
		targetA = data[0][1]
		# print(targetA)
		inputB = data[1][0]
		# print(inputB.shape)
		targetB = data[1][1]
		# print(targetB.shape)
		inputA = inputA.cuda(opt.gpu_idx)
		targetA = targetA.cuda(opt.gpu_idx)
		inputB = inputB.cuda(opt.gpu_idx)
		targetB = targetB.cuda(opt.gpu_idx)

		# compute output
		xA = model.proj(inputA)
		outputA = model.discriminatorA(xA)
		xB = model.proj(inputB)
		outputB = model.discriminatorB(xB)
		lossB = opt.bce_weight*criterion(outputB, targetB)
		# print(opt.entropy_weight)
		lossA = opt.entropy_weight*criterionG(outputA, targetA)
		# print(x.max())
		# print(lossB)
		if opt.bce_weight == 0:
			# print('no detection')
			tot_loss = lossA
		else:
			tot_loss = lossA + lossB
		# measure accuracy and record loss
		acc = accuracy(outputB, targetB)
		lossesB.update(lossB.item(), inputB.size(0))
		top1.update(acc[0].item(), inputB.size(0))
		lossesA.update(lossA.item(), inputA.size(0))

		# compute gradient and do SGD step
		optimizerG.zero_grad()
		tot_loss.backward()
		# print(model.proj.bias.grad)
		optimizerG.step()
		# if a!=b:
		#     print('Generator Parameters are getting updated \n')
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % opt.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Detection Loss {lossB.val:.4f} ({lossB.avg:.4f})\t'
				  'Gen Loss {lossA.val:.4f} ({lossA.avg:.4f})\t'
				  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
				epoch, i, len(train_loader[0]), batch_time=batch_time,
				data_time=data_time, lossB=lossesB, lossA=lossesA, top1=top1))

	if retval == 'loss':
		return lossesB.avg
	elif retval == 'acc':
		return top1.avg
	elif retval == 'both':
		return top1.avg, lossesB.avg, lossesA.avg


def train_onlygenerator_multiop(train_loader, model, criterion, criterionG, optimizerG, epoch, opt, retval='both', negative=False):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	lossesA = AverageMeter()
	lossesB = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	# if we are given a list of train loaders, then we wish to simultaneously take inputs from both data loaders,
	#   each loader being one class
	if isinstance(train_loader, list):

		for i, data in enumerate(zip(train_loader[0], train_loader[1])):

			# measure data loading time
			data_time.update(time.time() - end)

			# these two lines are the only thing different for list of train_loaders... everything else is same
			input = torch.cat((data[0][0], data[1][0]))
			target = torch.Tensor(data[0][0].shape[0] * [0] + data[1][0].shape[0] * [1]).type(torch.int64)
			# print(input.shape)
			# print(target.shape)
			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			# compute output
			x = model.proj(input)
			outputA = model.discriminatorA(x)
			outputB = model.discriminatorB(x)
			# a = list(model.proj.parameters())[0].clone()
			# print(output.shape)
			# print(target.shape)
			lossB = opt.bce_weight*criterion(outputB, target)
			# print(opt.entropy_weight)
			lossA = opt.entropy_weight*criterionG(outputA, target)
			tv_loss = None
			l1_loss = None
			h = 0
			for W in model.proj.parameters():
				if h == 0:
					# w_x = W.size()[1]
					tv_loss = opt.tv_reg*(W[:, 1:] - W[:, :W.shape[1] - 1]).pow(2).sum().div(W.shape[0]*W.shape[1])
					l1_loss = opt.l1_reg*W.norm(1)
					# print(tv_loss)
					h = h+1
			tot_loss = lossA + lossB + l1_loss + tv_loss
			
			# print(tv_loss)
			# measure accuracy and record loss
			acc = accuracy(outputB, target)
			lossesB.update(lossB.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))
			lossesA.update(lossA.item(), input.size(0))

			# compute gradient and do SGD step
			optimizerG.zero_grad()
			tot_loss.backward()
			optimizerG.step()
			# b = list(model.proj.parameters())[0].clone()
			# print(a)
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Detection Loss {lossB.val:.4f} ({lossB.avg:.4f})\t'
					  'Gen Loss {lossA.val:.4f} ({lossA.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch, i, len(train_loader[0]), batch_time=batch_time,
					data_time=data_time, lossB=lossesB, lossA=lossesA, top1=top1))



	# else only single data loader
	else:

		for i, (input, target) in enumerate(train_loader):

			# measure data loading time
			data_time.update(time.time() - end)

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)

			# compute output
			x = model.proj(input)
			outputA = model.discriminatorA(x)
			outputB = model.discriminatorB(x)
			a = list(model.proj.parameters())[0].clone()
			# print(output.shape)
			# print(target.shape)
			lossB = opt.bce_weight*criterion(outputB, target)
			# print(opt.entropy_weight)
			lossA = opt.entropy_weight*criterionG(outputA, target)
			tv_loss = None
			l1_loss = None
			for W in model.proj.parameters():
				tv_loss = opt.tv_reg*TV_X(W)
				l1_loss = opt.l1_reg*W.norm(1)
			tot_loss = lossA + lossB + tv_loss + l1_loss
			# measure accuracy and record loss
			acc = accuracy(outputB, target)
			lossesB.update(lossB.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))
			lossesA.update(lossA.item(), input.size(0))

			# compute gradient and do SGD step
			optimizerG.zero_grad()
			tot_loss.backward()
			optimizerG.step()
			b = list(model.proj.parameters())[0].clone()
			print(a)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % opt.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Detection Loss {lossB.val:.4f} ({lossB.avg:.4f})\t'
					  'Gen Loss {lossA.val:.4f} ({lossA.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch, i, len(train_loader[0]), batch_time=batch_time,
					data_time=data_time, lossB=lossesB, lossA=lossesA, top1=top1))
				#print('\n Output of network:', str(output))

	if retval == 'loss':
		return lossesB.avg
	elif retval == 'acc':
		return top1.avg
	elif retval == 'both':
		return top1.avg, lossesB.avg, lossesA.avg


def validate(val_loader, model, criterion, opt, forward='', retval='acc'):

	# confusion_mat = np.zeros((opt.num_classes, opt.num_classes))

	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	top10 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(val_loader):

			input = input.cuda(opt.gpu_idx, non_blocking=True)
			target = target.cuda(opt.gpu_idx, non_blocking=True)
			# print(target.shape)
			# compute output
			if forward == 'A':
				x = model.proj(input)
				output = model.discriminatorA(x)
			elif forward == 'B':
				x = model.proj(input)
				output = model.discriminatorB(x)
			else:
				output = model(input)
			# print(output.shape)
			loss = criterion(output, target)

			# measure accuracy and record loss
			if retval == 'acc_top1_top5_top10':
				acc, acc5, acc10 = accuracy(output, target, topk=(1,5,10))
				top5.update(acc5[0].item(), input.size(0))
				top10.update(acc10[0].item(), input.size(0))
			else:
				acc = accuracy(output, target)
			top1.update(acc[0].item(), input.size(0))
			losses.update(loss.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			# update confusion matrix (can't just do confusion_mat[targets, pred_outs] += 1 since the operation doesn't
			# accumulate for duplicate target-pred_out pairs)
			# _, pred_outs = output.max(1)
			# pred_outs = pred_outs.cpu().numpy()
			# targets_numpy = target.cpu().numpy()
			# for i in range(len(pred_outs)):
			#     confusion_mat[targets_numpy[i], pred_outs[i]] += 1

			if i % opt.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					   i, len(val_loader), batch_time=batch_time, loss=losses,
					   top1=top1))
				#print('\n Output of network:', str(output))

		print(' * Acc@1 {top1.avg:.3f} '
			  .format(top1=top1))

	if retval == 'loss':
		return losses.avg
	elif retval == 'acc':
		return top1.avg
	elif retval == 'both':
		return top1.avg, losses.avg
	elif retval == 'acc_top1_top5_top10':
		return top1.avg, top5.avg, top10.avg


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def decrease_learning_rate(lr, optimizer, degree):
	curr_lr = lr * (0.1 ** degree)
	optimizer.param_groups[0]['lr'] = curr_lr  # only change the first param_group (for optimizerB, don't include gen)
	return curr_lr


## Deprecated functions

def train_fix_epochs(train_loader, train_params, opt, forward='', num_lrs=3, thresh=0, max_epochs=1000, val_loader=None):
	"""This function is train_with_decreasing_lr without decreasing the learning rate

	Args:
		train_loader:
		train_params:
		opt:
		forward:
		num_lrs:
		thresh:
		max_epochs:
		val_loader:

	Returns:
		float: Training classification accuracy
		float: Training loss value
		float: Validation classification accuracy
		float: Validation loss value

	"""

	# train_with_decreasing_lr does not decrease lr until wait_epochs number of epochs
	# setting wait_epochs to max_epochs + 1 means lr will never be decreased
	print('train_fix_epochs is deprecated. Please run train_with_decreasing_lr with decrease_lr=False')
	tr_acc, tr_loss, val_acc, val_loss = train_with_decreasing_lr(train_loader, train_params, opt, forward=forward,
																  num_lrs=1, thresh=thresh, max_epochs=max_epochs,
																  val_loader=val_loader, decrease_lr=False)
	return tr_acc, tr_loss, val_acc, val_loss

def train_with_decreasing_lr_multiop(train_loader, train_params, opt, forward='', num_lrs=3, thresh=0, max_epochs=1000, val_loader=None):
	raise NameError('train_with_decreasing_lr_multiop() is depracated. Pls switch to train_with_decreasing_lr() -- it is exactly the same.')


def validate_multiop(val_loader, model, criterion, opt, forward='', retval='acc'):
	raise NameError('validate_multiop() is depracated. Pls switch to validate() -- it is exactly the same.')


def train_multiop(train_loader, model, criterion, optimizer, epoch, opt, forward='', negative=False, retval='acc'):
	raise NameError('train_multiop() is depracated. Pls switch to train() -- it is exactly the same.')


def train_fix_epochs_multiop(train_loader, train_params, opt, forward='', num_lrs=3, thresh=0, max_epochs=1000, val_loader=None):
	raise NameError('train_fix_epochs_multiop() is depracated. Pls switch to train_fix_epochs() -- it is exactly the same.')
