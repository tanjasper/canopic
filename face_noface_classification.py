"""General script to perform training and testing of discriminators through encoders

Steps:
(1) Set up parameters, GPUs, directories, seed, log
(2) Prepare data loaders
(3) Create EncoderDiscriminatorNet model
(4) Prepare criterion and optimizers
(5) Train
(6) Test
(7) Save results"""

import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import random
import torch.backends.cudnn as cudnn
from models.multi_net import model_from_checkpoint, EncoderDiscriminatorNet
from fns_all import *
import os
import data_fns
import warnings
import sys
import pprint
from datetime import datetime
from pytz import timezone
from settings.util import load_encoder_settings, load_discriminator_settings
from utils import save_accuracy_plot

# some parameters
parser = argparse.ArgumentParser()
# where to save results:
parser.add_argument('--save_dir', help='directory where results will be saved')
parser.add_argument('--save_name_model', default='face_noface_classification_model.tar', help='filename where model will be saved (must end in .tar)')
parser.add_argument('--save_name_accuracies', default='face_noface_classification_accuracies.txt')
parser.add_argument('--save_name_plot', default='face_noface_classification_plots')
# data:
parser.add_argument('--im_dim', default=74, type=int)
parser.add_argument('--tr_filenames_face', type=str)
parser.add_argument('--val_filenames_face', type=str)
parser.add_argument('--test_filenames_face', type=str)
parser.add_argument('--tr_filenames_noface', type=str)
parser.add_argument('--val_filenames_noface', type=str)
parser.add_argument('--test_filenames_noface', type=str)
parser.add_argument('--imdir_face', default='data/vggface2')
parser.add_argument('--imdir_noface', default='data/ILSVRC2012/images')
parser.add_argument('--root_data_dir', default='data/results')
# training:
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--max_epochs', default=1000, type=int)
parser.add_argument('--wait_epochs', default=10, type=int)
parser.add_argument('--num_lrs', default=3, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--type_optimizer', default='Adam')
# loading information:
parser.add_argument('--encoder_checkpoint_path', default='', help='checkpoint where encoder will be loaded from')
parser.add_argument('--encoder_settings_json', type=str)
parser.add_argument('--discriminator_settings_json', type=str)
parser.add_argument('--checkpoint_path', default='')
# other stuff:
parser.add_argument('--num_threads', default=4, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--gpu_idx', default=0, type=int)
parser.add_argument('--print_freq', default=250, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--seed', default=1313, type=int)

parser.set_defaults(learn_noise=False, checkpoint_modelonly=False)

def main():

    # (1) Set up:
    # call up parameters
    opt = parser.parse_args()
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
    save_dir = os.path.join(opt.root_data_dir, opt.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # let every print go to both stdout and to a logfile
    sys.stdout = Logger(save_dir, 'face_noface_classification_')
    # print out setup info:
    print('======== Log ========')
    print(datetime.now(timezone('EST')))
    print('\n')
    print("Command ran:\n%s\n\n" % " ".join([x for x in sys.argv]))
    print("Opt:")
    pprint.pprint(vars(opt))
    print("\n")

    # (2) Prepare data
    train_loader, val_loader = prepare_data(opt)
    # Load training checkpoint path for train_with_decreasing_lr
    if opt.checkpoint_path:
        if opt.checkpoint_path == 'latest':
            load_checkpoint_path = os.path.join(save_dir, 'train_latest_checkpoint.tar')
        else:
            load_checkpoint_path = os.path.join(opt.root_data_dir, opt.checkpoint_path)
    else:
        load_checkpoint_path = None

    # (3) Create EncoderDiscriminatorNet model
    # copy settings json files over to the save_dir to keep record
    shutil.copyfile(opt.discriminator_settings_json, os.path.join(save_dir, 'discriminator_settings.json'))
    if opt.encoder_settings_json:
        shutil.copyfile(opt.encoder_settings_json, os.path.join(save_dir, 'encoder_settings.json'))
    discriminator_arch, discriminator_settings = load_discriminator_settings(opt.discriminator_settings_json)
    if opt.encoder_checkpoint_path:  # load encoder from provided checkpoint
        model = EncoderDiscriminatorNet(opt, None, discriminator_arch, None,
                                        discriminatorB_settings=discriminator_settings)
        model.encoder_from_checkpoint(torch.load(os.path.join(opt.root_data_dir, opt.encoder_checkpoint_path)))
        if opt.encoder_settings_json:  # add encoders if user gave more
            encoder_archs, encoder_settings = load_encoder_settings(opt.encoder_settings_json)
            model.add_encoders(opt, encoder_archs, encoder_settings)
    else:  # create new encoder
        encoder_archs, encoder_settings = load_encoder_settings(opt.encoder_settings_json)
        model = EncoderDiscriminatorNet(opt, None, discriminator_arch, encoder_archs,
                                        discriminatorB_settings=discriminator_settings,
                                        encoder_settings=encoder_settings)
    model = model.cuda(opt.gpu_idx)

    # (4) Criterion and optimizers
    criterion = nn.CrossEntropyLoss().cuda(opt.gpu_idx)
    if opt.type_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.discriminatorB.parameters(), opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.discriminatorB.parameters(), opt.lr, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    # (5) Train
    model.freeze_generator()
    model.unfreeze_discriminatorB()
    train_params = {
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'epoch': 0,
        'lr': opt.lr,
        'wait_epochs': opt.wait_epochs
    }
    tr_accs, tr_losses, val_accs, val_losses = train_with_decreasing_lr(train_loader, train_params, opt, forward='B',
                                                                        max_epochs=opt.max_epochs,
                                                                        num_lrs=opt.num_lrs, val_loader=val_loader,
                                                                        best_val_model=True, return_multi_accs=True,
                                                                        decrease_lr=True, save_checkpoint_path=save_dir,
                                                                        save_checkpoint_freq=1,
                                                                        load_checkpoint_path=load_checkpoint_path)

    # TODO: this is temporary -- in case there is a bug with testing
    print('Saving checkpoint')
    save_dict = {
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': model.discriminatorB.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'opt': opt,
        'encoder_archs': model.encoder_archs,
        'encoder_settings': model.encoder_settings,
        'discriminator_arch': model.discriminatorB_arch,
        'discriminator_settings': model.discriminatorB_settings,
        'tr_accs': tr_accs,
        'val_accs': val_accs
    }
    torch.save(save_dict, os.path.join(save_dir, 'id_classification_post_train_checkpoint.tar'))

    # (6) Test
    test_acc = validate(val_loader, model, criterion, opt, forward='B', retval='acc')

    # (7) Save results
    print('Saving final model')
    save_dict = {
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': model.discriminatorB.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'opt': opt,
        'encoder_archs': model.encoder_archs,
        'encoder_settings': model.encoder_settings,
        'discriminator_arch': model.discriminatorB_arch,
        'discriminator_settings': model.discriminatorB_settings,
        'tr_accs': tr_accs,
        'val_accs': val_accs,
        'test_acc': test_acc
    }
    torch.save(save_dict, os.path.join(save_dir, opt.save_name_model))
    print('Done saving final model')
    best_val_epoch = val_accs.index(max(val_accs))
    with open(os.path.join(save_dir, opt.save_name_accuracies), 'w') as fp:
        fp.write('Training data (face): ' + opt.tr_filenames_face + '\n')
        fp.write('Validation data (face): ' + opt.val_filenames_face + '\n')
        fp.write('Test data (face): ' + opt.test_filenames_face + '\n')
        fp.write('Training data (no face): ' + opt.tr_filenames_noface + '\n')
        fp.write('Validation data (no face): ' + opt.val_filenames_noface + '\n')
        fp.write('Test data (no face): ' + opt.test_filenames_noface + '\n\n')
        fp.write('Training accuracy: %f\n' % tr_accs[best_val_epoch])
        fp.write('Validation accuracy: %f\n' % val_accs[best_val_epoch])
        fp.write('Test accuracy: %f\n' % test_acc)
    save_accuracy_plot([[x/100 for x in tr_accs], [x/100 for x in val_accs]], os.path.join(save_dir, opt.save_name_plot), ['Training', 'Val'],
                       ylim=(0, 1))

    print('Finished identity classification')


def prepare_data(opt):
    """Returns data loaders for training, validation, and test"""

    im_dim = opt.im_dim
    # (1) define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(im_dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(round(im_dim / 0.875)),  # remove some background
        transforms.CenterCrop(im_dim),
        transforms.ToTensor(),
    ])

    # (2) prepare training and val data loader
    # prepare datasets
    train_dataset = data_fns.DatasetFromMultipleFilenames([opt.imdir_face, opt.imdir_noface],
                                                          [opt.tr_filenames_face, opt.tr_filenames_noface],
                                                          train_transforms)
    val_dataset = data_fns.DatasetFromMultipleFilenames([opt.imdir_face, opt.imdir_noface],
                                                [opt.val_filenames_face, opt.val_filenames_noface],
                                                test_transforms)
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads,
                                               shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.num_threads, pin_memory=True)
    return train_loader, val_loader


class Logger(object):
    def __init__(self, save_dir, savename):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, savename + "log.txt"), "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Required for Python 3 compatibility"""
        pass


if __name__ == '__main__':
    main()
