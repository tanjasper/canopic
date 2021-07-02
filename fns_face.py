import torchvision.transforms as transforms
import data_fns
import torch

def prepare_face_data(opt):
    # (1) define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # (2) define datasets
    train_dataset = data_fns.DatasetFromMultipleFilenames(
        [opt.face_dir, opt.noface_dir], [opt.tr_face_filenames, opt.tr_noface_filenames], train_transforms)
    train_face_dataset = data_fns.DatasetFromFilenames(opt.face_dir, opt.tr_face_filenames, train_transforms)
    val_dataset = data_fns.DatasetFromMultipleFilenames(
        [opt.face_dir, opt.noface_dir], [opt.val_face_filenames, opt.val_noface_filenames], val_transforms)
    val_face_dataset = data_fns.DatasetFromFilenames(opt.face_dir, opt.val_face_filenames, val_transforms)
    # data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
    train_face_loader = torch.utils.data.DataLoader(
        train_face_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads, pin_memory=True)
    val_face_loader = torch.utils.data.DataLoader(
        val_face_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads, pin_memory=True)
    return train_loader, train_face_loader, val_loader, val_face_loader