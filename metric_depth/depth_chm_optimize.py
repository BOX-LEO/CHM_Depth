import argparse
import logging
import os
import pprint
import random
import OpenEXR
import Imath
import cv2
import rasterio
import optuna

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from util.utils import init_log

from tqdm import tqdm


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.00001, type=float)
parser.add_argument('--max-depth', default=40, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--bs', default=8, type=int)
parser.add_argument('--trainable', default='full', choices=['full', 'head'])
parser.add_argument('--gt', default='pseudo', choices=['pseudo', 'chm'])
parser.add_argument('--loss', default='l1', choices=['l1', 'silog'])
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', default = '/home/boxiang/work/dao2/segment/Depth-Anything-V2/checkpoints', type=str, required=False)


class DepthDataset(Dataset):
    def __init__(self, image_files, depth_files, input_size=(384,768),max_depth=40):
        self.image_files = image_files
        self.image_files.sort()
        self.depth_files = depth_files
        self.depth_files.sort()
        self.input_size = input_size
        self.max_depth = max_depth
        assert len(image_files) == len(depth_files), f"The number of image files {len(image_files)} and depth files {len(depth_files)} must be the same"
        # interppolate the chm files with scale in transform

        
    def __len__(self):
        return len(self.image_files)

    def image2tensor(self, raw_image):        
        transform = Compose([
            Resize(
                height=self.input_size[0],
                width=self.input_size[1],
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        
        return image, (h, w)

    def read_exr_depth(self,file_path, channel='Y'):
        exr_file = OpenEXR.InputFile(file_path)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Read the Z (depth) channel
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = exr_file.channel(channel, pt)
        depth = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
        return depth

    def read_tif_height(self, file_path):
        chm = rasterio.open(file_path)
        chm = chm.read(1)
        chm = chm.astype(np.float32)
        # flip vertically
        chm = np.flipud(chm)
        return chm
 
    def __getitem__(self, idx):
        # read the image
        if self.image_files[idx].endswith('.png'):
            image = cv2.imread(self.image_files[idx], cv2.IMREAD_UNCHANGED)
        else:
            print('image file:', self.image_files[idx])
            raise ValueError('The image should be in png format')
        
        image, (h,w) = self.image2tensor(image)
        
        # read the depth map
        if self.depth_files[idx].endswith('.exr'):
            depth = self.read_exr_depth(self.depth_files[idx])
        elif self.depth_files[idx].endswith('.tif'):
            height = self.read_tif_height(self.depth_files[idx])
            height = height - height.min()
            depth = self.max_depth - height
        elif self.depth_files[idx].endswith('.npy'):
            height = np.load(self.depth_files[idx])
            depth = self.max_depth - height

        else:
            raise ValueError('The depth should be in exr or tif format instead of {}'.format(self.depth_files[idx].split('.')[-1]))
        
        image = image.squeeze(0)
        return image, depth

def main():
    torch.cuda.empty_cache()
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    # rank, world_size = setup_distributed(port=args.port)
    rank = 0
    world_size = 1

    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        # writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    
    # Load dataset based on GT type
    if args.gt == 'chm':
        # Use CHM (Canopy Height Model) data
        train_gt_folders = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/trainset/chm'
        train_image_folders = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/trainset/image'
        depth_gt_files = [os.path.join(train_gt_folders, f) for f in os.listdir(train_gt_folders)\
                                                            if f.endswith('.tif') or f.endswith('.exr') or f.endswith('.npy')]
        image_files = [os.path.join(train_image_folders, f) for f in os.listdir(train_image_folders) if f.endswith('.png')]
    elif args.gt == 'pseudo':
        # Use pseudo ground truth data (you may need to adjust paths)
        train_gt_folders = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/trainset/pseudo_gt'  # Adjust path as needed
        train_image_folders = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/trainset/image'
        depth_gt_files = [os.path.join(train_gt_folders, f) for f in os.listdir(train_gt_folders)\
                                                            if f.endswith('.tif') or f.endswith('.exr') or f.endswith('.npy')]
        image_files = [os.path.join(train_image_folders, f) for f in os.listdir(train_image_folders) if f.endswith('.png')]
    else:
        raise ValueError(f'GT type {args.gt} not supported')
    
    # sort the files
    depth_gt_files.sort()
    image_files.sort()
    assert len(image_files) == len(depth_gt_files), f"The number of image files {len(image_files)} and depth files {len(depth_gt_files)} must be the same"

    # split the dataset into train and validation sets
    num_samples = len(image_files)
    indices = list(range(num_samples))
    random.seed(42)
    random.shuffle(indices)
    split = int(np.floor(0.9 * num_samples))
    train_indices, val_indices = indices[:split], indices[split:]
    train_image_files = [image_files[i] for i in train_indices]
    train_depth_gt_files = [depth_gt_files[i] for i in train_indices]
    val_image_files = [image_files[i] for i in val_indices]
    val_depth_gt_files = [depth_gt_files[i] for i in val_indices]

    trainset = DepthDataset(train_image_files, train_depth_gt_files, input_size=size)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4,shuffle=True)
    
    valset = DepthDataset(val_image_files, val_depth_gt_files, input_size=size)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4)
    
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    def objective(trail):
        lr = trail.suggest_float('lr', 1e-8, 1e-4, log=True)
        epochs = trail.suggest_int('epochs', 10, 100, step=10)

        model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
        
        if args.pretrained_from:
            model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()

        # Set trainable parameters based on ablation experiment
        if args.trainable == 'head':
            # Only train the head (non-pretrained parameters)
            for name, param in model.named_parameters():
                if 'pretrained' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif args.trainable == 'full':
            # Train all parameters
            for name, param in model.named_parameters():
                param.requires_grad = True
    
        criterion = SiLogLoss()
        
        # Set up optimizer with different learning rates based on trainable components
        if args.trainable == 'head':
            # Only optimize non-pretrained parameters
            optimizer = AdamW([
                {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': lr * 10.0}
            ], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        else:  # full
            # Optimize all parameters with different learning rates
            optimizer = AdamW([
                {'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': lr},
                {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': lr * 10.0}
            ], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        
        total_iters = epochs * len(trainloader)
        
        # train the model
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i, (img,depth) in enumerate(trainloader):
                optimizer.zero_grad()
                img = img.cuda().float()
                depth = depth.cuda().float()
                if torch.isnan(img).any().item() or torch.isinf(img).any().item():
                    print('img contains NaN or Inf')
                    return float('inf')
                if not depth.min() > 0:
                    print(f'depth min is {depth.min()} is not greater than 0')
                    return float('inf')
                valid_mask = torch.ones_like(depth).cuda()
                if random.random() < 0.5:
                    img = img.flip(-1)
                    depth = depth.flip(-1)
                    valid_mask = valid_mask.flip(-1)

                pred = model(img)
                pred = F.interpolate(pred[:,None], size=depth.shape[-2:], mode='bilinear', align_corners=True)
                pred.squeeze_(1)
                if torch.isnan(pred).any().item() or torch.isinf(pred).any().item():
                    print('pred contains NaN or Inf')
                    return float('inf')

                if args.loss == 'l1':
                    loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)) \
                    + 0.1 * torch.nn.functional.l1_loss(pred, depth)
                elif args.loss == 'silog':
                    loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
                else:
                    raise ValueError(f'Loss {args.loss} not supported')
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                iters = epoch * len(trainloader) + i
                
                lr_ = lr * (1 - iters / total_iters) ** 0.9
                
                # Update learning rates based on trainable components
                if args.trainable == 'head':
                    optimizer.param_groups[0]["lr"] = lr_ * 10.0
                else:  # full
                    optimizer.param_groups[0]["lr"] = lr_
                    optimizer.param_groups[1]["lr"] = lr_ * 10.0

        # evaluate the model    
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (img, depth) in enumerate(valloader):
                img = img.cuda().float()
                depth = depth.cuda().float()
                if torch.isnan(img).any().item() or torch.isinf(img).any().item():
                    print('img contains NaN or Inf')
                    return float('inf')
                if not depth.min() > 0:
                    print('depth min is not greater than 0')
                    return float('inf')
                valid_mask = torch.ones_like(depth).cuda()
                if random.random() < 0.5:
                    img = img.flip(-1)
                    depth = depth.flip(-1)
                    valid_mask = valid_mask.flip(-1)

                pred = model(img)
                pred = F.interpolate(pred[:,None], size=depth.shape[-2:], mode='bilinear', align_corners=True)
                # print(pred.shape)
                # print(depth.shape)
                if torch.isnan(pred).any().item() or torch.isinf(pred).any().item():
                    print('pred contains NaN or Inf')
                    return float('inf')

                pred.squeeze_(1)
                if args.loss == 'l1':
                    cur_loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)) \
                    + 0.1 * torch.nn.functional.l1_loss(pred, depth)
                elif args.loss == 'silog':
                    cur_loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
                else:
                    raise ValueError(f'Loss {args.loss} not supported')
                val_loss += cur_loss.item()
            val_loss /= len(valloader)
            print(f'val_loss: {val_loss}')
            return val_loss
        
                
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, n_jobs=1)
    print(f'Best trial: {study.best_trial.number}')
    print(f'Best value: {study.best_trial.value}')
    print(f'Best params: {study.best_trial.params}')
    best_params = study.best_trial.params
    lr = best_params['lr']
    epochs = best_params['epochs']
    # lr = 1.6440558364300636e-05
    # epochs = 100
    del trainloader, valloader, trainset, valset
    # train the model with the best parameters
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    if args.pretrained_from:
        model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    
    # Set trainable parameters based on ablation experiment
    if args.trainable == 'head':
        # Only train the head (non-pretrained parameters)
        for name, param in model.named_parameters():
            if 'pretrained' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif args.trainable == 'full':
        # Train all parameters
        for name, param in model.named_parameters():
            param.requires_grad = True
    
    dataset = DepthDataset( image_files, depth_gt_files, input_size=size)
    dataloader = DataLoader(dataset, batch_size=args.bs, pin_memory=True, num_workers=4,shuffle=True)

    # Define loss function based on ablation experiment
    criterion = SiLogLoss()
    
    # Set up optimizer with different learning rates based on trainable components
    if args.trainable == 'head':
        # Only optimize non-pretrained parameters
        optimizer = AdamW([
            {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': lr * 10.0}
        ], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    else:  # full
        # Optimize all parameters with different learning rates
        optimizer = AdamW([
            {'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': lr},
            {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': lr * 10.0}
        ], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = epochs * len(dataloader)
    pbar = tqdm(total=total_iters, desc='Training', disable=(rank != 0))
    pbar.set_postfix({'epoch': 0, 'loss': 0.0, 'lr': lr})
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, (img,depth) in enumerate(dataloader):
            optimizer.zero_grad()
            
            img = img.cuda().float()
            depth = depth.cuda().float()
            assert not torch.isnan(img).any().item(), 'img contains NaN'
            assert not torch.isinf(img).any().item(), 'img contains Inf'
            assert depth.min() > 0, 'depth min is not greater than 0'
            valid_mask = torch.ones_like(depth).cuda()
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)
            pred = F.interpolate(pred[:,None], size=depth.shape[-2:], mode='bilinear', align_corners=True)

            assert not torch.isnan(pred).any().item(), 'pred contains NaN'
            assert not torch.isinf(pred).any().item(), 'pred contains Inf'

            pred.squeeze_(1)

            
            if args.loss == 'l1':
                loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)) \
                + 0.1 * torch.nn.functional.l1_loss(pred, depth)
            elif args.loss == 'silog':
                loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            else:
                raise ValueError(f'Loss {args.loss} not supported')

            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(dataloader) + i
            
            lr_ = lr * (1 - iters / total_iters) ** 0.9
            
            # Update learning rates based on trainable components
            if args.trainable == 'head':
                optimizer.param_groups[0]["lr"] = lr_ * 10.0
            else:  # full
                optimizer.param_groups[0]["lr"] = lr_
                optimizer.param_groups[1]["lr"] = lr_ * 10.0
                
            pbar.set_postfix({'epoch': epoch + 1, 'loss': total_loss / (i + 1), 'lr': lr_})
            pbar.update(1)

    # Generate checkpoint filename based on ablation experiment configuration
    checkpoint_name = f'martell_{args.gt}_{args.trainable}_{args.loss}_ablation.pth'
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': vars(args),  # Save experiment configuration
    }
    torch.save(checkpoint, os.path.join(args.save_path, checkpoint_name))


if __name__ == '__main__':
    main()