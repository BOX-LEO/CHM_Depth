import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2 # install depth-anything-v2 first
import argparse
import os
import numpy as np
from PIL import Image
import OpenEXR
import Imath
import rasterio

# conda activate digitalf

def read_exr_depth(file_path, channel='Y'):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the Z (depth) channel
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    # print(exr_file.header())
    raw = exr_file.channel(channel, pt)
    depth = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
    return depth
def read_tif_height(file_path):
    chm = rasterio.open(file_path)
    chm = chm.read(1)
    chm = chm.astype(np.float32)
    # flip vertically
    chm = np.flipud(chm)
    return chm
parser = argparse.ArgumentParser(description='Depth Anything V2')
parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--max-depth', type=int, default=40, help='Maximum depth for the model. Default is 40m.')
parser.add_argument('--img-path', type=str, default='/home/boxiang/work/dao2/USGS_LiDAR/martell_data/testset/image')
parser.add_argument('--outdir', type=str, default='/home/boxiang/work/dao2/USGS_LiDAR/martell_data/abalation/test_set_result/chm_full')
parser.add_argument('--input-size', type=int, default=500)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

image_path = args.img_path
out_dir = args.outdir
os.makedirs(out_dir, exist_ok=True)

metric_depth_files = os.listdir(out_dir)
if len(metric_depth_files) > 0:
    print(f'Found {len(metric_depth_files)} metric depth files in {out_dir}.')
else:
    print('starting to generate metric depth files...')
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')

    # Load the model
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    model.load_state_dict(torch.load('/home/boxiang/work/dao2/segment/Depth-Anything-V2/checkpoints/martell_chm_full_l1_ablation.pth', map_location='cpu')['model'])
    model = model.to(DEVICE).eval()

    # get predicted depth map

    # pseudo_gt_files = os.listdir(pseudo_gt_path)
    # image_files = [os.path.join(image_path, f.replace('.npy', '.png')) for f in pseudo_gt_files]
    image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png')]
    image_files.sort()
    for f in image_files:
        img = cv2.imread(f)
        depth = model.infer_image(img)
        filename = os.path.basename(f).replace('.png', '.npy')
        out_path = os.path.join(out_dir, filename)
        np.save(out_path, depth)
        print(f'Saved depth map to {out_path}')
    print('Done')
