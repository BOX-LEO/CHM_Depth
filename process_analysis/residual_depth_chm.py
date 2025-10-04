import numpy as np
import os
from scipy.ndimage import gaussian_filter,gaussian_gradient_magnitude
import cv2
import rasterio
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from PIL import Image


def plot_depth_maps(chm, fused_chm, depth_pred, regularized_depth, matched_regularized_depth, image, output_file,scale = 1):
    plt.figure(figsize=(30, 20))
    plt.subplot(2, 4, 1)
    plt.imshow(chm)
    plt.colorbar()
    plt.title('original CHM')
    plt.axis('off')

    plt.subplot(2,4, 2)
    plt.imshow(fused_chm)
    plt.colorbar()
    plt.title('Fused CHM')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(depth_pred)
    plt.colorbar()
    plt.title('Depth Prediction')
    plt.axis('off')
    
    
    
    plt.subplot(2, 4, 4)
    plt.imshow(image)
    plt.title('image')

    
    plt.subplot(2, 4, 5)
    plt.imshow(regularized_depth)
    plt.colorbar()
    plt.title('Regularized Depth')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    gt_h, gt_w = chm.shape[-2:]
    pred_h, pred_w = regularized_depth.shape[-2:]
    kernel_h, kernel_w  =pred_h // gt_h, pred_w // gt_w
    pseudo_gt_resized = torch.from_numpy(regularized_depth).unsqueeze(0)
    pseudo_gt_resized = F.max_pool2d(pseudo_gt_resized[:, None], kernel_size=(kernel_h*scale, kernel_w*scale)).squeeze().numpy()
    chm_resized = torch.from_numpy(chm).unsqueeze(0)
    chm_resized = F.max_pool2d(chm_resized[:, None], kernel_size=(scale, scale)).squeeze().numpy()
    plt.scatter(pseudo_gt_resized.flatten(), chm_resized.flatten(), s=2)
    plt.xlabel('pseudo_gt ')
    plt.ylabel('GT')
    r2 = r2_score(pseudo_gt_resized.flatten(), chm_resized.flatten())
    plt.title(f'Pixel to Pixel Relationship, R2={r2:.4f}')

    plt.subplot(2, 4, 7)
    plt.imshow(matched_regularized_depth)
    plt.colorbar()
    plt.title('Matched Regularized Depth')
    plt.axis('off') 

    plt.subplot(2, 4, 8)
    gt_h, gt_w = chm.shape[-2:]
    pred_h, pred_w = matched_regularized_depth.shape[-2:]
    kernel_h, kernel_w  =pred_h // gt_h, pred_w // gt_w
    pseudo_gt_resized_matched = torch.from_numpy(matched_regularized_depth).unsqueeze(0)
    pseudo_gt_resized_matched = F.max_pool2d(pseudo_gt_resized_matched[:, None], kernel_size=(kernel_h*scale, kernel_w*scale)).squeeze().numpy()
   
    plt.scatter(pseudo_gt_resized_matched.flatten(), chm_resized.flatten(), s=2)
    plt.xlabel('pseudo_gt matched')
    plt.ylabel('GT')
    r2_matched = r2_score(pseudo_gt_resized_matched.flatten(), chm_resized.flatten())
    plt.title(f'Pixel to Pixel Relationship, R2={r2_matched:.4f}')

    
    plt.savefig(output_file)
    plt.close()


def plot_depth_r2(fused_chm, regularized_depth, output_file):
            # plot image, tuned_md, gt side by side
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(fused_chm)
        plt.colorbar()
        plt.title('chm')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(regularized_depth)
        plt.colorbar()
        plt.title('Regularized Depth')
        plt.axis('off')

        plt.subplot(1, 3,3)
        pseudo_gt_resized = torch.from_numpy(regularized_depth).unsqueeze(0)
        pseudo_gt_resized = F.interpolate(pseudo_gt_resized[:, None], fused_chm.shape[-2:], mode='bilinear', align_corners=True).squeeze().numpy()
        plt.scatter(pseudo_gt_resized.flatten(), fused_chm.flatten(), s=2)
        plt.xlabel('pseudo_gt ')
        plt.ylabel('GT')
        r2 = r2_score(pseudo_gt_resized.flatten(), fused_chm.flatten())
        plt.title(f'Pixel to Pixel Relationship, R2={r2:.4f}')

        plt.savefig(output_file)
        plt.close()


def read_tif_height(file_path):
    chm = rasterio.open(file_path)
    chm = chm.read(1)
    chm = chm.astype(np.float32)
    # flip vertically
    chm = np.flipud(chm)
    return chm


def regularized_depth(org_depth,target_depth):
    """
    Regularize the depth map using the fused CHM.
    :param org_depth: Original depth map
    :param target_depth: Target depth map
    :return: Regularized depth map
    """
    # Normalize the original depth map to match the scale of the fused CHM
    org_depth = org_depth / np.nanmax(org_depth) * np.nanmax(target_depth)
    
    # Calculate the residual depth
    residual_depth = fused_chm - org_depth
    
    # Smooth the residual depth with a Gaussian filter
    # smooth the residual depth with a Gaussian filter
    residual_depth = gaussian_filter(residual_depth, sigma=5)
    gradient_mag = gaussian_gradient_magnitude(residual_depth, sigma=5)
    alpha = 1.0  # control sharpness of transition
    weight = np.exp(-alpha * (gradient_mag / gradient_mag.max()))
    residual_depth = residual_depth * weight
    residual_depth  = gaussian_filter(residual_depth, sigma=31)
    
    # Blend the residual depth with the original depth map
    regularized_depth = org_depth + residual_depth

    # m2: clip the regularized depth to be between 0 and the max of the target depth
    regularized_depth = np.clip(regularized_depth, 0, np.nanmax(target_depth))

    # # m3: clip the regularized depth with 0 and then normalize the regularized depth map to match the scale of the target depth map
    # # normalize the regularized depth map to match the scale of the target depth map
    # target_depth_max = np.nanmax(target_depth)
    # regularized_max = np.nanmax(regularized_depth)
    # regularized_depth = np.clip(regularized_depth, 0, regularized_max)
    # regularized_depth = regularized_depth / regularized_max * target_depth_max
    
    return regularized_depth


if __name__ == "__main__":
    # Define paths for the fused CHM and depth prediction files
    base_path = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/testset'
    fused_chm_path = os.path.join(base_path, 'chm_smooth_50')
    depth_pred_path = os.path.join(base_path, 'metric_depth')
    image_path = os.path.join(base_path, 'image')

    fused_chm_files = [os.path.join(fused_chm_path, f) for f in os.listdir(fused_chm_path) if f.endswith('.tif') or f.endswith('.npy')]
    depth_pred_files = [os.path.join(depth_pred_path, f) for f in os.listdir(depth_pred_path) if f.endswith('.npy')]
    image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png') or f.endswith('.jpg')]

    fused_chm_files.sort()
    depth_pred_files.sort()
    image_files.sort()

    output_path = os.path.join(base_path, 'pseudo_gt')
    os.makedirs(output_path, exist_ok=True)



    for fused_chm_file, depth_pred_file,image_file in zip(fused_chm_files, depth_pred_files,image_files):
        image = Image.open(image_file)
        chm = np.load(fused_chm_file) if fused_chm_file.endswith('.npy') else read_tif_height(fused_chm_file)
        chm = chm- np.nanmin(chm)  # Normalize to start from 0
        depth_pred = np.load(depth_pred_file)
        # interpolate fused_chm to match the depth_pred shape
        fused_chm = cv2.resize(chm, (depth_pred.shape[1], depth_pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        fused_chm = match_histograms(fused_chm, chm, channel_axis=None)

        depth_pred = np.nanmax(depth_pred) - depth_pred  # Convert metric depth to height map


        regularized_pred = regularized_depth(depth_pred, fused_chm)
        regularized_pred_m = match_histograms(regularized_pred, fused_chm, channel_axis=None)
        
        regularized_gt = regularized_depth(regularized_pred, regularized_pred_m)
        


        # matched_regularized_depth = match_histograms(regularized_depth, fused_chm, channel_axis=None)


        np.save(os.path.join(output_path, f'{os.path.basename(fused_chm_file).replace(".tif", ".npy").replace(".png", ".npy")}'), regularized_gt)

        # plot_depth_maps(chm,fused_chm, depth_pred, regularized_pred, regularized_gt,image,
        #             os.path.join(output_path, f'residual_depth_{os.path.basename(fused_chm_file).replace(".tif", ".png").replace(".npy", ".png")}'))

        # plot_depth_r2(chm, regularized_gt,
        #             os.path.join(output_path, f'residual_depth_r2_{os.path.basename(fused_chm_file).replace(".tif", ".png").replace(".npy", ".png")}'))


        # scale_factor = fused_chm / (depth_pred+ 0.01)
        # # clip scale factor to be between 0 and 3
        # scale_factor = np.clip(scale_factor, 0, 3)

        # # scale_factor = gaussian_filter(scale_factor, sigma=3)
        # # element wise multiply the depth_pred with the scale factor

        # regularized_depth = depth_pred * scale_factor

        # plot_depth_maps(fused_chm, depth_pred, scale_factor, regularized_depth,
        #             os.path.join(output_path, f'scale_factor_{os.path.basename(fused_chm_file).replace(".tif", ".png").replace(".npy", ".png")}'))