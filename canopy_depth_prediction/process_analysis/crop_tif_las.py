import os
import laspy
import numpy as np
import pandas as pd
import rasterio
from sklearn.neighbors import NearestNeighbors
from pyproj import Transformer, CRS
from PIL import Image
import argparse
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cv2
import time
import shutil
from tqdm import tqdm
from scipy.stats import binned_statistic_2d

'''
this script is used crop both the tif and las file and generate the CHM for martell data
'''



def get_las_crs(input):
    # read the las file
    if isinstance(input, str):
        infile = laspy.read(input)
    else:
        infile = input
    # get the crs of the las file
    crs = infile.header.parse_crs()
    # if crs is epsg code
    try:
        horizontal_crs = crs.sub_crs_list[0]
        horizontal_epsg = horizontal_crs.to_epsg()
        # print('horizontal_epsg:', horizontal_epsg)
        return horizontal_epsg
    except:
        return crs

def crop_las(las_path,top_left,bottom_right,output_path):
    x_min, y_max = top_left
    x_max, y_min = bottom_right
    las = laspy.read(las_path)
    x = las.x
    y = las.y

    # Create a mask for points within the bounding box
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    if np.sum(mask)==0:
        print('no points in the bounding box')
        return
    # Create a new LAS file with the cropped points
    cropped = laspy.LasData(las.header)
    cropped.points = las.points[mask]

    # Update the bounding box (optional but recommended)
    cropped.update_header()

    cropped.write(output_path)

def canopy_height_model(file_path:str,
                      avg_ground:float = 0.0,
                      top_left:tuple = (None,None),
                      bottom_right:tuple = (None,None),
                      gird_num:int = 20,
                      output_CHM_file:str=None,
                      smooth:bool=True,
                      crs=None):
    infile = laspy.read(file_path)
    x,y,z= infile.x,infile.y,infile.z
    coords=np.vstack((x,y,z)).transpose()
    classification = infile.classification

    # digital Terrain model
    # filter out the ground points
    t = time.time()
    ground_mask = (classification==2)
    ground_x, ground_y, ground_z = x[ground_mask], y[ground_mask], z[ground_mask]
    (x_min,y_max) = top_left
    (x_max,y_min) = bottom_right
    grid_res_x = (x_max-x_min)/(gird_num-1)
    grid_res_y = (y_max-y_min)/(gird_num-1)
    # define the grid
    x_grid = np.arange(x_min, x_max+ 0.0001, grid_res_x)  # add a small value to avoid rounding errors
    y_grid = np.arange(y_min, y_max+ 0.0001, grid_res_y)  # add a small value to avoid rounding errors
    x_grid[-1] = x_max
    y_grid[-1] = y_max
    X,Y = np.meshgrid(x_grid, y_grid)
    
    assert len(x_grid) == len(y_grid) == gird_num , f'grid number {gird_num} does not match the grid size {len(x_grid)} and {len(y_grid)}'
    
    
    # interpolate the ground points to the grid
    if len(ground_x) > 0:
        print('found ground points')
        DTM = griddata((ground_x, ground_y), ground_z, (X,Y), method='nearest')
    else:
        
        DTM = np.zeros_like(X)

    # filter out outliers of z, keep mean+-3*std
    # z_mean = np.mean(z)
    # z_std = np.std(z)
    # z_mask = (z>z_mean-3*z_std) & (z<z_mean+3*z_std)
    # z[~z_mask] = 0 # set outliers to 0

    stat, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z
        , statistic='max',
        bins=[gird_num, gird_num])
    DSM = stat.T  # transpose (row, column) to (x, y)
    # print('finished DSM in', time.time()-t)

    print('max-min DSM:', np.nanmax(DSM)-np.nanmin(DSM))
    DSM = np.nan_to_num(DSM, nan=np.nanmin(DSM))  # replace NaN with min value
    CHM = DSM - DTM

    if smooth:
        CHM_smooth = cv2.GaussianBlur(CHM, (3, 3), 0)
        CHM = np.maximum(CHM, CHM_smooth)  # get max of smooth CHM and original CHM
    print('max-min CHM:', np.nanmax(CHM)-np.nanmin(CHM))
    # get max of smooth CHM and original CHM

    if output_CHM_file is not None:
        transform = rasterio.transform.from_origin(x.min(), y.max(), grid_res_x, grid_res_y)
        with rasterio.open(output_CHM_file, 'w', driver='GTiff', height=CHM.shape[0], width=CHM.shape[1],\
                            count=1, dtype=CHM.dtype, crs=crs, transform=transform) as dst:
            dst.write(CHM, 1)
    return CHM

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-si','--save_image', type=str, default='False', help='save the crop image(crop of tif based on lidar data) or not')
    argparser.add_argument('-sc','--save_CHM', type=str, default='True', help='save the CHM or not')
    argparser.add_argument('-sm','--smooth', type=str, default='True', help='smooth the CHM or not')
    argparser.add_argument('-cps','--crop_size', type=int, default=1000, help='crop size of the tif file in pixels')
    argparser.add_argument('-stps','--crop_stap_size', type=int, default=500, help='crop step size of the tif file in pixels')
    argparser.add_argument('-grid','--grid_num', type=int, default=50, help='number of grids to generate the CHM')
    argparser.add_argument('-proj','--proj_file', type=str, default='/home/boxiang/work/dao2/USGS_LiDAR/martell_data/JW4D20240829_01_las1_2.prj', help='projection file of the las file')

    args = argparser.parse_args()
    save_image = args.save_image=='True'
    save_CHM = args.save_CHM=='True'
    smooth = args.smooth=='True'
    crop_size = args.crop_size
    crop_step_size = args.crop_stap_size
    grid_num = args.grid_num

    lidar_data_path = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/NewGood_Martell_plot4D_sample100m.las'
    input_tif = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/orthomosaic_20240829_Plot4D_JW.tif'
    base_path = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/testset'
    # os.makedirs(base_path, exist_ok=True)

    crop_tif_path = os.path.join(base_path, 'image')
    crop_las_path = os.path.join(base_path, 'lidar')
    if smooth:
        chm_path = os.path.join(base_path, f'chm_smooth_{grid_num}')
    else:
        chm_path = os.path.join(base_path, 'chm')

    os.makedirs(crop_tif_path, exist_ok=True)
    os.makedirs(crop_las_path, exist_ok=True)
    os.makedirs(chm_path, exist_ok=True)

    # convert laz file to las file
    convert_laz = False
    if convert_laz:
        laz_files = os.listdir(lidar_data_path)
        for laz_file in laz_files:
            if laz_file.endswith('.laz'):
                las_file = os.path.join(lidar_data_path, laz_file[:-4]+'.las')
                if not os.path.exists(las_file):
                    print('converting', laz_file)
                    infile = laspy.read(os.path.join(lidar_data_path, laz_file))
                    infile.write(las_file)
                    print('converted', laz_file)
    
    # combine laz files to one laz file
    combine_laz = False
    if combine_laz:
        output_file = os.path.join(lidar_data_path, 'combined.laz')
        las_files = os.listdir(lidar_data_path)
        las_files = [os.path.join(lidar_data_path, las_file) for las_file in las_files if las_file.endswith('.las')]
        shutil.copyfile(las_files[0], output_file)
        with laspy.open(output_file, mode='a') as writer:
            for las_file in las_files[1:]:
                with laspy.open(las_file) as reader:
                    if reader.header.point_format != writer.header.point_format:
                        raise ValueError("Mismatched point formats.")
                    for points in tqdm(reader.chunk_iterator(1_000_000)):
                        writer.append_points(points)

    # get coordinate system of the tif file
    with rasterio.open(input_tif) as src:
        tif_crs = CRS.from_user_input(src.crs)
        print("Axis units:", tif_crs.axis_info[0].unit_name, "/", tif_crs.axis_info[1].unit_name)
     

    # get the crs of the las file
    if args.proj_file is not None:
        with open(args.proj_file, 'r') as f:
            proj = f.read()
        las_crs = CRS.from_wkt(proj)
    else:
        las_crs = get_las_crs(lidar_data_path)
        print('las_crs:', las_crs)

    try:
        transformer_tif2las = Transformer.from_crs(tif_crs, las_crs, always_xy=True)
    except:
        raise ValueError(f'the crs of the tif file {tif_crs} and the las file {las_crs} are not compatible')
    
    las = laspy.read(lidar_data_path)
    las_top_left = (las.x.min(), las.y.max())
    las_bottom_right = (las.x.max(), las.y.min())


    # print(dir(las.header))
    # print("z scale:", las.header.scales[2])
    # print("z offset:", las.header.offsets[2])
    # print("z min/max:", las.header.mins[2], las.header.maxs[2])
    # print("z max- min:", las.z.max() - las.z.min())
    # z = las.z
    # print('z min/max:', z.min(), z.max())

    with rasterio.open(input_tif) as src:
        tif_top_left = (src.bounds.left, src.bounds.top)
        tif_bottom_right = (src.bounds.right, src.bounds.bottom)
        print('tif_top_left:', tif_top_left,'tif_bottom_right:', tif_bottom_right) # in tif coordinates
        
        # get the resolution of the tif file and the chm grid
        window = ((0, crop_size), (0, crop_size)) # pixel coordinate system
        pix2coor_transform = src.window_transform(window)
        tif_top_left = pix2coor_transform * (0, 0) # top left corner of crop tif in tif coordinates
        tif_bottom_right = pix2coor_transform * (crop_size, crop_size) # bottom right corner of crop tif in tif coordinates
        tif_top_left = transformer_tif2las.transform(tif_top_left[0], tif_top_left[1]) # top left corner of crop tif in las coordinates
        tif_bottom_right = transformer_tif2las.transform(tif_bottom_right[0], tif_bottom_right[1]) # bottom right corner of crop tif in las coordinates
        h,w = tif_top_left[1] - tif_bottom_right[1],  tif_bottom_right[0]-tif_top_left[0]
        resolution = (w / crop_size, h / crop_size)
        print('tif resolution:', resolution)
        print('chm grid resolution:', (w / grid_num, h / grid_num))
        # get the height and width of the tif file
        h,w = src.height,src.width
        print('tif_h:', h, 'tif_w:', w)
    
    
    progress_bar = tqdm(total=(w+1)*(h+1)//(crop_step_size**2), desc='Cropping', unit='crop')
    # read the tif file
    with rasterio.open(input_tif) as src:
        for c in range(0, w+1, crop_step_size):
            for r in range(0, h+1, crop_step_size):
                output_image_file = f'crop_{c}_{r}.png'
                # get the crop area
                window = ((r,r+crop_size), (c,c+crop_size)) # pixel coordinate system
                # print('window:', window)
                pix2coor_transform = src.window_transform(window)
                 # get geometry coordinates in tif coordinates system
                tif_top_left = pix2coor_transform * (0, 0)
                tif_bottom_right = pix2coor_transform * (crop_size, crop_size)
                # transform corner of tif to lidar coordinates system
                tif_top_left = transformer_tif2las.transform(tif_top_left[0], tif_top_left[1]) # top left corner of crop tif in las coordinates
                tif_bottom_right = transformer_tif2las.transform(tif_bottom_right[0], tif_bottom_right[1]) # bottom right corner of crop tif in las coordinates
                
                # check if the crop area is fully inside the las file
                if not (tif_top_left[0] >= las_top_left[0] and tif_bottom_right[0] <= las_bottom_right[0] and \
                        tif_top_left[1] <= las_top_left[1] and tif_bottom_right[1] >= las_bottom_right[1]):
                    progress_bar.update(1)
                    continue

                data = src.read(window=window)
                # check if the image is rgb infrared
                if data.shape[0]==4:
                    rgb = data[:3,:,:]
                elif data.shape[0]==3:
                    rgb = data
                else:
                    raise ValueError('the tif file should be rgb or rgb infrared')

                rgb = np.clip(rgb,0,255).astype(np.uint8)
                rgb = np.moveaxis(rgb, 0, -1)
                # check it there are too much pixels in the image that are pure white 
                # (crop at the edge of the image has lots of white pixels indicating low quality)
                if np.sum(rgb==255)>3000:
                    progress_bar.update(1)
                    continue
                # save the image
                if save_image:
                    img=Image.fromarray(rgb)
                    out_path = os.path.join(crop_tif_path, output_image_file)
                    img.save(out_path)

                # crop the las file
                las_output_file = os.path.join(crop_las_path, output_image_file[:-4]+'.las')
                crop_las(lidar_data_path, tif_top_left, tif_bottom_right, las_output_file)
                # generate the CHM
                if save_CHM:
                    chm_output_file = os.path.join(chm_path, output_image_file[:-4]+'.tif')
                    CHM = canopy_height_model(las_output_file, 
                                            avg_ground=0.0,
                                            top_left=tif_top_left,
                                            bottom_right=tif_bottom_right,
                                            gird_num=grid_num,
                                            output_CHM_file=chm_output_file,
                                            smooth=smooth,
                                            crs=las_crs)

                    # print('CHM shape:', CHM.shape)

                progress_bar.update(1)
    progress_bar.close()


    
