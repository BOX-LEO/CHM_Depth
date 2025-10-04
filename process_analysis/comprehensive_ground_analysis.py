"""
COMPREHENSIVE GROUND ANALYSIS SCRIPT

This script performs comprehensive analysis of LiDAR-derived data focusing on:
1. Sample distribution analysis across different %ground thresholds
2. Ground threshold analysis with CHM downsampling using different strategies for ground vs non-ground regions

The script analyzes how model predictions (metric depth) perform against ground truth (CHM) 
under different conditions related to ground coverage (%ground).

Key Features:
- Analyzes scale=1 samples only (individual tiles)
- Tests thresholds from 10% to 0.1% ground coverage
- Implements smart downsampling: max pooling for vegetation, mean pooling for ground
- Provides comprehensive statistics and insights
- Saves results to multiple CSV files

"""

import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import r2_score
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns

def read_tif_height(file_path):
    """
    Read TIF file and return height data
    
    Args:
        file_path (str): Path to the TIF file
        
    Returns:
        np.ndarray: Height data as float32 array, vertically flipped
    """
    chm = rasterio.open(file_path)
    chm = chm.read(1)
    chm = chm.astype(np.float32)
    # flip vertically to match coordinate system
    chm = np.flipud(chm)
    return chm


def smart_downsample(height_map, filter_size):
    """
    Smart downsampling of height_map using different strategies for ground vs non-ground regions
    
    Ground regions (height_map < 5m): Use mean pooling to preserve average ground height
    Non-ground regions (height_map >= 5m): Use max pooling to preserve canopy structure
    
    Args:
        height_map (np.ndarray): height_map array to downsample
        filter_size (int): Size of the pooling filter
        
    Returns:
        np.ndarray: Downsampled height_map array
    """
    if filter_size == 1:
        return height_map  # No downsampling
    
    h, w = height_map.shape
    
    # Calculate output dimensions
    out_h = h // filter_size
    out_w = w // filter_size
    
    # Create output array
    downsampled = np.zeros((out_h, out_w), dtype=np.float32)
    
    for i in range(out_h):
        for j in range(out_w):
            # Extract the patch
            start_h = i * filter_size
            end_h = start_h + filter_size
            start_w = j * filter_size
            end_w = start_w + filter_size
            
            patch = height_map[start_h:end_h, start_w:end_w]
            
            # Check if patch contains ground pixels
            ground_mask = patch < 5
            non_ground_mask = patch >= 5
            
            # if np.any(ground_mask) and np.any(non_ground_mask):
            #     # Mixed patch: use weighted average
            #     ground_mean = np.mean(patch[ground_mask]) if np.any(ground_mask) else 0
            #     non_ground_max = np.max(patch[non_ground_mask]) if np.any(non_ground_mask) else 0
                
            #     # Weight by proportion of each type
            #     ground_weight = np.sum(ground_mask) / patch.size
            #     non_ground_weight = np.sum(non_ground_mask) / patch.size
            #     downsampled[i, j] = ground_weight * ground_mean + non_ground_weight * non_ground_max
                
            # elif np.any(ground_mask):
            #     # Pure ground patch: use mean
            #     downsampled[i, j] = np.mean(patch[ground_mask])
            # else:
            #     # Pure non-ground patch: use max
            #     downsampled[i, j] = np.max(patch)
            # if np.any(ground_mask):
            #     # Pure ground patch: use min
            #     downsampled[i, j] = np.mean(patch[ground_mask])
            # else:
            #     # Pure non-ground patch: use max
            #     downsampled[i, j] = np.max(patch)
            downsampled[i, j] = np.mean(patch)
    return downsampled

def extract_coordinates(filename):
    """
    Extract x, y coordinates from filename crop_<x>_<y>.ext
    
    Args:
        filename (str): Filename in format crop_<x>_<y>.ext
        
    Returns:
        tuple: (x, y) coordinates as integers, or (None, None) if parsing fails
    """
    match = re.match(r'crop_(\d+)_(\d+)\.', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def analyze_sample_distribution(df_scale1):
    """
    Analyze the distribution of samples across different %ground thresholds
    
    This function provides detailed statistics about how samples are distributed
    across different ground coverage percentages, including:
    - Individual threshold ranges
    - Cumulative distributions
    - Mean R² scores and other metrics for each range
    
    Args:
        df_scale1 (pd.DataFrame): DataFrame containing scale=1 samples
        
    Returns:
        tuple: (distribution_df, cumulative_df) - Two DataFrames with analysis results
    """
    
    print("\n" + "=" * 80)
    print("SAMPLE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Define thresholds from 10% to 0.1% ground coverage
    thresholds = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.2, 0.1]
    
    print(f"Found {len(df_scale1)} scale=1 samples")
    print(f"Mean %ground: {df_scale1['%ground'].mean():.2f}%")
    print(f"Std %ground: {df_scale1['%ground'].std():.2f}%")
    print(f"Min %ground: {df_scale1['%ground'].min():.2f}%")
    print(f"Max %ground: {df_scale1['%ground'].max():.2f}%")
    
    # Analyze sample distribution across threshold ranges
    distribution_data = []
    
    for i, threshold in enumerate(thresholds):
        if i == 0:
            # First threshold: samples > threshold
            samples_in_range = df_scale1[df_scale1['%ground'] > threshold]
            range_label = f"> {threshold}%"
        else:
            # Other thresholds: samples between previous and current threshold
            prev_threshold = thresholds[i-1]
            samples_in_range = df_scale1[(df_scale1['%ground'] > threshold) & (df_scale1['%ground'] <= prev_threshold)]
            range_label = f"{threshold}% - {prev_threshold}%"
        
        count = len(samples_in_range)
        percentage = (count / len(df_scale1)) * 100 if len(df_scale1) > 0 else 0
        
        distribution_data.append({
            'range': range_label,
            'count': count,
            'percentage': percentage,
            'mean_r2': samples_in_range['r2_chm'].mean() if count > 0 else np.nan,
            'mean_ground': samples_in_range['%ground'].mean() if count > 0 else np.nan,
            'mean_chm_std': samples_in_range['chm_std'].mean() if count > 0 else np.nan
        })
    
    # Create distribution DataFrame
    distribution_df = pd.DataFrame(distribution_data)
    
    print("\nSample Distribution by %ground Threshold:")
    print(distribution_df.to_string(index=False, float_format='%.2f'))
    
    # Cumulative analysis - samples above each threshold
    print("\n" + "-" * 60)
    print("CUMULATIVE ANALYSIS")
    print("-" * 60)
    
    cumulative_data = []
    for threshold in thresholds:
        samples_above = df_scale1[df_scale1['%ground'] > threshold]
        count = len(samples_above)
        percentage = (count / len(df_scale1)) * 100 if len(df_scale1) > 0 else 0
        
        cumulative_data.append({
            'threshold': threshold,
            'samples_above': count,
            'percentage_above': percentage,
            'mean_r2': samples_above['r2_chm'].mean() if count > 0 else np.nan,
            'mean_ground': samples_above['%ground'].mean() if count > 0 else np.nan,
            'mean_chm_std': samples_above['chm_std'].mean() if count > 0 else np.nan
        })
    
    cumulative_df = pd.DataFrame(cumulative_data)
    print("Cumulative Sample Distribution:")
    print(cumulative_df.to_string(index=False, float_format='%.2f'))
    
    return distribution_df, cumulative_df

def analyze_ground_thresholds_with_downsampling(df_scale1, chm_path, pred_path,plot_flag=False):
    """
    Analyze R² scores for different %ground thresholds with smart CHM downsampling
    
    This function performs the core analysis by:
    1. Filtering samples based on %ground thresholds
    2. Loading prediction and CHM data for each sample
    3. Applying smart downsampling to CHM (different strategies for ground vs canopy)
    4. Computing combined R² scores for all pixels in each threshold group
    5. Testing multiple downsampling levels (50, 25, 10, 5)
    
    Args:
        df_scale1 (pd.DataFrame): DataFrame containing scale=1 samples
        chm_path (str): Path to CHM data directory
        pred_path (str): Path to prediction data directory
        
    Returns:
        pd.DataFrame: Results DataFrame with R² scores for each threshold and filter size
    """
    
    print("\n" + "=" * 80)
    print("GROUND THRESHOLD ANALYSIS WITH SMART DOWNSAMPLING")
    print("=" * 80)
    
    # Define thresholds from 10% to 0.1% ground coverage, low pass threshold(ground coverage > threshold)
    thresholds = [0.1,0.2,0.5,1,2,3,4,5,6,7,8,9,10]
    
    # Define downsampling filter sizes (1 = no downsampling)
    ################################################################
    target_sizes = [(5,5),(10,10),(25,25),(50,50)]
    # target_sizes = [(50, 50)]
    all_results = []
    
    for target_size in target_sizes:
        print(f"\n{'='*60}")
        print(f"ANALYSIS WITH SMART CHM DOWNSAMPLING (target_size = {target_size})")
        print(f"{'='*60}")
        
        results = []
        
        for threshold in tqdm(thresholds, desc=f"Processing thresholds (target_size={target_size})"):
            # Select samples where %ground > threshold
            filtered_samples = df_scale1[df_scale1['%ground'] > threshold]
            
            if len(filtered_samples) == 0:
                results.append({
                    'target_size': target_size,
                    'threshold': threshold,
                    'r2_chm_combined': np.nan,
                    'number_of_pixels': 0,
                    'number_of_samples': 0
                })
                continue
            
            print(f"\nThreshold {threshold}%: {len(filtered_samples)} samples")
            
            # Collect all prediction and CHM arrays
            all_pred_pixels = []
            all_chm_pixels = []
            
            for _, row in filtered_samples.iterrows():
                x, y = extract_coordinates(f"crop_{row['coordinate']}.npy")
                if x is None or y is None:
                    continue
                    
                # Load prediction and CHM files
                pred_file = os.path.join(pred_path, f"crop_{x}_{y}.npy")
                chm_file = os.path.join(chm_path, f"crop_{x}_{y}.tif")
                
                if not os.path.exists(pred_file) or not os.path.exists(chm_file):
                    continue
                
                try:
                    pred = np.load(pred_file)
                    chm = read_tif_height(chm_file)
                    
                    # Convert metric depth to height map (same as in original analysis)
                    max_depth = 40
                    pred = np.array(pred, dtype=np.float32)
                    pred = max_depth - pred # convert to height map
                    chm = chm - chm.min() # normalize to 0
                    
                    # Apply smart downsampling to CHM if filter_size > 1
                    filter_size = chm.shape[0]//target_size[0]
                    if filter_size > 1:
                        chm = smart_downsample(chm, filter_size)
                    
                    # Resize prediction to match CHM dimensions
                    filter_size = pred.shape[0]//target_size[0]
                    pred_resized = smart_downsample(pred, filter_size)
                    
                    # Flatten and add to collections
                    all_pred_pixels.extend(pred_resized.flatten())
                    all_chm_pixels.extend(chm.flatten())
                    
                except Exception as e:
                    print(f"Error processing {x}_{y}: {e}")
                    continue
            
            # Convert to numpy arrays
            all_pred_pixels = np.array(all_pred_pixels)
            all_chm_pixels = np.array(all_chm_pixels)
            
            
            # Compute R² score for combined pixels
            if len(all_pred_pixels) > 0:
                r2_score_combined = r2_score(all_chm_pixels, all_pred_pixels)
                rmse = np.sqrt(np.mean((all_chm_pixels - all_pred_pixels)**2))
                # scatter plot
                if target_size == (50, 50) and 'pseudo_full' in pred_path and plot_flag:
                    plt.figure(figsize=(10, 10))
                    
                    # Create density visualization
                    sns.kdeplot(x=all_chm_pixels, y=all_pred_pixels, fill=True, cmap='YlOrBr', 
                               thresh=0.05, levels=100, alpha=1)

                    plt.xlim(-3, 40)
                    plt.ylim(-3, 40)
                    # Overlay scatter plot of all points
                    plt.scatter(all_chm_pixels, all_pred_pixels, alpha=0.2, color = 'brown', s=2 , edgecolors='none')
                    
                    # Calculate and plot the line of best fit
                    z = np.polyfit(all_chm_pixels, all_pred_pixels, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(-3, 40, 100)
                    plt.plot(x_range, p(x_range), color='fuchsia', linewidth=2, label='Best Fit')
                    
                    # Plot the y=x line
                    plt.plot([-3, 40], [-3, 40], color='black', linewidth=2, linestyle='--', label='y=x')
                    
                    plt.legend()
                    plt.xlabel('CHM')
                    plt.ylabel('Prediction')
                    plt.title(f'R²: {r2_score_combined:.4f}, RMSE: {rmse:.4f}')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f'scatter_plot_YlOrBr_{target_size}_{threshold}.png', dpi=300, bbox_inches='tight')
                    plt.close()
            else:
                r2_score_combined = np.nan
            
            results.append({
                'target_size': target_size,
                'threshold': threshold,
                'r2_chm_combined': r2_score_combined,
                'number_of_pixels': len(all_pred_pixels),
                'number_of_samples': len(filtered_samples),
                'rmse': rmse
            })
            
            print(f"  Pixels: {len(all_pred_pixels):,}, R²: {r2_score_combined:.4f}")
        
        all_results.extend(results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Print summary tables for each target size
    for target_size in target_sizes:
        print(f"\n{'='*60}")
        print(f"SUMMARY RESULTS FOR TARGET_SIZE = {target_size}")
        print(f"{'='*60}")
        target_results = results_df[results_df['target_size'] == target_size]
        print(target_results[['threshold', 'r2_chm_combined', 'number_of_pixels', 'number_of_samples']].to_string(index=False, float_format='%.4f'))
    
    # Additional analysis and insights
    print("\n" + "=" * 60)
    print("ADDITIONAL INSIGHTS")
    print("=" * 60)
    
    # Find best threshold for each target size
    for target_size in target_sizes:
        target_results = results_df[results_df['target_size'] == target_size].dropna()
        if len(target_results) > 0:
            best_idx = target_results['r2_chm_combined'].idxmax()
            best_threshold = target_results.loc[best_idx, 'threshold']
            best_r2 = target_results.loc[best_idx, 'r2_chm_combined']
            best_pixels = target_results.loc[best_idx, 'number_of_pixels']
            
            print(f"\nTarget size {target_size}:")
            print(f"  Best R² score: {best_r2:.4f} at threshold {best_threshold}%")
            print(f"  Pixels included: {best_pixels:,}")
            print(f"  Samples included: {target_results.loc[best_idx, 'number_of_samples']}")
    
    # Create pivot table for easy comparison across target sizes
    print(f"\n{'='*60}")
    print("R² SCORES COMPARISON TABLE")
    print(f"{'='*60}")
    pivot_table = results_df.pivot(index='threshold', columns='target_size', values='r2_chm_combined')
    print(pivot_table.round(4))
    
    return results_df

def analyze_ground_thresholds(df_scale1, pseudo_gt_path, pred_path):
    """
    Analyze R² scores for different %ground thresholds without downsampling
    
    This function performs analysis by:
    1. Filtering samples based on %ground thresholds
    2. Loading prediction and pseudo_gt data for each sample (both at 1000x1000 resolution)
    3. Computing combined R² scores for all pixels in each threshold group
    4. No downsampling is applied - both datasets maintain original 1000x1000 resolution
    
    Args:
        df_scale1 (pd.DataFrame): DataFrame containing scale=1 samples
        pseudo_gt_path (str): Path to pseudo ground truth data directory
        pred_path (str): Path to prediction data directory
        
    Returns:
        pd.DataFrame: Results DataFrame with R² scores for each threshold
    """
    
    print("\n" + "=" * 80)
    print("GROUND THRESHOLD ANALYSIS WITHOUT DOWNSAMPLING")
    print("=" * 80)
    
    # Define thresholds from 10% to 0.1% ground coverage, low pass threshold(ground coverage > threshold)
    thresholds = [0.1]
    
    results = []
    
    for threshold in tqdm(thresholds, desc="Processing thresholds (no downsampling)"):
        # Select samples where %ground > threshold
        filtered_samples = df_scale1[df_scale1['%ground'] > threshold]
        
        if len(filtered_samples) == 0:
            results.append({
                'threshold': threshold,
                'r2_combined': np.nan,
                'rmse_combined': np.nan,
                'number_of_pixels': 0,
                'number_of_samples': 0
            })
            continue
        
        print(f"\nThreshold {threshold}%: {len(filtered_samples)} samples")
        
        # Collect all prediction and pseudo_gt arrays
        all_pred_pixels = []
        all_pseudo_gt_pixels = []
        
        for _, row in filtered_samples.iterrows():
            x, y = extract_coordinates(f"crop_{row['coordinate']}.npy")
            if x is None or y is None:
                continue
                
            # Load prediction and pseudo_gt files
            pred_file = os.path.join(pred_path, f"crop_{x}_{y}.npy")
            pseudo_gt_file = os.path.join(pseudo_gt_path, f"crop_{x}_{y}.npy")
            
            if not os.path.exists(pred_file) or not os.path.exists(pseudo_gt_file):
                continue
            
            try:
                pred = np.load(pred_file)
                pseudo_gt = np.load(pseudo_gt_file)
                
                # Convert metric depth to height map (same as in original analysis)
                max_depth = 40
                pred = np.array(pred, dtype=np.float32)
                pred = max_depth - pred  # convert to height map
                pseudo_gt = np.array(pseudo_gt, dtype=np.float32)
                pseudo_gt = pseudo_gt - pseudo_gt.min()  # normalize to 0
                
                # No downsampling - use original 1000x1000 resolution
                # Flatten and add to collections
                all_pred_pixels.extend(pred.flatten())
                all_pseudo_gt_pixels.extend(pseudo_gt.flatten())
                
            except Exception as e:
                print(f"Error processing {x}_{y}: {e}")
                continue
        
        # Convert to numpy arrays
        all_pred_pixels = np.array(all_pred_pixels)
        all_pseudo_gt_pixels = np.array(all_pseudo_gt_pixels)
        
        # Compute R² score and RMSE for combined pixels
        if len(all_pred_pixels) > 0:
            r2_score_combined = r2_score(all_pseudo_gt_pixels, all_pred_pixels)
            rmse_combined = np.sqrt(np.mean((all_pseudo_gt_pixels - all_pred_pixels)**2))
            
            # # Create scatter plot for visualization
            # plt.figure(figsize=(10, 10))
            
            # # Create density visualization
            # # sns.kdeplot(x=all_pseudo_gt_pixels, y=all_pred_pixels, fill=True, cmap='YlOrBr', 
            # #            thresh=0.05, levels=100, alpha=1)

            # plt.xlim(-3, 40)
            # plt.ylim(-3, 40)
            # # Overlay scatter plot of all points
            # plt.scatter(all_pseudo_gt_pixels, all_pred_pixels, alpha=0.2, color='brown', s=2, edgecolors='none')
            
            # # Calculate and plot the line of best fit
            # z = np.polyfit(all_pseudo_gt_pixels, all_pred_pixels, 1)
            # p = np.poly1d(z)
            # x_range = np.linspace(-3, 40, 100)
            # plt.plot(x_range, p(x_range), color='fuchsia', linewidth=2, label='Best Fit')
            
            # # Plot the y=x line
            # plt.plot([-3, 40], [-3, 40], color='black', linewidth=2, linestyle='--', label='y=x')
            
            # plt.legend()
            # plt.xlabel('Pseudo Ground Truth')
            # plt.ylabel('Prediction')
            # plt.title(f'R²: {r2_score_combined:.4f}, RMSE: {rmse_combined:.4f}')
            # plt.grid(True, alpha=0.3)
            # plt.savefig(f'scatter_plot_no_downsampling_{threshold}.png', dpi=300, bbox_inches='tight')
            # plt.close()
        else:
            r2_score_combined = np.nan
            rmse_combined = np.nan
        
        results.append({
            'threshold': threshold,
            'r2_combined': r2_score_combined,
            'rmse_combined': rmse_combined,
            'number_of_pixels': len(all_pred_pixels),
            'number_of_samples': len(filtered_samples)
        })
        
        print(f"  Pixels: {len(all_pred_pixels):,}, R²: {r2_score_combined:.4f}, RMSE: {rmse_combined:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS (NO DOWNSAMPLING)")
    print(f"{'='*60}")
    print(results_df[['threshold', 'r2_combined', 'rmse_combined', 'number_of_pixels', 'number_of_samples']].to_string(index=False, float_format='%.4f'))
    
    # Additional analysis and insights
    print("\n" + "=" * 60)
    print("ADDITIONAL INSIGHTS")
    print("=" * 60)
    
    # Find best threshold
    valid_results = results_df.dropna()
    if len(valid_results) > 0:
        best_idx = valid_results['r2_combined'].idxmax()
        best_threshold = valid_results.loc[best_idx, 'threshold']
        best_r2 = valid_results.loc[best_idx, 'r2_combined']
        best_rmse = valid_results.loc[best_idx, 'rmse_combined']
        best_pixels = valid_results.loc[best_idx, 'number_of_pixels']
        
        print(f"\nBest R² score: {best_r2:.4f} at threshold {best_threshold}%")
        print(f"Best RMSE: {best_rmse:.4f} at threshold {best_threshold}%")
        print(f"Pixels included: {best_pixels:,}")
        print(f"Samples included: {valid_results.loc[best_idx, 'number_of_samples']}")
    
    return results_df

def comprehensive_ground_analysis(sub_path):
    """
    Main function that runs all ground analysis components
    
    This function orchestrates the entire analysis pipeline:
    1. Loads the prediction analysis results CSV
    2. Filters for scale=1 samples only (individual tiles)
    3. Runs sample distribution analysis
    4. Runs ground threshold analysis with smart downsampling
    5. Saves all results to CSV files
    6. Provides overall statistics and insights
    
    Returns:
        dict: Dictionary containing all analysis results DataFrames
    """
    
    print("=" * 80)
    print("COMPREHENSIVE GROUND ANALYSIS")
    print("=" * 80)
    
    # Set up paths to data directories
    root_path = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/testset'
    chm_path = os.path.join(root_path, 'chm_smooth_50')
    pseudo_gt_path = os.path.join(root_path, 'pseudo_gt')
    pred_root_path = '/home/boxiang/work/dao2/USGS_LiDAR/martell_data/abalation/test_set_result/'
    
    
    pred_path = os.path.join(pred_root_path, sub_path)
    # Load the CSV data from previous analysis
    df = pd.read_csv(f'/home/boxiang/work/dao2/USGS_LiDAR/martell_data/abalation/test_set_result/abalation_{sub_path}.csv')

    # Filter for scale=1 only (individual tiles)
    df_scale1 = df[df['scale'] == 1].copy()
    print(f"Found {len(df_scale1)} scale=1 samples")
    
    # Run all analyses
    print("\nStarting sample distribution analysis...")
    # distribution_df, cumulative_df = analyze_sample_distribution(df_scale1)
    
    print("\nStarting ground threshold analysis with smart downsampling...")
    downsampling_results = analyze_ground_thresholds_with_downsampling(df_scale1, chm_path, pred_path)
    # downsampling_results = analyze_ground_thresholds(df_scale1, pseudo_gt_path, pred_path)

    
    # Save all results to CSV files
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # # Save sample distribution results
    # distribution_file = os.path.join(pred_root_path, f'sample_distribution_analysis_{sub_path}.csv')
    # distribution_df.to_csv(distribution_file, index=False)
    # print(f"Sample distribution results saved to: {distribution_file}")
    
    # cumulative_file = os.path.join(pred_root_path, f'cumulative_distribution_analysis_{sub_path}.csv')
    # # cumulative_df.to_csv(cumulative_file, index=False)
    # print(f"Cumulative distribution results saved to: {cumulative_file}")
    
    # # Save downsampling analysis results
    # downsampling_file = os.path.join(pred_root_path, f'ground_threshold_analysis_with_downsampling_{sub_path}.csv')
    # downsampling_results.to_csv(downsampling_file, index=False)
    # print(f"Downsampling analysis results saved to: {downsampling_file}")
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(df_scale1)}")
    print(f"Mean %ground: {df_scale1['%ground'].mean():.2f}%")
    print(f"Std %ground: {df_scale1['%ground'].std():.2f}%")
    print(f"Mean R² (individual samples): {df_scale1['r2_chm'].mean():.4f}")
    print(f"Std R² (individual samples): {df_scale1['r2_chm'].std():.4f}")

    # print the summary statistics of the downsampling results with different target sizes and thresholds in a table
    
    
    print(f"\nAnalysis complete! All results have been saved to the output directory.")
    
    return {
        ################################################################
        # 'distribution': distribution_df,
        # 'cumulative': cumulative_df,
        'downsampling': downsampling_results
    }

if __name__ == "__main__":
    # Run the comprehensive analysis
    ################################################################
    # sub_path = ['pseudo_full', 'pseudo_head','chm_head','chm_full']
    sub_path = ['pseudo_full']
    for s in sub_path:
        results = comprehensive_ground_analysis(s) 