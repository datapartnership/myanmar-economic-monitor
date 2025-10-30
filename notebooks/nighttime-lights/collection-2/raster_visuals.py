"""
Raster visualization utilities for nighttime lights analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from rasterio.merge import merge
from rasterio.features import geometry_mask
import rasterio


def process_raster_tiles(file_list, threshold=0.5):
    """
    Mosaic multiple raster tiles and extract data.
    
    Parameters
    ----------
    file_list : list
        List of file paths to raster tiles
    threshold : float, optional
        Minimum value threshold for filtering, by default 0.5
        
    Returns
    -------
    tuple
        (data_array, transform, extent)
    """
    print(f"Mosaicking {len(file_list)} tiles...")
    src_files = [rasterio.open(f) for f in file_list]
    data, transform = merge(src_files)
    data = data[0]  # Get first band
    
    # Close source files
    for src in src_files:
        src.close()
    
    # Get extent from transform
    height, width = data.shape
    extent = [
        transform.c,  # left
        transform.c + transform.a * width,  # right
        transform.f + transform.e * height,  # bottom
        transform.f  # top
    ]
    
    print(f"Combined data shape: {data.shape}")
    
    return data, transform, extent


def mask_raster_to_boundary(data, transform, boundary_gdf, threshold=0.5):
    """
    Mask raster data to a boundary and filter low values.
    
    Parameters
    ----------
    data : numpy.ndarray
        Raster data array
    transform : affine.Affine
        Affine transform for the raster
    boundary_gdf : geopandas.GeoDataFrame
        Boundary geometries to mask to
    threshold : float, optional
        Minimum value threshold, by default 0.5
        
    Returns
    -------
    numpy.ndarray
        Masked and filtered data array
    """
    # Ensure CRS matches
    if boundary_gdf.crs != 'EPSG:4326':
        boundary_wgs84 = boundary_gdf.to_crs('EPSG:4326')
    else:
        boundary_wgs84 = boundary_gdf
    
    # Get geometries
    shapes = [geom for geom in boundary_wgs84.geometry]
    
    # Create mask
    mask = geometry_mask(
        shapes,
        transform=transform,
        invert=True,
        out_shape=data.shape
    )
    
    # Apply mask
    data_masked = np.where(mask, data, np.nan)
    
    # Filter out near-zero values
    data_filtered = np.where(data_masked > threshold, data_masked, np.nan)
    
    return data_filtered


def calculate_common_scale(data_list, threshold=0.5, percentile=99):
    """
    Calculate common scale for multiple datasets.
    
    Parameters
    ----------
    data_list : list of numpy.ndarray
        List of data arrays
    threshold : float, optional
        Minimum value for scale, by default 0.5
    percentile : int, optional
        Percentile for maximum value, by default 99
        
    Returns
    -------
    tuple
        (vmin, vmax)
    """
    # Combine all non-NaN values
    combined = np.concatenate([d[~np.isnan(d)] for d in data_list])
    
    vmin = threshold
    vmax = np.nanpercentile(combined, percentile)
    
    return vmin, vmax


def plot_raster_comparison(data1, data2, extent1, extent2, 
                          title1, title2, main_title, 
                          vmin, vmax, 
                          boundary_gdf=None, 
                          boundary_kwargs=None,
                          xlim=None, ylim=None,
                          figsize=(20, 10), 
                          cmap='viridis',
                          source_text='Source: NASA Black Marble VNP46A3 Collection-2'):
    """
    Create side-by-side comparison plot of two rasters.
    
    Parameters
    ----------
    data1, data2 : numpy.ndarray
        Raster data arrays
    extent1, extent2 : list
        Extents [left, right, bottom, top] for each raster
    title1, title2 : str
        Titles for each subplot
    main_title : str
        Main figure title
    vmin, vmax : float
        Color scale limits
    boundary_gdf : geopandas.GeoDataFrame, optional
        Boundary to overlay
    boundary_kwargs : dict, optional
        Styling for boundary overlay
    xlim : tuple, optional
        X-axis limits (longitude)
    ylim : tuple, optional
        Y-axis limits (latitude)
    figsize : tuple, optional
        Figure size, by default (20, 10)
    cmap : str, optional
        Colormap name, by default 'viridis'
    source_text : str, optional
        Source attribution text
        
    Returns
    -------
    tuple
        (fig, axes)
    """
    if boundary_kwargs is None:
        boundary_kwargs = {'color': 'white', 'linewidth': 1.5, 'alpha': 0.8}
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot first raster
    im1 = axes[0].imshow(data1, cmap=cmap, extent=extent1, vmin=vmin, vmax=vmax)
    axes[0].set_title(title1, fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    
    # Add boundary if provided
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=axes[0], **boundary_kwargs)
    
    # Set limits if provided
    if xlim is not None:
        axes[0].set_xlim(xlim)
    if ylim is not None:
        axes[0].set_ylim(ylim)
    
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Radiance')
    
    # Plot second raster
    im2 = axes[1].imshow(data2, cmap=cmap, extent=extent2, vmin=vmin, vmax=vmax)
    axes[1].set_title(title2, fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    
    # Add boundary if provided
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=axes[1], **boundary_kwargs)
    
    # Set limits if provided
    if xlim is not None:
        axes[1].set_xlim(xlim)
    if ylim is not None:
        axes[1].set_ylim(ylim)
    
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Radiance')
    
    # Add overall title and source
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98)
    fig.text(0.5, 0.02, source_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    return fig, axes


def print_raster_statistics(data1, data2, label1, label2):
    """
    Print comparison statistics for two raster datasets.
    
    Parameters
    ----------
    data1, data2 : numpy.ndarray
        Raster data arrays
    label1, label2 : str
        Labels for each dataset
    """
    mean1 = np.nanmean(data1)
    mean2 = np.nanmean(data2)
    max1 = np.nanmax(data1)
    max2 = np.nanmax(data2)
    
    change_pct = ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0
    
    print(f"\n--- Statistics ---")
    print(f"{label1}: Mean={mean1:.2f}, Max={max1:.2f}")
    print(f"{label2}: Mean={mean2:.2f}, Max={max2:.2f}")
    print(f"Change: {change_pct:.2f}%")
