import os
import glob
import re
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import requests
from pathlib import Path
import time

# BlackMarble data processing class
class BlackMarbleProcessor:
    def __init__(self, input_dir, output_dir):
        """
        Initialize BlackMarble processor class
        
        Parameters:
        -----------
        input_dir : str
            Directory with h5 files
        output_dir : str
            Output directory for TIFs
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all h5 files
        self.h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
        print(f"Found {len(self.h5_files)} h5 files")
        
        # Check existing TIF files to avoid redundant processing
        self.existing_tifs = set([os.path.splitext(os.path.basename(f))[0] 
                            for f in glob.glob(os.path.join(output_dir, "*.tif"))])
        print(f"Found {len(self.existing_tifs)} existing TIF files")
        
        # Attempt to load BlackMarble tiles information (similar to BlackMarbler's approach)
        try:
            self.bm_tiles = self._load_bm_tiles()
        except Exception as e:
            print(f"Could not load BlackMarble tiles information: {str(e)}")
            self.bm_tiles = None

    def _load_bm_tiles(self):
        """
        Load BlackMarble tiles information (similar to what BlackMarbler does)
        """
        # This would ideally load from the same source as BlackMarbler, but for now just return None
        return None

    def _remove_fill_value(self, data, variable, fill_value=255):
        """
        Remove fill values from data based on variable type
        Similar to the remove_fill_value function in BlackMarbler
        
        Parameters:
        -----------
        data : numpy.ndarray
            The data array
        variable : str
            Variable name
        fill_value : int, optional
            Default fill value
            
        Returns:
        --------
        numpy.ndarray
            Data with fill values replaced by NaN
        """
        # List of variables using 255 as fill value
        fill_value_255 = [
            "Granule",
            "Mandatory_Quality_Flag",
            "Latest_High_Quality_Retrieval",
            "Snow_Flag",
            "DNB_Platform",
            "Land_Water_Mask",
            "AllAngle_Composite_Snow_Covered_Quality",
            "AllAngle_Composite_Snow_Free_Quality",
            "NearNadir_Composite_Snow_Covered_Quality",
            "NearNadir_Composite_Snow_Free_Quality",
            "OffNadir_Composite_Snow_Covered_Quality",
            "OffNadir_Composite_Snow_Free_Quality",
            "Gap_Filled_DNB_BRDF-Corrected_NTL",
        ]
        
        # List of variables using -999.9 as fill value
        fill_value_negative = ["UTC_Time"]
        
        # List of variables using -32768 as fill value
        fill_value_negative_32768 = [
            "Sensor_Azimuth",
            "Sensor_Zenith",
            "Solar_Azimuth",
            "Solar_Zenith",
            "Lunar_Azimuth",
            "Lunar_Zenith",
            "Glint_Angle",
            "Moon_Illumination_Fraction",
            "Moon_Phase_Angle"
        ]
        
        # List of variables using 65535 as fill value
        fill_value_65535 = [
            "DNB_At_Sensor_Radiance_500m",
            "BrightnessTemperature_M12",
            "BrightnessTemperature_M13",
            "BrightnessTemperature_M15",
            "BrightnessTemperature_M16",
            "QF_Cloud_Mask",
            "QF_DNB",
            "QF_VIIRS_M10",
            "QF_VIIRS_M11",
            "QF_VIIRS_M12",
            "QF_VIIRS_M13",
            "QF_VIIRS_M15",
            "QF_VIIRS_M16",
            "Radiance_M10",
            "Radiance_M11",
            "QF_Cloud_Mask",
            "DNB_BRDF-Corrected_NTL",
            "DNB_Lunar_Irradiance",
            
            "AllAngle_Composite_Snow_Covered",
            "AllAngle_Composite_Snow_Covered_Num",
            "AllAngle_Composite_Snow_Free",
            "AllAngle_Composite_Snow_Free_Num",
            "NearNadir_Composite_Snow_Covered",
            "NearNadir_Composite_Snow_Covered_Num",
            "NearNadir_Composite_Snow_Free",
            "NearNadir_Composite_Snow_Free_Num",
            "OffNadir_Composite_Snow_Covered",
            "OffNadir_Composite_Snow_Covered_Num",
            "OffNadir_Composite_Snow_Free",
            "OffNadir_Composite_Snow_Free_Num",
            "AllAngle_Composite_Snow_Covered_Std",
            "AllAngle_Composite_Snow_Free_Std",
            "NearNadir_Composite_Snow_Covered_Std",
            "NearNadir_Composite_Snow_Free_Std",
            "OffNadir_Composite_Snow_Covered_Std",
            "OffNadir_Composite_Snow_Free_Std"
        ]
        
        # Create a copy of the data to avoid modifying the original
        data_cleaned = data.copy()
        
        # Apply appropriate fill value replacement based on variable type
        if variable in fill_value_255:
            data_cleaned[data_cleaned == 255] = np.nan
        elif variable in fill_value_negative:
            data_cleaned[data_cleaned == -999.9] = np.nan
        elif variable in fill_value_negative_32768:
            data_cleaned[data_cleaned == -32768] = np.nan
        elif variable in fill_value_65535 or fill_value == 65535:
            data_cleaned[data_cleaned == 65535] = np.nan
            
        # All other values, including other negative values, are preserved as-is
        # This matches the R code's approach, which only removes specific fill values
            
        return data_cleaned

    def _apply_scaling_factor(self, data, variable):
        """
        Apply scaling factor to variables according to Black Marble user guide
        Similar to the apply_scaling_factor function in BlackMarbler
        
        Parameters:
        -----------
        data : numpy.ndarray
            The data array
        variable : str
            Variable name
            
        Returns:
        --------
        numpy.ndarray
            Scaled data
        """
        # List of variables that would need scaling in pre-Collection 2 data
        scaling_variables = [
            # VNP46A1
            "DNB_At_Sensor_Radiance",
            
            # VNP46A2
            "DNB_BRDF-Corrected_NTL",
            "Gap_Filled_DNB_BRDF-Corrected_NTL",
            "DNB_Lunar_Irradiance",
            
            # VNP46A3/4
            "AllAngle_Composite_Snow_Covered",
            "AllAngle_Composite_Snow_Covered_Std",
            "AllAngle_Composite_Snow_Free",
            "AllAngle_Composite_Snow_Free_Std",
            "NearNadir_Composite_Snow_Covered",
            "NearNadir_Composite_Snow_Covered_Std",
            "NearNadir_Composite_Snow_Free",
            "NearNadir_Composite_Snow_Free_Std",
            "OffNadir_Composite_Snow_Covered",
            "OffNadir_Composite_Snow_Covered_Std",
            "OffNadir_Composite_Snow_Free",
            "OffNadir_Composite_Snow_Free_Std"
        ]
        
        # For Collection 2 data, NO scaling is needed as explicitly noted in the BlackMarbler code:
        # "# Not needed with Collection 2"
        
        # Return data without scaling
        return data.copy()

    def _extract_tile_info(self, file_name):
        """
        Extract tile information from file name
        
        Parameters:
        -----------
        file_name : str
            H5 file name
            
        Returns:
        --------
        tuple
            h, v tile indices
        """
        tile_match = re.search(r'h(\d+)[vx](\d+)', file_name)
        if not tile_match:
            raise ValueError(f"Could not extract tile info from {file_name}")
            
        h = int(tile_match.group(1))
        v = int(tile_match.group(2))
        
        return h, v

    def _calculate_bounds(self, h, v):
        """
        Calculate bounds from tile indices
        
        Parameters:
        -----------
        h : int
            h tile index
        v : int
            v tile index
            
        Returns:
        --------
        tuple
            left, top, right, bottom bounds
        """
        # Calculate bounds based on tile indices
        left = -180.0 + (h * 10.0)
        top = 90.0 - (v * 10.0)
        right = left + 10.0
        bottom = top - 10.0
        
        return left, top, right, bottom

    def _find_dataset(self, h5_file):
        """
        Find appropriate dataset in h5 file
        
        Parameters:
        -----------
        h5_file : str
            Path to h5 file
            
        Returns:
        --------
        tuple
            data, variable, scale_factor, fill_value
        """
        data = None
        variable = None
        scale_factor = 1.0  # Default scaling factor
        fill_value = 255  # Default fill value
        
        # List of possible dataset paths to try, prioritizing the Gap_Filled_DNB_BRDF-Corrected_NTL dataset
        # Similar to BlackMarbler's logic in file_to_raster function
        dataset_paths = [
            'HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL',
            'HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_BRDF-Corrected_NTL',
            'Gap_Filled_DNB_BRDF-Corrected_NTL',
            'DNB_BRDF-Corrected_NTL',
            'HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/NearNadir_Composite_Snow_Free',
            'NearNadir_Composite_Snow_Free'
        ]
        
        try:
            # Try opening with different options
            for mode in ['r', 'r+']:
                if data is not None:
                    break
                    
                try:
                    with h5py.File(h5_file, mode) as f:
                        # Try each dataset path
                        for path in dataset_paths:
                            try:
                                if path in f:
                                    data = f[path][:]
                                    variable = path.split('/')[-1]
                                    
                                    # Try to get scale factor and fill value
                                    try:
                                        scale_factor = f[path].attrs.get('scale_factor', 1.0)
                                    except:
                                        pass
                                        
                                    try:
                                        fill_value = f[path].attrs.get('_FillValue', 65535)
                                    except:
                                        pass
                                        
                                    break
                            except Exception as e:
                                pass
                        
                        # If we didn't find a dataset through direct paths, try searching
                        if data is None:
                            datasets = []
                            
                            def collect_datasets(name, obj):
                                if isinstance(obj, h5py.Dataset) and len(obj.shape) == 2:
                                    datasets.append((name, obj.shape))
                            
                            f.visititems(collect_datasets)
                            
                            # Try to find a dataset with a shape that looks like a raster
                            for name, shape in datasets:
                                if shape[0] > 1000 and shape[1] > 1000:  # Likely a raster
                                    try:
                                        data = f[name][:]
                                        variable = name.split('/')[-1]
                                        
                                        # Try to get scale factor and fill value
                                        try:
                                            scale_factor = f[name].attrs.get('scale_factor', 1.0)
                                        except:
                                            pass
                                            
                                        try:
                                            fill_value = f[name].attrs.get('_FillValue', 65535)
                                        except:
                                            pass
                                            
                                        break
                                    except Exception:
                                        pass
                except Exception as e:
                    print(f"Error opening {h5_file} in {mode} mode: {str(e)}")
        except Exception as e:
            print(f"Error processing {h5_file}: {str(e)}")
            
        return data, variable, scale_factor, fill_value

    def process_file(self, h5_file):
        """
        Process a single h5 file and convert to GeoTIFF
        
        Parameters:
        -----------
        h5_file : str
            Path to h5 file
            
        Returns:
        --------
        str or None
            Path to output TIFF file if successful, None otherwise
        """
        base_name = os.path.basename(h5_file)
        base_name_no_ext = os.path.splitext(base_name)[0]
        
        # Skip if TIF already exists
        if base_name_no_ext in self.existing_tifs:
            print(f"Skipping {base_name} - TIF already exists")
            return None
        
        print(f"Processing {base_name}")
        
        try:
            # Extract tile info
            h, v = self._extract_tile_info(base_name)
            
            # Calculate bounds
            left, top, right, bottom = self._calculate_bounds(h, v)
            
            # Find dataset
            data, variable, scale_factor, fill_value = self._find_dataset(h5_file)
            
            if data is None:
                print(f"Could not find appropriate dataset in {base_name}, skipping file")
                return None
            
            # Process data
            data = data.astype(np.float32)
            
            # Apply BlackMarbler's fill value and scaling logic
            data = self._remove_fill_value(data, variable, fill_value)
            data = self._apply_scaling_factor(data, variable)
            
            # The R code doesn't specially handle negative values other than specific fill values
            # but the zonal stats code is having issues with negative values
            
            # Option to handle negative values (not in the original BlackMarbler code)
            # Set very negative values (< -100) to NaN
            data[data < -100] = np.nan
            
            # Set remaining negative values to 0
            data[data < 0] = 0
            
            # Calculate resolution
            xres = (right - left) / data.shape[1]
            yres = (top - bottom) / data.shape[0]
            
            # Create transform
            transform = from_origin(left, top, xres, yres)
            
            # Output filename
            output_file = os.path.join(self.output_dir, f"{base_name_no_ext}.tif")
            
            # Write GeoTIFF
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=CRS.from_epsg(4326),
                transform=transform,
                nodata=np.nan
            ) as dst:
                dst.write(data, 1)
            
            print(f"Saved to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            return None

    def process_all(self):
        """
        Process all h5 files in the input directory
        
        Returns:
        --------
        list
            Paths to all output TIFF files
        """
        output_files = []
        
        for h5_file in self.h5_files:
            output_file = self.process_file(h5_file)
            if output_file:
                output_files.append(output_file)
                
        print("Processing complete!")
        return output_files


# Main function
def main():
    # Set your paths here
    input_dir = "../../../data/ntl/raw_daily_20250401/"  # Directory with h5 files
    output_dir = "../../../data/ntl/myanmar_nightlights_tif"  # Output directory for TIFs
    
    # Create and run processor
    processor = BlackMarbleProcessor(input_dir, output_dir)
    processor.process_all()


if __name__ == "__main__":
    main()