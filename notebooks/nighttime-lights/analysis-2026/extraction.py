import os
from pathlib import Path
import geopandas as gpd
import pandas as pd

import numpy as np
from rasterstats import zonal_stats
from getpass import getpass
from dotenv import dotenv_values
from blackmarble import BlackMarble, Product
from blackmarble import raster, extract

# Get repo root directory (3 levels up from this script)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"

secrets_path = Path.home() / ".config" / "myanmar-economic-monitor" / "secrets.env"
secrets = dotenv_values(secrets_path)
blackmarble_token = secrets.get("BLACKMARBLE_TOKEN", "").strip()

if not blackmarble_token:
    blackmarble_token = getpass("Enter BlackMarble token (input hidden): ").strip()
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    secrets_path.write_text(f"BLACKMARBLE_TOKEN={blackmarble_token}\n")
    os.chmod(secrets_path, 0o600)

bm = BlackMarble(token=blackmarble_token)

mmr_adm0 = gpd.read_file(DATA_DIR / 'boundaries/mmr_polbnda_adm0_250k_mimu_20240215.shp')
mmr_adm1 = gpd.read_file(DATA_DIR / 'boundaries/mmr_polbnda_adm1_250k_mimu_20240215.shp')
mmr_adm2 = gpd.read_file(DATA_DIR / 'boundaries/mmr_polbnda_adm2_250k_mimu_20240215.shp')
mmr_adm3 = gpd.read_file(DATA_DIR / 'boundaries/mmr_polbnda_adm3_250k_mimu_20240215.shp')
mmr_adm4 = gpd.read_file(DATA_DIR / 'boundaries/mmr_polbnda_adm4_250k_mimu_20240215.shp')
mmr_sez = gpd.read_file(DATA_DIR / 'boundaries/industrial__special_economic_zones_sept2019.shp')
# Buffer SEZ points into 10km polygons so rasterstats can compute zonal stats
mmr_sez['geometry'] = mmr_sez.to_crs(epsg=32647).buffer(10000).to_crs(mmr_sez.crs)

# SEZ union geometries for industrial zone masking (5km and 10km)
mmr_sez_pts = gpd.read_file(DATA_DIR / 'boundaries/industrial__special_economic_zones_sept2019.shp')
sez_5km = mmr_sez_pts.to_crs(epsg=32647).buffer(5000).to_crs(mmr_sez_pts.crs)
sez_10km = mmr_sez_pts.to_crs(epsg=32647).buffer(10000).to_crs(mmr_sez_pts.crs)
sez_5km_union = sez_5km.union_all()
sez_10km_union = sez_10km.union_all()

gas_flaring = pd.read_csv(DATA_DIR / 'ntl/gas_flare_locations.csv')
gas_flaring_10km = gpd.GeoDataFrame(gas_flaring, geometry=gpd.points_from_xy(gas_flaring.longitude, gas_flaring.latitude), crs='EPSG:4326')
gas_flaring_10km['geometry'] = gas_flaring_10km.to_crs(epsg=32647).buffer(10000).to_crs(gas_flaring_10km.crs)

gas_flaring_5km = gpd.GeoDataFrame(gas_flaring, geometry=gpd.points_from_xy(gas_flaring.longitude, gas_flaring.latitude), crs='EPSG:4326')
gas_flaring_5km['geometry'] = gas_flaring_5km.to_crs(epsg=32647).buffer(5000).to_crs(gas_flaring_5km.crs)

# Dissolve gas flaring buffers into single union geometries
gf_5km_union = gas_flaring_5km.union_all()
gf_10km_union = gas_flaring_10km.union_all()


def make_mask_variants(gdf, mask_union):
    """Create masked and inverse-masked versions of admin boundaries."""
    gdf = gdf.copy()
    gdf['_merge_id'] = range(len(gdf))

    gdf_in = gdf.copy()
    gdf_in['geometry'] = gdf_in.geometry.intersection(mask_union)
    gdf_in = gdf_in[~gdf_in.is_empty].reset_index(drop=True)

    gdf_out = gdf.copy()
    gdf_out['geometry'] = gdf_out.geometry.difference(mask_union)
    gdf_out = gdf_out[~gdf_out.is_empty].reset_index(drop=True)

    return gdf, gdf_in, gdf_out


def extract_mask_variant(gdf_variant, rasters_ds, col_name):
    """Extract zonal stats from in-memory rasters for a mask variant."""
    if gdf_variant.empty:
        return None
    try:
        var_name = [v for v in rasters_ds.data_vars][0]
        da = rasters_ds[var_name]
        time_dim = 'time' if 'time' in da.dims else da.dims[0]
        records = []
        for t_idx in range(da.sizes[time_dim]):
            slice_da = da.isel({time_dim: t_idx})
            arr = slice_da.values.astype('float64')
            transform = slice_da.rio.transform()
            nodata = da.rio.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            stats = zonal_stats(
                gdf_variant, arr, affine=transform, stats=['sum'],
                nodata=np.nan, all_touched=True
            )
            date_val = pd.Timestamp(da[time_dim].values[t_idx])
            for i, s in enumerate(stats):
                records.append({
                    '_merge_id': gdf_variant.iloc[i]['_merge_id'],
                    'date': date_val,
                    col_name: s['sum'] if s['sum'] is not None else 0.0,
                })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"  Warning: skipping {col_name} — {e}")
        return None


mmr_adm0.drop(columns='date', inplace=True)
mmr_adm1.drop(columns='date', inplace=True)
mmr_adm2.drop(columns='date', inplace=True)
mmr_adm3.drop(columns='date', inplace=True)
mmr_adm4.drop(columns='date', inplace=True)
#mmr_sez.drop(columns='date', inplace=True)

start_date = "2012-01-01"
end_date_monthly = "2026-03-01"

end_date_annual = "2025-01-01"

for products in [Product.VNP46A3]:

    if products == Product.VNP46A4:
        print("Extracting VNP46A4 (annual composites)...")
        freq = "YS"
        folder = 'annual'
        end_date = end_date_annual

    else:
        print("Extracting VNP46A3 (monthly composites)...")
        freq = "MS"
        folder = 'monthly'
        end_date = end_date_monthly

    rasters = raster.bm_raster(
        mmr_adm0, 
        products,
        pd.date_range(start_date, end_date, freq=freq),
        token = blackmarble_token,
        output_directory=str(DATA_DIR / f'ntl/collection2/raw/{folder}'),
        output_skip_if_exists=True
    )

    out_dir = DATA_DIR / f'ntl/collection2/rasters/{folder}'
    out_dir.mkdir(parents=True, exist_ok=True)
    for var in rasters.data_vars:
        rasters[var].rio.to_raster(out_dir / f'{var}.tif')

    for admin_level, mmr_gdf in zip(
        [0, 1, 2, 3, 4, 'sez'],
        [mmr_adm0, mmr_adm1, mmr_adm2, mmr_adm3, mmr_adm4, mmr_sez]
    ):
        print(f"Extracting for admin level {admin_level}...")

        date_range = pd.date_range(start_date, end_date, freq=freq)
        extract_kwargs = dict(
            product_id=products,
            date_range=date_range,
            token=blackmarble_token,
            output_directory=str(DATA_DIR / f'ntl/collection2/rasters/{folder}/'),
            output_skip_if_exists=True,
        )

        # Create masked variants for all masks
        gdf_with_id, gdf_gf_5km, gdf_nogf_5km = make_mask_variants(mmr_gdf, gf_5km_union)
        _, gdf_gf_10km, gdf_nogf_10km = make_mask_variants(mmr_gdf, gf_10km_union)
        _, gdf_ind_5km, gdf_noind_5km = make_mask_variants(mmr_gdf, sez_5km_union)
        _, gdf_ind_10km, gdf_noind_10km = make_mask_variants(mmr_gdf, sez_10km_union)

        # Extract total NTL (ntl_sum)
        extracted = extract.bm_extract(gdf_with_id, **extract_kwargs)

        # --- Gas flaring 5km / 10km ---
        for gdf_in, gdf_out, prefix in [
            (gdf_gf_5km, gdf_nogf_5km, 'ntl_gf_5km'),
            (gdf_gf_10km, gdf_nogf_10km, 'ntl_gf_10km'),
            (gdf_ind_5km, gdf_noind_5km, 'ntl_ind_5km'),
            (gdf_ind_10km, gdf_noind_10km, 'ntl_ind_10km'),
        ]:
            col_in = f'{prefix}_sum' if 'gf' in prefix else f'{prefix}_sum'
            col_out = f'{prefix.replace("gf", "nogf").replace("ind", "noind")}_sum'

            extr_in = extract_mask_variant(gdf_in, rasters, col_in)
            if extr_in is not None:
                extracted = extracted.merge(extr_in, on=['_merge_id', 'date'], how='left')
            else:
                extracted[col_in] = 0.0

            extr_out = extract_mask_variant(gdf_out, rasters, col_out)
            if extr_out is not None:
                extracted = extracted.merge(extr_out, on=['_merge_id', 'date'], how='left')
            else:
                extracted[col_out] = 0.0

        # Fill NaN for admin areas with no mask overlap
        mask_cols = [
            'ntl_gf_5km_sum', 'ntl_nogf_5km_sum',
            'ntl_gf_10km_sum', 'ntl_nogf_10km_sum',
            'ntl_ind_5km_sum', 'ntl_noind_5km_sum',
            'ntl_ind_10km_sum', 'ntl_noind_10km_sum',
        ]
        for col in mask_cols:
            extracted[col] = extracted[col].fillna(0.0)

        out_path = DATA_DIR / f'ntl/collection2/{folder}/ntl_mmr_adm{admin_level}_{folder}.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        extracted.drop(columns=['geometry', '_merge_id', '_join_id'], errors='ignore').to_csv(out_path, index=False)