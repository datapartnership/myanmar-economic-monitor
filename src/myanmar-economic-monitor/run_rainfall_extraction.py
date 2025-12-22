"""
Run CHIRPS rainfall data extraction for Myanmar
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from chirps_rainfall_extraction import CHIRPSRainfallExtractor


def main():
    """Extract rainfall data for Myanmar at ADM0 and ADM1 levels."""

    # Initialize extractor
    print("Initializing CHIRPS Rainfall Extractor...")
    extractor = CHIRPSRainfallExtractor(
        output_dir="../../data/rainfall", boundaries_dir="../../data/boundaries"
    )

    # Authenticate GEE
    print("\nAuthenticating Google Earth Engine...")
    extractor.authenticate_gee()

    # Set date range - let's get data from 2020 to 2024
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    print(f"\nExtracting rainfall data from {start_date} to {end_date}")
    print("=" * 60)

    # 1. Extract national-level (ADM0) rainfall
    print("\n1. Extracting national-level (ADM0) rainfall...")
    national_rainfall = extractor.get_rainfall_data(
        start_date=start_date, end_date=end_date, temporal_resolution="monthly"
    )

    # Save ADM0 data
    adm0_file = extractor.save_to_csv(
        national_rainfall, "myanmar_rainfall_adm0_monthly.csv"
    )
    print(f"✓ Saved ADM0 data: {adm0_file}")
    print(f"  Records: {len(national_rainfall)}")
    print(
        f"  Date range: {national_rainfall['date'].min()} to {national_rainfall['date'].max()}"
    )

    # 2. Extract regional-level (ADM1) rainfall
    print("\n2. Loading Myanmar ADM1 boundaries...")
    myanmar_adm1 = extractor.load_myanmar_boundaries(admin_level=1)
    print(f"✓ Loaded {len(myanmar_adm1)} regions")

    print("\n3. Extracting regional-level (ADM1) rainfall...")
    print("   This may take several minutes...")
    regional_rainfall = extractor.get_rainfall_by_region(
        start_date=start_date,
        end_date=end_date,
        regions_gdf=myanmar_adm1,
        admin_column="ADM1_EN",
        temporal_resolution="monthly",
    )

    # Save ADM1 data
    adm1_file = extractor.save_to_csv(
        regional_rainfall, "myanmar_rainfall_adm1_monthly.csv"
    )
    print(f"✓ Saved ADM1 data: {adm1_file}")
    print(f"  Records: {len(regional_rainfall)}")
    print(f"  Regions: {regional_rainfall['region'].nunique()}")
    print(
        f"  Date range: {regional_rainfall['date'].min()} to {regional_rainfall['date'].max()}"
    )

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"National data: {adm0_file}")
    print(f"Regional data: {adm1_file}")
    print("\nRegions included:")
    for region in sorted(regional_rainfall["region"].unique()):
        print(f"  - {region}")


if __name__ == "__main__":
    main()
