# Nighttime Lights

Nighttime lights have become a commonly used resource to estimate changes in local economic activity. This section shows changes in nighttime lights in Turkiye from before and after the earthquake.

## Data

We use nighttime lights data from the VIIRS Black Marble dataset. Raw nighttime lights data requires correction due to cloud cover and stray light, such as lunar light. The Black Marble dataset applies advanced algorithms to correct raw nighttime light values and calibrate data so that trends in lights over time can be meaningfully analyzed. We use daily and monthly data from VIIRS Black Marble.

The data for the analysis can be accessed from:

* [Gas Flaring Location Data](https://datacatalog.worldbank.org/search/dataset/0037743)

* __Black Marble Nighttime Lights:__ There are two options to access the data:

  * The code [here](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/02_download_black_marble.R) downloads raw data from the [NASA archive](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A3/) and processes the data for Turkiye---mosaicing raster tiles together to cover Turkiye. Running the code requires a NASA bearer token; the documentation [here](https://github.com/ramarty/blackmarbler) describes how to obtain a token.

  * Pre-processed data can be downloaded from [here](URL), using the __Night Time Lights BlackMarble Data__

## Methodology

We extract average nighttime lights within each administrative unit in Mynamar. We distinguish lights between lights observed in gas flaring locations and lights in other locations. Oil extraction and production involves gas flaring, which produces significant volumes of light. Separately examining lights in gas flaring and other locations allows distinguishing between lights generated due to oil production versus other sources of human activity. We use data on the locations of gas flaring sites from the [Global Gas Flaring Reduction Partnership](https://www.worldbank.org/en/programs/gasflaringreduction); we remove lights within 5km of gas flaring sites.

## Implementation

Code to replicate the analysis can be found [here](https://github.com/datapartnership/myanmar-economic-monitor/tree/ntl/notebooks/nighttime-lights). 

The code largely relies on an R package (`blackmarbler`) that is currently being created to faciliate downloading and processing Black Marble nighttime lights data. The package [documentation](https://ramarty.github.io/blackmarbler/) provides some generic examples for how to download data, make a map of nighttime lights, and show trends in nighttime lights. The below code leverages the package to produce analytics for a country.

The main script ([_main.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/_main.R)) loads all packages and runs all scripts for the analysis. Below we document scripts for (1) creating analysis-ready datasets and (2) producing analytics (eg, figures) of nighttime lights data.

### Create Analysis-Ready Nighttime Lights Datasets

The below code downloads and processes nighttime lights data. 

* [01_clean_gas_flaring_data.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/01_clean_gas_flaring_data.R): Produces a clean dataset of the locations of gas flaring locations. This dataset is needed because we summarize nighttime lights for (1) all lights, (2) lights excluding gas flaring loations, and (3) lights only in gas flaring locations.
* [02_download_black_marble.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/02_download_black_marble.R): Downloads Black Marble nighttime lights data for annual, monthly, and daily nighttime lights data. The script exports a geotiff file of nighttime lights for each time period (eg, for each year for annual data). If the script is run at a later date, only data that has not already been downloaded will be downloaded.
* [03_aggregate.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/03_aggregate): Aggregates nighttime lights to administrative levels 0, 1 and 2. Calculates average nighttime lights within each administrative unit. The script produces a separate file for each time period. For example, for daily data, a single dataset is saved for aggregated nighttime lights for January 1, 2023, at the ADM2 level. The next script appends files across time periods (eg, days) into a single dataset. Separate files are saved in order to facilitate aggregating new data, as aggregating data can take some time. The script skips aggregating data that has already been processed.
* [04_append.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/04_append.R): Appends aggregated data for each time period. Creates a separate datasets for annual, monthly, and daily nighttime lights data.

### Nighttime Lights Analytics

The below code analyzes the nighttime lights data, producing figures and tables.

* [05_adm_trends.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/05_adm_trends.R): Produce figures of trends in nighttime lights
* [05_pc_2019_maps.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/05_pc_2019_maps.R): Produce maps showing percent change in nighttime lights relative to 2019
* [05_map_ntl_annual_separate.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/05_map_ntl_annual_separate.R): Produce maps of nighttime lights for each year, separately
* [05_map_ntl_annual_together.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/05_map_ntl_annual_together.R): Produce a figure showing maps of nighttime lights for each year available

## Findings

### Maps of Nighttime Lights

The below figures show maps of nighttime lights. The first figure shows nighttime lights in the latest year currently available (2022), and the second figure shows nighttime lights in each year.

```{figure} ../../reports/figures/ntl_bm_2022.png
---
scale: 50%
align: center
---
Nighttime lights in 2022
```

```{figure} ../../reports/figures/ntl_bm_allyears.png
---
scale: 50%
align: center
---
Nighttime lights
```

### Trends in Nighttime Lights

#### Annual

The below figures show trends in average nighttime lights.

```{figure} ../../reports/figures/ntl_trends_adm0.png
---
scale: 50%
align: center
---
Trends in Nighttime Lights at Country Level
```

```{figure} ../../reports/figures/ntl_trends_adm1.png
---
scale: 50%
align: center
---
Trends in Nighttime Lights at ADM 1
```

#### Monthly

```{figure} ../../reports/figures/ntl_trends_adm1_monthly.png
---
scale: 50%
align: center
---
Trends in Nighttime Lights at ADM 1, Monthly
```

The below figure shows trends in nighttime lights, showing trends in the z-score. The z-score is computed within each admin 1 unit.

```{figure} ../../reports/figures/ntl_trends_adm1_monthly_zscore.png
---
scale: 50%
align: center
---
Trends in Nighttime Lights at ADM 1 using Z-Score, Monthly
```

### Maps of Changes in Nighttime Lights

The below maps show the percent change in nighttime lights relative to 2019.

```{figure} ../../reports/figures/pc_2019_maps1.png
---
scale: 50%
align: center
---
Percent Change in Nighttime Lights: AMD 1 Level
```

```{figure} ../../reports/figures/pc_2019_maps2.png
---
scale: 50%
align: center
---
Percent Change in Nighttime Lights: AMD 2 Level
```

```{figure} ../../reports/figures/pc_2019_maps3.png
---
scale: 50%
align: center
---
Percent Change in Nighttime Lights: AMD 3 Level
```


## Limitations

Nighttime lights are a common data source for measuring local economic activity. However, it is a proxy that is strongly---although imperfectly---correlated with measures of interest, such as population, local GDP, and wealth. Consequently, care must be taken in interpreting reasons for changes in lights.


