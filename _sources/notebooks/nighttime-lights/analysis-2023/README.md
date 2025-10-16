# Nighttime Lights 2023 Analysis

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

The code largely relies on an R package (`blackmarbler`) that is currently being created to facilitate downloading and processing Black Marble nighttime lights data. The package [documentation](https://ramarty.github.io/blackmarbler/) provides some generic examples for how to download data, make a map of nighttime lights, and show trends in nighttime lights. The below code leverages the package to produce analytics for a country.

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

```{figure} ../../../reports/figures/ntl_bm_2022.png
---
scale: 50%
align: center
---
Nighttime lights in 2022
```

```{figure} ../../../reports/figures/ntl_bm_allyears.png
---
scale: 50%
align: center
---
Nighttime lights
```

### Trends in Nighttime Lights

#### Annual

The below figures show trends in average nighttime lights.

```{figure} ../../../reports/figures/ntl_trends_adm0.png
---
scale: 50%
align: center
---
Trends in Nighttime Lights at Country Level
```

```{figure} ../../../reports/figures/ntl_trends_adm1.png
---
scale: 50%
align: center
---
Trends in Nighttime Lights at ADM 1
```

#### Monthly

```{figure} ../../../reports/figures/ntl_trends_adm1_monthly.png
---
scale: 50%
align: center
---
Trends in Nighttime Lights at ADM 1, Monthly
```

The below figure shows trends in nighttime lights, showing trends in the z-score. The z-score is computed within each admin 1 unit.

```{figure} ../../../reports/figures/ntl_trends_adm1_monthly_zscore.png
---
scale: 50%
align: center
---
Trends in Nighttime Lights at ADM 1 using Z-Score, Monthly
```

### Maps of Changes in Nighttime Lights

The below maps show the percent change in nighttime lights relative to 2019.

```{figure} ../../../reports/figures/pc_2019_maps1.png
---
scale: 50%
align: center
---
Percent Change in Nighttime Lights: AMD 1 Level
```

```{figure} ../../../reports/figures/pc_2019_maps2.png
---
scale: 50%
align: center
---
Percent Change in Nighttime Lights: AMD 2 Level
```

```{figure} ../../../reports/figures/pc_2019_maps3.png
---
scale: 50%
align: center
---
Percent Change in Nighttime Lights: AMD 3 Level
```

### Gas Flaring Nighttime Lights

The below figure shows the location of inland gas flaring locations, which are concentrated within one location in Myanmar.

```{figure} ../../../reports/figures/ntl_bm_gf_2022.png
---
scale: 50%
align: center
---
Gas Flaring Map
```

The below figure shows trends in nighttime lights in gas flaring locations.

```{figure} ../../../reports/figures/ntl_gas_flaring_trends.png
---
scale: 50%
align: center
---
Trends in nighttime lights in gas flaring locations
```

### Special Economic Zones

The below figure shows trends in nighttime lights within special economic zones, averaged across special economic zones.

```{figure} ../../../reports/figures/sez_ntl_trends_avg.png
---
scale: 50%
align: center
---
Change in nighttime lights from 2012 to 2022 in Special Economic Zones, averaged across zones
```

The below figure shows trends in nighttime lights within special economic zones.

```{figure} ../../../reports/figures/sez_ntl_trends.png
---
scale: 50%
align: center
---
Change in nighttime lights from 2012 to 2022 in Special Economic Zones
```

```{figure} ../../../reports/figures/sez_ntl_trends_select_zones.png
---
scale: 50%
align: center
---
Change in nighttime lights from 2012 to 2022 in Select Special Economic Zones
```

```{figure} ../../../reports/figures/sez_ntl_map.png
---
scale: 50%
align: center
---
Change in nighttime lights from 2012 to 2022 in Special Economic Zones
```

### GDP vs Nighttime Lights: Annual

This section explores the association between GDP and nighttime lights using annual data. We use data since 2012, the start of the nighttime lights data.

```{figure} ../../../reports/figures/gdp_ntl_annual_scatter.png
---
scale: 50%
align: center
---
Association between Nighttime Lights and GDP
```

```{figure} ../../../reports/figures/gdp_ntl_annual_scatter_log.png
---
scale: 50%
align: center
---
Association between Nighttime Lights and GDP, logging both NTL and GDP
```

```{figure} ../../../reports/figures/gdp_ntl_annual_trends.png
---
scale: 50%
align: center
---
Trends in GDP and Nighttime Lights
```

```{figure} ../../../reports/figures/gdp_ntl_annual_trends_log.png
---
scale: 50%
align: center
---
Trends in GDP and Nighttime Lights, Logged
```

```{figure} ../../../reports/figures/gdp_ntl_annual_trends_pc.png
---
scale: 50%
align: center
---
Trends in GDP and Nighttime Lights: Percent Change since 2012
```

```{figure} ../../../reports/figures/gdp_ntl_annual_trends_pc_log.png
---
scale: 50%
align: center
---
Trends in GDP and Nighttime Lights: Percent Change since 2012 (Logged)
```

Below shows regressions results, regressing GDP on average nighttime lights.

<table style="text-align:center"><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="2"><em>Dependent variable:</em></td></tr>
<tr><td></td><td colspan="2" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><td>GDP (Billions)</td><td>log(GDP)</td></tr>
<tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">NTL</td><td>37.299<sup>**</sup></td><td></td></tr>
<tr><td style="text-align:left"></td><td>(11.661)</td><td></td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">log(NTL)</td><td></td><td>0.257<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(0.074)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">Constant</td><td>48.294<sup>***</sup></td><td>25.109<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(5.290)</td><td>(0.066)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>10</td><td>10</td></tr>
<tr><td style="text-align:left">R<sup>2</sup></td><td>0.561</td><td>0.601</td></tr>
<tr><td style="text-align:left">Adjusted R<sup>2</sup></td><td>0.506</td><td>0.551</td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td colspan="2" style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
</table>

### GDP vs Nighttime Lights: Quarterly

This section explores the association between GDP and nighttime lights using annual data. We use available quarterly data from 2015 to 2019.

```{figure} ../../../reports/figures/gdp_ntl_quarterly_scatter.png
---
scale: 50%
align: center
---
Association between Nighttime Lights and GDP
```

```{figure} ../../../reports/figures/gdp_ntl_quarterly_scatter_log.png
---
scale: 50%
align: center
---
Association between Nighttime Lights and GDP, logging both NTL and GDP
```

```{figure} ../../../reports/figures/gdp_ntl_quarterly_trends.png
---
scale: 50%
align: center
---
Trends in GDP and Nighttime Lights
```

```{figure} ../../../reports/figures/gdp_ntl_quarterly_trends_log.png
---
scale: 50%
align: center
---
Trends in GDP and Nighttime Lights, Logged
```

```{figure} ../../../reports/figures/gdp_ntl_quarterly_trends_pc.png
---
scale: 50%
align: center
---
Trends in GDP and Nighttime Lights: Percent Change since 2012
```

```{figure} ../../../reports/figures/gdp_ntl_quarterly_trends_pc_log.png
---
scale: 50%
align: center
---
Trends in GDP and Nighttime Lights: Percent Change since 2012 (Logged)
```

Below shows regressions results, regressing GDP on average nighttime lights.

<table style="text-align:center"><tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="4"><em>Dependent variable:</em></td></tr>
<tr><td></td><td colspan="4" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><td>GDP (Millions)</td><td>log(GDP)</td><td>GDP (Millions)</td><td>log(GDP)</td></tr>
<tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td><td>(3)</td><td>(4)</td></tr>
<tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">NTL</td><td>11,871,105.000<sup>*</sup></td><td></td><td>12,350,574.000<sup>**</sup></td><td></td></tr>
<tr><td style="text-align:left"></td><td>(5,917,835.000)</td><td></td><td>(5,727,722.000)</td><td></td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td></td><td></td></tr>
<tr><td style="text-align:left">log(NTL)</td><td></td><td>0.604<sup>**</sup></td><td></td><td>0.622<sup>**</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(0.291)</td><td></td><td>(0.280)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td></td><td></td></tr>
<tr><td style="text-align:left">2020 Onwards</td><td></td><td></td><td>-314,581.800</td><td>-0.024</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td>(2,636,518.000)</td><td>(0.143)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td></td><td></td></tr>
<tr><td style="text-align:left">Constant</td><td>9,085,884.000<sup>*</sup></td><td>16.816<sup>***</sup></td><td>8,685,543.000<sup>*</sup></td><td>16.819<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(5,062,361.000)</td><td>(0.084)</td><td>(4,908,431.000)</td><td>(0.082)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td><td></td><td></td></tr>
<tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Start Year</td><td>2012</td><td>2012</td><td>2012</td><td>2012</td></tr>
<tr><td style="text-align:left">End Year</td><td>2020</td><td>2020</td><td>2022</td><td>2022</td></tr>
<tr><td style="text-align:left">Observations</td><td>36</td><td>36</td><td>44</td><td>44</td></tr>
<tr><td style="text-align:left">R<sup>2</sup></td><td>0.106</td><td>0.113</td><td>0.103</td><td>0.110</td></tr>
<tr><td style="text-align:left">Adjusted R<sup>2</sup></td><td>0.080</td><td>0.087</td><td>0.060</td><td>0.066</td></tr>
<tr><td colspan="5" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td colspan="4" style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
</table>

## Limitations

Nighttime lights are a common data source for measuring local economic activity. However, it is a proxy that is strongly---although imperfectly---correlated with measures of interest, such as population, local GDP, and wealth. Consequently, care must be taken in interpreting reasons for changes in lights.
