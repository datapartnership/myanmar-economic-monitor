# Nighttime Lights 2024 Analysis

Nighttime lights have become a commonly used resource to estimate changes in local economic activity. This section shows changes in nighttime lights in Myanmar.

## Data

We use nighttime lights data from the VIIRS Black Marble dataset. Raw nighttime lights data requires correction due to cloud cover and stray light, such as lunar light. The Black Marble dataset applies advanced algorithms to correct raw nighttime light values and calibrate data so that trends in lights over time can be meaningfully analyzed. We use annual data from VIIRS Black Marble.

## Methodology

We extract average nighttime lights within different units in Myanmar. We distinguish lights between lights observed in gas flaring locations and lights in other locations. Oil extraction and production involves gas flaring, which produces significant volumes of light. Separately examining lights in gas flaring and other locations allows distinguishing between lights generated due to oil production versus other sources of human activity. We use data on the locations of gas flaring sites from the [Global Gas Flaring Reduction Partnership](https://www.worldbank.org/en/programs/gasflaringreduction); we remove lights within 5km of gas flaring sites.

## Implementation

Code to download and aggregate nighttime lights data can be found [here](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/analysis-2024/01_process_data).

The main script ([_main.R](https://github.com/datapartnership/myanmar-economic-monitor/tree/main/notebooks/nighttime-lights/analysis-2024/_main.R)) loads all packages and runs all scripts for the analysis.

## Findings

### Special Economic Zones

```{figure} ../../../reports/figures/ntl_sez_avg.png
---
scale: 50%
align: center
---
Nighttime Lights
```

```{figure} ../../../reports/figures/ntl_sez_1.png
---
scale: 50%
align: center
---
Nighttime Lights
```

```{figure} ../../../reports/figures/ntl_sez_2.png
---
scale: 50%
align: center
---
Nighttime Lights
```

```{figure} ../../../reports/figures/ntl_sez_3.png
---
scale: 50%
align: center
---
Nighttime Lights
```

### Border Locations

```{figure} ../../../reports/figures/ntl_border_overall.png
---
scale: 50%
align: center
---
Nighttime Lights
```

```{figure} ../../../reports/figures/ntl_border_country.png
---
scale: 50%
align: center
---
Nighttime Lights
```

```{figure} ../../../reports/figures/ntl_border.png
---
scale: 50%
align: center
---
Nighttime Lights
```

### ADM 1 Units

```{figure} ../../../reports/figures/ntl_adm1.png
---
scale: 50%
align: center
---
Nighttime Lights
```

### Yangon

```{figure} ../../../reports/figures/ntl_adm1_yangon.png
---
scale: 50%
align: center
---
Nighttime Lights
```

```{figure} ../../../reports/figures/ntl_adm2_yangon.png
---
scale: 50%
align: center
---
Nighttime Lights
```

```{figure} ../../../reports/figures/ntl_adm3_yangon.png
---
scale: 50%
align: center
---
Nighttime Lights
```

### Nighttime lights and RWI

```{figure} ../../../reports/figures/rwi_ntl_cor.png
---
scale: 50%
align: center
---
Nighttime Lights and RWI Correlation
```

```{figure} ../../../reports/figures/rwi_q4_ntl_trends.png
---
scale: 50%
align: center
---
Nighttime Lights and RWI Trends
```

## Limitations

Nighttime lights are a common data source for measuring local economic activity. However, it is a proxy that is strongly---although imperfectly---correlated with measures of interest, such as population, local GDP, and wealth. Consequently, care must be taken in interpreting reasons for changes in nighttime lights.
