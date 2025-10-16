# Monitoring vegetation conditions

The significance of agricultural monitoring, particularly in countries like Myanmar, cannot be overstated. Myanmar's agriculture plays a crucial role in maintaining food security, providing livelihoods, and supporting the national economy. Continuous monitoring of vegetation dynamics can provide valuable insights into ecosystem health, early warnings of environmental degradation, and identify opportunities for sustainable land management practices.

![WorldCover](./images/climag-worldcover.png)

**Figure 1.** Myanmar land cover.

Remote sensing techniques, such as those employed through the use of Moderate Resolution Imaging Spectroradiometer ([MODIS](https://modis.gsfc.nasa.gov/about/)) Terra ([MOD13Q1](https://lpdaac.usgs.gov/products/mod13q1v061/)) and Aqua ([MYD13Q1](https://lpdaac.usgs.gov/products/myd13q1v061/)) Vegetation Indices 16-day L3 Global 250m time series data, have revolutionized the way we monitor agricultural conditions in Myanmar. By deriving variables such as ratio anomaly, difference anomaly, standardized anomaly, and vegetation condition index, these analyses enable the quantification of vegetation changes over time and across vast spatial extents.

Regular agricultural monitoring provides numerous benefits, including the ability to detect and mitigate the impacts of deforestation, land degradation, and desertification, as well as monitor crop health and inform agricultural decision-making. Furthermore, such monitoring can guide policymakers in the development of adaptive strategies and environmental policies, ensuring a sustainable future for Myanmar and its people.

Building on the foundation of monitoring agricultural conditions using MODIS MOD13Q1 and MYD13Q1 time series data, a more comprehensive understanding of vegetation dynamics in cropland areas can be achieved by incorporating phenological analysis. This is particularly important in a country like Myanmar, where agriculture is a vital sector for the economy and food security. Accurate and timely information on crop phenology can significantly enhance agricultural management, resource allocation, and the overall resilience of the farming sector.

To achieve this, the use of [TIMESAT](https://web.nateko.lu.se/timesat/timesat.asp), a software tool designed for the analysis of time series data, can be employed to extract critical phenological parameters such as the Start of Season (SOS), Mid of Season (MOS), and End of Season (EOS) from the Enhanced Vegetation Index (EVI) data. By first clipping the EVI data to the cropland extent, the analysis becomes more focused on the regions of interest, ensuring that the extracted parameters are directly relevant to agricultural practices.

This phenological information can then be used to guide farmers and agricultural stakeholders in Myanmar in making timely and informed decisions, such as when to plant, irrigate, or harvest their crops. Moreover, it can help identify potential threats to crop health and yield, such as disease outbreaks, pest infestations, or the effects of climate change, allowing for proactive and targeted interventions. Ultimately, incorporating phenological analysis into agricultural monitoring efforts provides a more holistic and actionable understanding of the complex dynamics that govern agricultural productivity and environmental sustainability in Myanmar.

## Data

In this study, we utilize a range of high-quality datasets to analyze vegetation conditions and phenology within Myanmars's cropland areas. The Data section introduces the sources of information employed, setting the stage for the methodologies and results presented later in the study. Our analysis incorporates EVI data from the MOD13Q1 and MYD13Q1 products and the cropland extent provided by the ESA WorldCover 2021 dataset. Combining these datasets allows for a focused assessment of the agricultural landscape, enabling a deeper understanding of the factors influencing agricultural productivity and environmental sustainability in the region.

### Crop extent

We used the Landsat-Derived Global Rainfed and Irrigated-Cropland Product ([LGRIP](https://lpdaac.usgs.gov/products/lgrip30v001/)) to mask out areas which aren't of interest in computing the EVI, i.e. built-up, water, forest, etc. As an extension of the Global Food Security-support Analysis Data ([GFSAD](https://www.usgs.gov/centers/western-geographic-science-center/science/global-food-security-support-analysis-data-30-m)) project, LGRIP maps the world’s agricultural lands by dividing them into irrigated and rainfed croplands, and calculates irrigated and rainfed areas for every country in the world. LGRIP data are produced using Landsat 8 time-series satellite sensor data for the 2014-2017 time period to create a nominal 2015 product.

The cropland class has value equal to `2` and `3`, which will be used within Google Earth Engine to generate the mask. There are two ways to download the LGRIP, via [Data Pool](https://e4ftl01.cr.usgs.gov/COMMUNITY/LGRIP30.001/) or [NASA Earthdata Search](https://search.earthdata.nasa.gov/search?q=C2592845930-LPDAAC_ECS).

![Cropland](./images/climag-cropland.png)

**Figure 2.** Cropland

### Vegetation Indices

The EVI from both MOD13Q1 and MYD13Q1 downloaded using Google Earth Engine ([GEE](https://earthengine.google.com/)) which involves some process:

* Combine the two 16-day composites into a synthethic synthetic 8-day composite containing data from both Aqua and Terra.

* Applying Quality Control Bitmask, keep only pixels with good quality.

* Applying cropland mask, keep only pixels identified as a cropland based on ESA WorldCover.

* Clipped for Myanmar and batch export the collection to Google Drive.

![VI](./images/climag-evi-202304.png)

**Figure 3.** Vegetation indices, April 2023.

## Products

We present a summary of the key derived variables employed in our analysis to monitor vegetation conditions and dynamics within Myanmar's cropland areas. These variables, which are derived from the EVI data and the cropland extent, enable a comprehensive understanding of the factors influencing agricultural productivity and environmental sustainability in the region.

```{note}
It is important to note that the original MODIS EVI data downloaded from sources such as USGS or GEE is provided in a range of 0-10,000 to minimize storage requirements and maintain data integrity. To convert these values to their actual EVI range of 0-1, it is typically necessary to multiply the data by a scale factor of 0.0001. However, in this study, we have opted to retain the unscaled values to conserve storage space and ensure efficient data processing.

Please be aware that the derived variables and any subsequent analysis conducted in this study will be based on the original, unscaled EVI values. The results and interpretations should be considered accordingly, acknowledging that the values presented are in the 0-10,000 range rather than the conventional 0-1 EVI range. This approach does not compromise the overall analysis, but users of the data and findings should be mindful of the difference in scale when comparing the results to other studies or datasets that use the standard EVI scaling.
```

### Vegetation-derived anomaly

This study focuses on analyzing various EVI derived products, such as the ratio, difference, standardized anomaly, and Vegetation Condition Index (VCI). These derived products provide valuable insights into vegetation health, vigor, and responses to environmental factors like climate change and land-use practices. By examining the EVI-derived products, we can assess vegetation dynamics and identify patterns and trends that may impact ecosystems and human livelihoods.

#### Ratio Anomaly

This variable represents the relative difference between the current vegetation condition and the long-term average, expressed as a ratio. By comparing current EVI values to historical trends, ratio anomalies can help identify areas experiencing significant deviations in vegetation health, which may indicate potential issues such as drought, disease, or pest infestation.

The anomaly is calculated based on percentage of the average

`"anomaly (%)" = 100 * ("evi" / "mean_evi")`

where `evi` is current EVI and `mean_evi` is the long-term average of EVI.

![RatioAnom](./images/climag-ratioanom-202304.png)

**Figure 4.** Ratio anomaly, April 2023.

#### Difference Anomaly

Similar to the ratio anomaly, the difference anomaly measures the absolute difference between the current vegetation condition and the long-term average. This variable can provide insights into the magnitude of deviations in vegetation health, aiding in the detection and prioritization of regions requiring intervention or further investigation.

The anomaly is calculated based on difference from the average

`"anomaly (%)" = "evi" - "mean_evi"`

where `evi` is current EVI and `mean_evi` is the long-term average of EVI.

![DiffAnom](./images/climag-diffanom-202304.png)

**Figure 5.** Difference anomaly, April 2023.

#### Standardized Anomaly

The standardized anomaly is a dimensionless measure that accounts for variations in the mean and standard deviation of the time series data, allowing for a more robust comparison of anomalies across different regions or time periods. This variable can help identify abnormal vegetation conditions that may warrant further analysis or management action.

The anomaly is calculated based on difference from the average and standard deviation.

`"anomaly (%)" = ("evi" - "mean_evi") / "std_evi"`

where `evi` is current EVI and `mean_evi` and `std_evi` is the long-term average and standard deviation of EVI.

![StdAnom](./images/climag-stdanom-202304.png)

**Figure 6.** Standardized anomaly, April 2023.

#### Vegetation Condition Index (VCI)

The Vegetation Condition Index (VCI) is a normalized index calculated from the Enhanced Vegetation Index (EVI) data, which ranges from 0 to 100, with higher values indicating better vegetation health. By providing a concise measure of the overall vegetation condition, the VCI enables the identification of trends and patterns in vegetation dynamics and the evaluation of the effectiveness of management interventions.

To calculate the VCI, the current EVI value is compared to its long-term minimum and maximum values using the following equation:

`"vci" = 100 * ("evi" - "min_evi") / ("max_evi" - "min_evi")`

where `evi` is current EVI and `min_evi` and `max_evi` is the long-term minimum and maximum of EVI.

![VCI](./images/climag-vci-202304.png)

**Figure 7.** Vegetation Condition Index, April 2023.

### Phenological metrics

In addition to the previously mentioned vegetation indices and derived products, this study also focuses on analyzing key phenological metrics, such as the Start of Season (SOS), Mid of Season (MOS), and End of Season (EOS). These metrics provide essential information on the timing and progression of the growing season, offering valuable insights into plant growth and development. By examining SOS, MOS, and EOS, we can assess the impacts of environmental factors, such as climate change and land-use practices, on vegetation dynamics. Furthermore, understanding these phenological patterns can help inform agricultural management strategies, conservation efforts, and policies for sustainable resource management.

#### Start of Season (SOS)

The SOS is a critical phenological metric that represents the onset of the growing season. By analyzing the timing of SOS, we can gain insights into how environmental factors, such as temperature and precipitation, impact vegetation growth and development. Understanding the SOS is vital for agricultural planning, as it can inform planting and harvesting schedules, irrigation management, and help predict potential yield outcomes.

This map displays the spatial distribution of the SOS for Season 1, indicating when vegetation begins its growth cycle across the study area.

![SOS1](./images/climag-sos1-2022.png)

**Figure 8.** Crop planting status - Season 1, 2022

This map represents the onset of the second growing season, illustrating the spatial distribution of the SOS for Season 2 across the study area.

![SOS2](./images/climag-sos2-2022.png)

**Figure 9.** Crop planting status - Season 2, 2022

#### Mid of Season (MOS)

The MOS is an essential phenological parameter that indicates the peak of the growing season. Examining MOS can provide valuable information about the overall health and productivity of vegetation, as it typically corresponds to the period of maximum photosynthetic activity and biomass accumulation. Assessing MOS trends can help identify changes in vegetation dynamics due to climate change, land-use practices, or other environmental stressors, and support the development of effective management strategies for agriculture and natural resources.

The MOS map for Season 1 illustrates the point in time when vegetation reaches its peak growth and productivity during the first growing season.

![MOS1](./images/climag-mos1-2022.png)

**Figure 10.** Time of middle of season - Season 1, 2022

Highlighting the peak growth and productivity during the second growing season, the MOS map for Season 2 provides insights into the vegetation dynamics during this period.

![MOS2](./images/climag-mos2-2022.png)

**Figure 11.** Time of middle of season - Season 2, 2022

#### End of Season (EOS)

The EOS is an important phenological marker signifying the conclusion of the growing season. Investigating the timing of EOS can reveal valuable information about the duration of the growing season and the overall performance of vegetation. EOS trends can help us understand how factors such as temperature, precipitation, and human-induced land-use changes impact ecosystems and their productivity. Information on EOS is also crucial for agricultural management, as it aids in planning harvest schedules and post-harvest activities, and supports the development of informed land management and conservation policies.

Depicting the conclusion of the growing period, the EOS map for Season 1 shows when vegetation growth ceases and senescence begins in the study area.

![EOS1](./images/climag-eos1-2022.png)

**Figure 12.** Crop harvest status - Season 1, 2022

The EOS map for Season 2 visualizes the conclusion of the second growing season, capturing the spatial pattern of vegetation senescence across the study area.

![EOS2](./images/climag-eos2-2022.png)

**Figure 13.** Crop harvest status - Season 2, 2022

### Tabular data

Time series data aggregate at admin level 2 from January 2002 - February 2023 for above variables are available at Sharepoint, accessible via this [link](https://worldbankgroup.sharepoint.com/:x:/r/teams/DevelopmentDataPartnershipCommunity-WBGroup/Shared%20Documents/Projects/Data%20Lab/Myanmar%20Economic%20Monitor/Data/MODIS%20VI%20250m/07_tables/mmr_phy_modis_evi_monthly_admin2_2002_2023.xlsx?d=w495b1112724243b68ce9939231f64ba6&csf=1&web=1&e=Fgoy6p).

Data aggregation process are done using ArcGIS [Zonal Statistics](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/zonal-statistics-as-table.htm).

Below is an example on how the number of areas for each growing stages can be easily calculated month-by-month from EVI derived products.

![GS2022](./images/climag-growingstages-2022.png)

**Figure 14.** Growing stages in 2022

## Implementation

We utilized GEE to acquire a time series of EVI data. The EVI data was then processed using the ArcPy library in ArcGIS to generate long-term statistics and derive various vegetation indices products. Following this, we employed the TIMESAT software to extract seasonality parameters from the processed vegetation data.

### Vegetation Indices and derivative product

In this study, we employed a three-step coding approach to analyze the time series EVI data and derive vegetation index products. The first step utilized GEE to efficiently batch download the time series EVI data.

* The code for downloading timeseries EVI in GEE: [gee-batch-export-mxd13q1.js](/notebooks/vegetation-conditions/gee-batch-export-mxd13q1.js)

Following this, an ArcPy script was executed to process the acquired data and calculate long-term statistics from the time series.

* The code for calculating long-term statistical value of EVI in Arcpy: [modis_8daystats.py](/notebooks/vegetation-conditions/modis_8daystats.py)

Lastly, another ArcPy script was employed to compute various vegetation index derived products, such as the ratio, difference, standardized anomaly, and VCI.

* The code for calculating derived EVI products in Arcpy: [modis_viproducts.py](/notebooks/vegetation-conditions/modis_viproducts.py)

### Phenological Metrics

The SOS, MOS, and EOS are typically derived from EVI. These metrics are calculated using various methods that identify critical points or thresholds in the vegetation index time series data. One common approach is the Timesat software, which fits functions (e.g., logistic, asymmetric Gaussian, Savitsky-Golay, Whittaker) to the time series data to identify these points. Here's a general overview of the approach:

1. Preprocessing: Detrend and smooth the vegetation index time series data to reduce noise.

2. Model Fitting: Fit a function (e.g., logistic, asymmetric Gaussian) to the smoothed time series data. The chosen function should adequately represent the seasonal pattern of vegetation growth and senescence.

3. Threshold Determination: Define thresholds for SOS, MOS, and EOS. These thresholds may be absolute values, or they may be based on a percentage of the maximum vegetation index value for the season (e.g., 20% for SOS and EOS, 100% for MOS).

4. Metric Calculation: Identify the points in the fitted function where the thresholds are met. These points correspond to the SOS, MOS, and EOS.

* SOS: The time step where the fitted function first exceeds the lower threshold, marking the start of significant vegetation growth.

* MOS: The time step where the fitted function reaches the maximum vegetation index value, indicating the peak of the growing season.

* EOS: The time step where the fitted function falls below the lower threshold again, signifying the end of significant vegetation growth.

Note that these metrics will depend on the choice of function, thresholds, and other methodological details. The equations for calculating SOS, MOS, and EOS may vary depending on the specific technique employed.

A how-to guideline on calculating the phenological metrics are available through **Seasonality_Parameters_Data_Extraction**

## Limitations and Assumptions

Getting VI data with good quality for all period are challenging (pixels covered with cloud, snow/ice, aerosol quantity, shadow) for optic data (MODIS). Cultivated area year by year are varies, due to MODIS data quality and crop type is not described, so the seasonal parameters are for general cropland.

At this point, the analysis is also limited to seasonal crops due to difficulty to capture the dynamics of perennial crops within a year. The value may not represent for smaller cropland and presented result are only based upon the most current available remote sensing data. As the climate phenomena is a dynamic situation, the current realities may differ from what is depicted in this document.

Ground check is necessary to ensure if satellite and field situation data are corresponding.

## Potential Application

Above products provides an important starting point for continuous monitoring of the crop planting status. Continuous monitoring could inform the following assessments:

1. How many districts are behind in planting? If there is a delay in some districts, and is planting acceleration necessary?
2. How many hectares are available for the next season?
3. Is the current harvest enough for domestic consumption?

Decision makers also need phonological data to decide on resource allocation issues or policy design:

1. Planting potential for the next months: assigning the distribution of agricultural inputs.
2. Mobilization of extension workers to monitor and implement mitigation strategies: adjustment of irrigation system in anticipation of drought or flood, pest control of infestation/disease to avoid crop failure, reservoir readiness for planting season.
3. Preparation of policy recommendations: assess ongoing situation, harvest estimate, price protection.

This information is necessary for both policy makers, farmers, and other agricultural actors (cooperatives, rural businesses). Negative consequences can be anticipated months ahead and resources can be prioritized on areas with higher risk or greater potential.
