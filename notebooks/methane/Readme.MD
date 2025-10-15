Methane (CH₄) emissions have increasingly been studied as potential indicators of industrial and manufacturing activity, particularly due to their strong association with fossil fuel extraction, processing, and distribution, which are integral components of industrial output. Several studies have demonstrated that fluctuations in CH₄ levels can mirror changes in oil and gas operations, providing a measurable signal for industrial activity patterns. For instance, Barkley et al. (2021) showed that co-emissions of methane and ethane closely track oil and gas production levels, with ethane serving as a reliable proxy that distinguishes fossil fuel sources from other methane emitters, suggesting CH₄ can serve as a real-time indicator of energy-intensive industrial activity. Similarly, Veefkind et al. (2023) used satellite observations over the Permian Basin to identify methane emissions showing moderate spatial correlation with oil and gas production and drilling activities, with a significant fraction attributed to frequent widespread sources from oil and gas infrastructure, reinforcing the linkage between CH₄ concentrations and industrial operations. Luo et al. (2025) further found that spatial variations in China's methane emissions from oil and gas systems increased from 0.5 to 4.0 TgCH₄ per year between 1990 and 2022, reflecting shifts in industrial structure and energy infrastructure expansion driven by economic development and the energy transition. While methane is also emitted from agriculture and waste, studies focusing on industrial regions demonstrate that its atmospheric patterns can serve as a proxy for manufacturing intensity and energy-related economic activity. In this analysis, we explore trends in CH₄ emissions to assess their potential linkage with industrial activity.

Common sources of CH₄ 
Agriculture (Livestock and rice cultivation)
Fossil fuel operations (oil, gas, and coal)
Waste management (landfills and wastewater)
Biomass burning and bioenergy

CH₄ Data
CH₄ is a potent greenhouse gas with a global warming potential approximately 28-36 times greater than carbon dioxide over a 100-year period, and 80+ times greater over a 20-year period. While CH₄ has a shorter atmospheric lifetime than CO2 (around 12 years), its high warming potential makes it a critical target for near-term climate mitigation. Methane enters the atmosphere through both human activities (oil and gas production, coal mining, solid waste landfills, wastewater treatment, and livestock operations) and natural processes (wetlands, permafrost thaw, and wildfires).
Carbon Mapper focuses specifically on detecting and quantifying large point source emissions—individual facilities or infrastructure emitting methane at rates detectable from space or aircraft. These "super-emitters" can contribute 20-60% of total emissions from certain sectors and regions, making them high-priority targets for mitigation efforts.
The emission rate (kg/hr) is calculated using the Integrated Methane Enhancement (IME) method, which measures the total mass of methane in a plume above background concentrations. This measurement is combined with wind speed and direction data from atmospheric reanalysis models (HRRR, ERA5, or ECMWF_IFS) to estimate the instantaneous emission rate at the time of observation. Emission rates are typically reported in kilograms per hour (kg/hr), with associated uncertainties derived from measurement and wind field variability.
Carbon Mapper collects methane plume data using imaging spectrometers deployed on satellites and aircraft, including NASA's EMIT sensor aboard the International Space Station, Planet's Tanager satellites, NASA's AVIRIS-NG, and Arizona State University's Global Airborne Observatory. These hyperspectral sensors detect methane by measuring absorption of shortwave infrared light at wavelengths sensitive to CH4. Plumes are detected at spatial resolutions ranging from 3-8 meters (aircraft) to 30-50 meters (satellites), enabling precise attribution to individual facilities. All data undergoes quality control review by trained analysts and is made publicly available through the Carbon Mapper Data Portal and API.

Data Access
Dataset: Carbon Mapper Data Portal (https://data.carbonmapper.org/#1/-8.2/105)
Dataset Provider: Carbon Mapper (https://carbonmapper.org/)
Granularity: 30-60 meters spatial resolution (varies by sensor: Tanager satellites at 30-43m, EMIT at 50-60m)
Frequency: Daily to weekly (varies by platform and campaign)
License: Data are provided under Carbon Mapper’s Non-Commercial Data License (See full terms at License Details(https://carbonmapper.org/terms))
Data Access: The Project SharePoint currently hosts raw data queried from Google Earth Engine and can be accessed by all project team members.

References 
https://www.iea.org/reports/global-methane-tracker-2024/understanding-methane-emissions
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD034194
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JD037479
https://www.nature.com/articles/s41467-025-58237-z
https://earth.org/sources-of-methane-in-the-atmosphere/