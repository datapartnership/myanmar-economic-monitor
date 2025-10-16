# Datasets and Data Products Summary

## Datasets

**Datasets** refer to **all** datasets used in the analytics prepared for a project. The Datasets table includes a description of the data and their update frequency, as well as access links and contact information for questions about use and access. Users should not require any datasets not included in this table to complete the analytical work for the Data Good.

Following is list of all Datasets used in this Data Good:

```{note}
**Project Sharepoint** links are only accessible to the project team. For permissions to access these data, please write to the contact provided. The **Development Data Hub** is the World Bank's central data catalogue and includes meta-data and license information.
```

Where feasible, all datasets that can be obtained through the Development Data Hub have been placed in a special collection: *forthcoming*

| ID  | Name | License | Description | Update Frequency | Access | Contact |
| --- | ---- | ------- | ----------- | ---------------- | ------ | ------- |
| 1   |   Myanmar Admin Boundaries    |   Open      |         Admin boundaries datasets obtained from the Myanmar Govt Open Datasets upto Ward level    |         Published Dec 28, 2022. Update freuency unavailable         |  [Myanmar Information Management Unit Geospatial Data](https://geonode.themimu.info/layers/?limit=100&offset=0)      |   [Sahiti Sarva](mailto:ssarva@worldbank.org), Data Lab      |
| 2     | Normalized Difference Vegetation Index (NDVI) | Open                   | Normalized Difference Vegetation Index datasets sourced from MODIS, used to measure change in agricultural production | Monthly                                                           | [European Commission, Anomaly Hotspots of Agricultural Production](https://mars.jrc.ec.europa.eu/asap/country.php?cntry=238) | [Benny Istanto](mailto:bistanto@worldbank.org), GOST                        |
| 3      | ACLED Conflict Data                           | Open                   | Timestamped, geolocated points where conflict took place collected based on news and   crowdsourced data              | Daily                                                             | [Project SharePoint](https://worldbankgroup.sharepoint.com.mcas.ms/teams/DevelopmentDataPartnershipCommunity-WBGroup/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=Yvwh8r&cid=fccdf23e%2D94d5%2D48bf%2Db75d%2D0af291138bde&FolderCTID=0x012000CFAB9FF0F938A64EBB297E7E16BDFCFD&id=%2Fteams%2FDevelopmentDataPartnershipCommunity%2DWBGroup%2FShared%20Documents%2FProjects%2FData%20Lab%2FMyanmar%20Economic%20Monitor%2FData%2FACLED&viewid=80cdadb3%2D8bb3%2D47ae%2D8b18%2Dc1dd89c373c5); [Development Data Hub](https://datacatalog.worldbank.org/int/search/dataset/0061835/acled---middle-east)                                              | [Sahiti Sarva](mailto:ssarva@worldbank.org), GOST                        |
| 4      | Nighttime Lights                              | Open                   | Nighttime lights from VIIRS/Black Marble                                                                              | Monthly (daily possible)                                          | [Project SharePoint](https://worldbankgroup.sharepoint.com.mcas.ms/teams/DevelopmentDataPartnershipCommunity-WBGroup/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=Yvwh8r&cid=fccdf23e%2D94d5%2D48bf%2Db75d%2D0af291138bde&FolderCTID=0x012000CFAB9FF0F938A64EBB297E7E16BDFCFD&id=%2Fteams%2FDevelopmentDataPartnershipCommunity%2DWBGroup%2FShared%20Documents%2FProjects%2FData%20Lab%2FMyanmar%20Economic%20Monitor%2FData%2FNighttime%20Lights%20BlackMarble&viewid=80cdadb3%2D8bb3%2D47ae%2D8b18%2Dc1dd89c373c5); [Development Data Hub](https://datacatalog.worldbank.org/int/data/dataset/0063879/syria__night_time_lights)                                              | [Benjamin Stewart](mailto:bstewart@worldbank.org), GOST; [Robert Marty](mailto:rmarty@worldbank.org),   DIME |
| 5   |   Gross Domestic Product   |         |    Quarterly GDP data at a national level         |       Quarterly from 2015 till date           |   [Project SharePoint](https://worldbankgroup.sharepoint.com.mcas.ms/teams/DevelopmentDataPartnershipCommunity-WBGroup/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=Yvwh8r&cid=fccdf23e%2D94d5%2D48bf%2Db75d%2D0af291138bde&FolderCTID=0x012000CFAB9FF0F938A64EBB297E7E16BDFCFD&id=%2Fteams%2FDevelopmentDataPartnershipCommunity%2DWBGroup%2FShared%20Documents%2FProjects%2FData%20Lab%2FMyanmar%20Economic%20Monitor%2FData%2FGDP%2FRawData&viewid=80cdadb3%2D8bb3%2D47ae%2D8b18%2Dc1dd89c373c5)     |         |

## Data Products Summary

**Data Products** are produced using the **Foundational Datasets** and can be further used to generate indicators and insights. All Data Products include documentation, references to original data sources (and/or information on how to access them), and a description of their limitations.

Following is a summary of Data Products used in this Data Good:

| ID  | Name | Description | Limitations | Foundational Datasets Used (ID#) |
| --- | ---- | ----------- | ----------- | -------------------------------- |
| A   | Changes in Observed Nighttime Lights     |    Yearly and monthly aggregated remote   sensing data used to derive changes in observed nighttime lights         |      Results reflect data from NASA’s nighttime lights data. Gas Flaring is accounted for, however, the lights data also depends on cloud cover       |               1,4,5                   |
| B   |   Change in Agricultural Production    |    Monthly change in Normalized Difference   in Vegetation Index as a proxy to measure change in agricultural production         |      The statistics are based on global datasets to flag warnings.       |         2                         |
| C   |  Change in Conflict    |     ACLED dataset is used to observe changes in conflict by latitude and longitude over time        |      This is a user generated dataset combined with media reported incidents       |                      1,3,5            |
