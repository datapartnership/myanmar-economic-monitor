# Myanmar Economic Monitor

# Filepaths --------------------------------------------------------------------
db_dir  <- file.path("~", "Dropbox", "World Bank", "Side Work", "Myanmar Economic Monitor")
git_dir <- file.path("~", "Documents", "Github", "myanmar-economic-monitor")

data_dir      <- file.path(db_dir, "data")
ntl_bm_dir    <- file.path(data_dir, "Nighttime Lights BlackMarble")
gas_flare_dir <- file.path(data_dir, "Global Gas Flaring")
gadm_dir      <- file.path(data_dir, "GADM")
gdp_dir       <- file.path(data_dir, "GDP")
wdi_dir       <- file.path(data_dir, "WDI")

fig_dir <- file.path(git_dir, "reports", "figures")

# Functions --------------------------------------------------------------------
pad3 <- function(x){
  if(nchar(x) == 1) out <- paste0("00", x)
  if(nchar(x) == 2) out <- paste0("0", x)
  if(nchar(x) == 3) out <- paste0(x)
  return(out)
}
pad3 <- Vectorize(pad3)

pad2 <- function(x){
  if(nchar(x) == 1) out <- paste0("0", x)
  if(nchar(x) == 2) out <- paste0(x)
  return(out)
}
pad2 <- Vectorize(pad2)

# Packages ---------------------------------------------------------------------
library(dplyr)
library(readr)
library(janitor)
library(raster)
library(sf)
library(stringr)
library(readxl)
library(exactextractr)
#library(blackmarbler)
library(geodata)
library(lubridate)
library(rgeos)
library(leaflet)
library(leaflet.extras)
library(ggplot2)
library(ggpubr)
library(tidyr)
library(purrr)
library(stargazer)
library(furrr)
library(stringr)
library(rhdf5)
library(raster)
library(dplyr)
library(sf)
library(lubridate)
library(WDI)
library(broom)
library(blackmarbler)

# Run Scripts ------------------------------------------------------------------

if(F){

  ntl_git_dir <- file.path(git_dir, "notebooks", "nighttime-lights")

  #### Cleaning
  source(file.path(ntl_git_dir, "01_clean_gas_flaring_data.R"))
  source(file.path(ntl_git_dir, "01_download_gadm.R"))
  source(file.path(ntl_git_dir, "02_download_black_marble.R"))
  source(file.path(ntl_git_dir, "03_aggregate.R"))
  source(file.path(ntl_git_dir, "04_append.R"))

  #### Analysis
  source(file.path(ntl_git_dir, "05_adm_trends.R"))
  source(file.path(ntl_git_dir, "05_comparison_2019_maps.R"))
  source(file.path(ntl_git_dir, "05_map_ntl_annual_separate.R"))
  source(file.path(ntl_git_dir, "05_map_ntl_annual_together.R"))

  source(file.path(ntl_git_dir, "05_gas_flaring_map.R"))

}
