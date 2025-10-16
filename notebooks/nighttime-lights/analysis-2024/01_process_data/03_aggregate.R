# Aggregate Nighttime Lights

# Do if make changes to how data processed and need to start from scratch
DELETE_DIR <- F

if(DELETE_DIR){
  unlink(file.path(ntl_bm_dir, "FinalData", "aggregated"), recursive = T)
  dir.create(file.path(ntl_bm_dir, "FinalData", "aggregated"))
}

adm0_sf <- read_sf(file.path(data_dir, "Boundaries", "RawData", "mmr_polbnda_adm0_250k_mimu.shp"))

# Prep SEZ ---------------------------------------------------------------------
sez_sf <- read_sf(file.path(data_dir, "SEZ", "RawData", "industrial__special_economic_zones_sept2019.shp"))
sez_sf <- sez_sf %>% st_buffer(dist = 2500)

adm0_rm_sez_sf <- st_difference(adm0_sf, sez_sf %>% st_union() %>% st_make_valid())

# Prep Border ------------------------------------------------------------------
border_df <- read_xlsx(file.path(data_dir, "Border Towns", "RawData", "Myanmar's Border Town Coordinates.xlsx"))
border_sf <- border_df %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

border_1km_sf <- border_sf %>% st_buffer(dist = 1000)
border_2_5km_sf <- border_sf %>% st_buffer(dist = 2500)
border_5km_sf <- border_sf %>% st_buffer(dist = 5000)
border_10km_sf <- border_sf %>% st_buffer(dist = 10000)

adm0_rm_border_1km_sf   <- st_difference(adm0_sf, border_1km_sf %>% st_union())
adm0_rm_border_2_5km_sf <- st_difference(adm0_sf, border_2_5km_sf %>% st_union())
adm0_rm_border_5km_sf   <- st_difference(adm0_sf, border_5km_sf %>% st_union())
adm0_rm_border_10km_sf  <- st_difference(adm0_sf, border_10km_sf %>% st_union())

# Loop through ROIs ------------------------------------------------------------
for(adm_level in c("bound1", "bound2", "sez", "0", "1", "2", "3",
                   "border_1km", "border_2_5km", "border_5km", "border_10km",
                   #"rwi",
                   "0_rm_sez",
                   "0_rm_border_1km",
                   "0_rm_border_2_5km",
                   "0_rm_border_5km",
                   "0_rm_border_10km")){

  if(adm_level == "sez"){
    roi_sf <- sez_sf
  } else if(adm_level == "bound1"){
    roi_sf <- read_sf(file.path(data_dir, "Boundaries", "RawData", "mmr_polbnda_adm1_250k_mimu.shp"))
  } else if(adm_level == "bound2"){
    roi_sf <- read_sf(file.path(data_dir, "Boundaries", "RawData", "mmr_polbnda_adm2_250k_mimu_1.shp"))
  } else if(adm_level == "border_1km"){
    roi_sf <- border_1km_sf
  } else if(adm_level == "border_2_5km"){
    roi_sf <- border_2_5km_sf
  } else if(adm_level == "border_5km"){
    roi_sf <- border_5km_sf
  } else if(adm_level == "border_10km"){
    roi_sf <- border_10km_sf
  } else if(adm_level == "rwi"){

    source("https://raw.githubusercontent.com/ramarty/fast-functions/master/R/functions_in_chunks.R")
    rwi_df <- read_csv(file.path(data_dir, "Relative Wealth", "MMR_relative_wealth_index.csv"))
    rwi_sf <- st_as_sf(x = rwi_df, coords = c("longitude", "latitude"), crs = 4326)
    rwi_buff_sf <- rwi_sf %>% st_buffer_chunks(dist = 1200, chunk_size = 500)

    roi_sf <- rwi_buff_sf
  } else if(adm_level == "0_rm_sez"){
    roi_sf <- adm0_rm_sez_sf
  } else if(adm_level == "0_rm_border_1km"){
    roi_sf <- adm0_rm_border_1km_sf
  } else if(adm_level == "0_rm_border_2_5km"){
    roi_sf <- adm0_rm_border_2_5km_sf
  } else if(adm_level == "0_rm_border_5km"){
    roi_sf <- adm0_rm_border_5km_sf
  } else if(adm_level == "0_rm_border_10km"){
    roi_sf <- adm0_rm_border_10km_sf
  } else{
    roi_sf <- read_sf(file.path(gadm_dir, "rawdata", paste0("gadm41_MMR_",adm_level,".json")))
  }

  # Loop through product -------------------------------------------------------
  # VNP46A2 = daily
  # VNP46A3 = monthly
  # VNP46A4 = annually

  for(product in c("VNP46A4", "VNP46A3")){

    ## Make directory to export files - organized by ROI and prduct
    OUT_DIR <- file.path(ntl_bm_dir, "FinalData", "aggregated", paste0("adm", adm_level, "_", product))
    dir.create(OUT_DIR)

    # Loop through rasters -----------------------------------------------------
    r_name_vec <- file.path(ntl_bm_dir, "FinalData", paste0(product, "_rasters")) %>% list.files()

    for(r_name_i in r_name_vec){

      OUT_FILE <- file.path(OUT_DIR, r_name_i %>% str_replace_all(".tif", ".Rds"))

      ## Check if file exists
      if(!file.exists(OUT_FILE)){

        ## Load raster and create rasters for just gas flaring and non gas flaring locations
        r <- raster(file.path(ntl_bm_dir, "FinalData", paste0(product, "_rasters"), r_name_i))

        ## Extract data
        ntl_df <- exact_extract(r, roi_sf, fun = c("mean", "median", "sum"))
        roi_sf$ntl_bm_mean   <- ntl_df$mean
        roi_sf$ntl_bm_median <- ntl_df$median
        roi_sf$ntl_bm_sum    <- ntl_df$sum

        ## Prep for export
        roi_df <- roi_sf %>%
          st_drop_geometry()

        ## Add date
        if(product == "VNP46A2"){
          year <- r_name_i %>% substring(12,15) %>% as.numeric()
          day  <- r_name_i %>% substring(12,21)
          date_r <- r_name_i %>%
            str_replace_all("VNP46A2_t", "") %>%
            str_replace_all(".tif", "") %>%
            str_replace_all("_", "-") %>%
            paste0("-01") %>%
            ymd()
        }

        if(product == "VNP46A3"){
          date_r <- r_name_i %>%
            str_replace_all("VNP46A3_t", "") %>%
            str_replace_all(".tif", "") %>%
            str_replace_all("_", "-") %>%
            paste0("-01") %>%
            ymd()
        }

        if(product == "VNP46A4"){
          # Just grab year
          date_r <- r_name_i %>%
            str_replace_all("VNP46A4_t", "") %>%
            str_replace_all(".tif", "") %>%
            as.numeric()
        }

        roi_df$date <- date_r

        ## Export
        saveRDS(roi_df, OUT_FILE)
      }
    }
  }
}
