# Download Black Marble Data

# Bearer Key -------------------------------------------------------------------
#### Bearer token from NASA
# To get token, see: https://github.com/ramarty/download_blackmarble
bearer <- read_csv("~/Desktop/bearer_bm.csv") %>% pull(token)

# Define Region of Interest ----------------------------------------------------
roi_sf <- read_sf(file.path(gadm_dir, "rawdata", paste0("gadm41_MMR_",0,".json")))

# Download ---------------------------------------------------------------------
bm_raster(roi_sf = roi_sf,
          product_id = "VNP46A4",
          date = 2012:2023,
          bearer = bearer,
          output_location_type = "file",
          file_dir = file.path(ntl_bm_dir, "FinalData", "VNP46A4_rasters"))

bm_raster(roi_sf = roi_sf,
          product_id = "VNP46A3",
          date = seq.Date(from = ymd("2012-01-01"), to = Sys.Date(), by = "month") %>% rev(),
          bearer = bearer,
          output_location_type = "file",
          file_dir = file.path(ntl_bm_dir, "FinalData", "VNP46A3_rasters"))

bm_raster(roi_sf = roi_sf,
          product_id = "VNP46A2",
          date = seq.Date(from = ymd("2022-01-01"), to = Sys.Date(), by = "day") %>% rev(),
          bearer = bearer,
          output_location_type = "file",
          file_dir = file.path(ntl_bm_dir, "FinalData", "VNP46A2_rasters"))
