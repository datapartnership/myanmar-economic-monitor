# Append Data

roi = "adm2"
product <- "VNP46A3"

for(roi in c("adm0", "adm1", "adm2", "adm3", "admsez", "admbound2", "admbound1")){ # "tessellation",
  for(product in c("VNP46A2", "VNP46A3", "VNP46A4")){

    df <- file.path(ntl_bm_dir, "FinalData", "aggregated", paste0(roi, "_", product)) %>%
      list.files(pattern = "*.Rds",
                 full.names = T) %>%
      map_df(readRDS)

    saveRDS(df, file.path(ntl_bm_dir, "FinalData", "aggregated",
                          paste0(roi, "_", product, ".Rds")))

    write_csv(df, file.path(ntl_bm_dir, "FinalData", "aggregated",
                            paste0(roi, "_", product, ".csv")))

  }
}
