# Export Data

df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                        paste0("adm",2, "_", "VNP46A3", ".Rds")))

df$NAME_2 %>% unique()

write_csv()
