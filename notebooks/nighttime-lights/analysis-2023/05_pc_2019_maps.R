# Comparison to 2019 Maps

roi <- 1

for(roi in 1:3){

  # Load data --------------------------------------------------------------------
  roi_sf <- read_sf(file.path(gadm_dir, "rawdata", paste0("gadm41_MMR_",roi,".json")))

  ntl_df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                              paste0("adm",roi, "_", "VNP46A4", ".Rds")))

  if(roi %in% 0) roi_sf$NAME_VAR <- roi_sf$GID_0
  if(roi %in% 1) roi_sf$NAME_VAR <- roi_sf$GID_1
  if(roi %in% 2) roi_sf$NAME_VAR <- roi_sf$GID_2
  if(roi %in% 3) roi_sf$NAME_VAR <- roi_sf$GID_3

  if(roi %in% 0) ntl_df$NAME_VAR <- ntl_df$GID_0
  if(roi %in% 1) ntl_df$NAME_VAR <- ntl_df$GID_1
  if(roi %in% 2) ntl_df$NAME_VAR <- ntl_df$GID_2
  if(roi %in% 3) ntl_df$NAME_VAR <- ntl_df$GID_3

  ntl_wide_df <- ntl_df %>%
    mutate(date = paste0("yr_", date)) %>%
    pivot_wider(id_cols = NAME_VAR,
                names_from = date,
                values_from = ntl_bm_mean) %>%
    mutate(yr_20_b19 = yr_2020 - yr_2019,
           yr_21_b19 = yr_2021 - yr_2019,
           yr_22_b19 = yr_2022 - yr_2019,

           yr_20_b19_pc = yr_20_b19/yr_2019*100,
           yr_21_b19_pc = yr_21_b19/yr_2019*100,
           yr_22_b19_pc = yr_22_b19/yr_2019*100)

  roi_sf <- roi_sf %>%
    left_join(ntl_wide_df, by = "NAME_VAR")

  # Map --------------------------------------------------------------------------
  # roi_sf %>%
  #   mutate(yr_2019_ln = log(yr_2019+1)) %>%
  #   ggplot() +
  #   geom_sf(aes(fill = yr_2019_ln),
  #           color = NA) +
  #   scale_fill_viridis_c() +
  #   theme_void() +
  #   labs(fill = "NTL (Logged)")

  # roi_sf %>%
  #   ggplot() +
  #   geom_sf(aes(fill = yr_20_b19_pc),
  #           color = NA) +
  #   scale_fill_viridis_c() +
  #   theme_void()

  roi_stacked_sf <- roi_sf %>%
    dplyr::select(NAME_VAR, yr_20_b19_pc, yr_21_b19_pc, yr_22_b19_pc, geometry) %>%
    pivot_longer(cols = -c(NAME_VAR, geometry)) %>%
    dplyr::mutate(name = case_when(
      name == "yr_20_b19_pc" ~ "2020",
      name == "yr_21_b19_pc" ~ "2021",
      name == "yr_22_b19_pc" ~ "2022",
    ))

  roi_stacked_sf$value[is.na(roi_stacked_sf$value)] <- 0

  roi_stacked_sf$value[roi_stacked_sf$value > 100]  <- 100
  roi_stacked_sf$value[roi_stacked_sf$value < -100] <- -100

  roi_stacked_sf %>%
    ggplot() +
    geom_sf(aes(fill = value),
            color = "black") +
    #scale_fill_viridis_c() +
    theme_void() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5)) +
    facet_wrap(~name) +
    labs(title = paste0("Percent Change in Nighttime Lights from 2019: ADM ", roi, " level"),
         fill = "% Change") +
    scale_fill_gradient2(low = "red",
                         mid = "white",
                         high = "forestgreen",
                         midpoint = 0,
                         limits = c(-100, 100),
                         breaks = c(-100, -50, 0, 50, 100),
                         labels = c("< -100", "-50", "0", "50", "> 100"))

  ggsave(file.path(fig_dir, paste0("pc_2019_maps",roi,".png")),
         height = 5, width = 8)

}
