# NTL Maps

# Make maps of nighttime lights for each year

# Load data --------------------------------------------------------------------
adm_sf <- read_sf(file.path(gadm_dir, "rawdata", paste0("gadm41_MMR_",0,".json")))

# Annual map -------------------------------------------------------------------
for(year_i in 2012:2022){
  print(year_i)

  r <- raster(file.path(ntl_bm_dir, "FinalData", "VNP46A4_rasters",
                        paste0("bm_VNP46A4_",year_i,".tif")))

  #### Prep data
  r <- r %>% crop(adm_sf) %>% mask(adm_sf)

  r_df <- rasterToPoints(r, spatial = TRUE) %>% as.data.frame()
  names(r_df) <- c("value", "x", "y")


  ## Remove very low values of NTL; can be considered noise
  r_df$value[r_df$value <= 1] <- 0

  ## Distribution is skewed, so log
  r_df$value_adj <- log(r_df$value+1)

  ##### Map
  p <- ggplot() +
    geom_raster(data = r_df,
                aes(x = x, y = y,
                    fill = value_adj)) +
    scale_fill_gradient2(low = "black",
                         mid = "yellow",
                         high = "red",
                         midpoint = 4) +
    labs(title = paste0("Nighttime Lights: ", year_i)) +
    coord_quickmap() +
    theme_void() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          legend.position = "none")

  ggsave(p, filename = file.path(fig_dir, paste0("ntl_bm_",year_i,".png")),
         height = 4, width = 2)
}
