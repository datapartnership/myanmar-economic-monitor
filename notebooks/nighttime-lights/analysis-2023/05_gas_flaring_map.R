# Gas Flaring Map

# Load data --------------------------------------------------------------------
adm_sf <- read_sf(file.path(gadm_dir, "rawdata", paste0("gadm41_MMR_",0,".json")))

r <- raster(file.path(ntl_bm_dir, "FinalData", "VNP46A4_rasters",
                      paste0("bm_VNP46A4_",2022,".tif")))

gf_df <- readRDS(file.path(gas_flare_dir, "FinalData", "gas_flare_locations.Rds"))

# Prep data --------------------------------------------------------------------
#### NTL Raster
r <- r %>% crop(adm_sf) %>% mask(adm_sf)

r_df <- rasterToPoints(r, spatial = TRUE) %>% as.data.frame()
names(r_df) <- c("value", "x", "y")


## Remove very low values of NTL; can be considered noise
r_df$value[r_df$value <= 1] <- 0

## Distribution is skewed, so log
r_df$value_adj <- log(r_df$value+1)

#### Gas Flaring
gf_sp <- gf_df
coordinates(gf_sp) <- ~longitude + latitude
gf_buff <- gBuffer(gf_sp, width = 30/111.12, byid = T)

gf_buff <- gf_buff[as.vector(gIntersects(gf_buff, as(adm_sf, "Spatial"), byid = T)),]

# Map --------------------------------------------------------------------------

gf_tidy_buff <- tidy(gf_buff)
gf_tidy_buff$gf_loc <- "Gas Flaring Location"

p <- ggplot() +
  geom_raster(data = r_df,
              aes(x = x, y = y,
                  fill = value_adj)) +
  geom_polygon(data = gf_tidy_buff,
               aes(x = long, y = lat, group = group, color = gf_loc),
               fill = NA) +
  scale_fill_gradient2(low = "black",
                       mid = "yellow",
                       high = "red",
                       midpoint = 4) +
  labs(title = paste0("Nighttime Lights: 2022"),
       color = NULL,
       fill = "Nighttime\nLights") +
  coord_quickmap() +
  theme_void() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave(p, filename = file.path(fig_dir, paste0("ntl_bm_gf_",2022,".png")),
       height = 6, width = 5)
