# SEZ Trends

# Load and prep data -----------------------------------------------------------
roi     <- "admsez"
product <- "VNP46A4"

ntl_df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                            paste0(roi, "_", product, ".Rds")))

## Sez
sez_sf <- read_sf(file.path(data_dir, "SEZ", "RawData",
                            "industrial__special_economic_zones_sept2019.shp"))

sez_df <- st_coordinates(sez_sf) %>%
  as.data.frame() %>%
  dplyr::rename(lon = X,
                lat = Y)
sez_df$Name <- sez_sf$Name

write_csv(sez_df, file.path(data_dir, "SEZ", "FinalData", "sez_locations.csv"))

ntl_df <- ntl_df %>%
  left_join(sez_df, by = "Name")

## Myn Map
adm_sf <- read_sf(file.path(gadm_dir, "rawdata", paste0("gadm41_MMR_",0,".json")))

# Shorten Names ----------------------------------------------------------------
ntl_df <- ntl_df %>%
  mutate(Name = case_when(
    Name == "Bago Industrial Complex (Local Industrial Zone)" ~ "Bago Industrial Complex\n(Local Industrial Zone)",
    Name == "Bago Industrial Complex (Foreign Industrial Zone)" ~ "Bago Industrial Complex\n(Foreign Industrial Zone)",
    Name == "Det Khi Na Thi Ri SME Industrial Zone (Nay Pyi Taw)" ~ "Det Khi Na Thi Ri SME\nIndustrial Zone (Nay Pyi Taw)",
    Name == "Korea-Myanmar Economic Cooperation Industrial Complex" ~ "Korea-Myanmar Economic\nCooperation Industrial Complex",
    Name == "Kyauk Tan Industrial Zone - Mawlamyine" ~ "Kyauk Tan Industrial\nZone - Mawlamyine",
    TRUE ~ Name
  ))

# Trends -----------------------------------------------------------------------
ntl_df %>%
  ggplot(aes(x = date, y = ntl_bm_mean)) +
  geom_col() +
  scale_x_continuous(labels = seq(2012, 2022, 4),
                     breaks = seq(2012, 2022, 4)) +
  labs(x = NULL,
       y = NULL,
       title = "Nighttime Lights: Special Economic Zones") +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
        plot.title = element_text(face = "bold",size = 16),
        strip.text = element_text(face = "bold", size = 10),
        strip.background = element_blank()) +
  facet_wrap(~Name,
             scales = "free_y",
             ncol = 5)

ggsave(filename = file.path(fig_dir, "sez_ntl_trends.png"),
       height = 20, width = 15)

# Trends - Select Zones --------------------------------------------------------
ntl_df %>%
  dplyr::filter(Name %in% c("Hlaingtharya Industrial Zone (1)",
                            "Shwepyithar Industrial Zone (3)",
                            "Mingaladon Industrial Zone",
                            "Thilawa Industrial Zone")) %>%
  ggplot(aes(x = date, y = ntl_bm_mean)) +
  geom_col() +
  scale_x_continuous(labels = seq(2012, 2022, 4),
                     breaks = seq(2012, 2022, 4)) +
  labs(x = NULL,
       y = NULL,
       title = "Nighttime Lights: Special Economic Zones") +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
        plot.title = element_text(face = "bold",size = 16),
        strip.text = element_text(face = "bold", size = 10),
        strip.background = element_blank()) +
  facet_wrap(~Name,
             scales = "free_y",
             ncol = 2)

# ntl_df %>%
#   dplyr::filter(Name %in% c("Hlaingtharya Industrial Zone (1)",
#                             "Shwepyithar Industrial Zone (3)",
#                             "Mingaladon Industrial Zone",
#                             "Thilawa Industrial Zone")) %>%
#   dplyr::select(Name, date, ntl_bm_mean) %>%
#   write_csv("~/Desktop/sez_ntl_trends_select_zones_rawdata.csv")

ggsave(filename = file.path(fig_dir, "sez_ntl_trends_select_zones.png"),
       height = 4, width = 6)

# SEZ Aggregate Trends ---------------------------------------------------------
ntl_df %>%
  group_by(date) %>%
  dplyr::summarise(ntl_bm_mean = mean(ntl_bm_mean)) %>%
  ungroup() %>%

  ggplot(aes(x = date, y = ntl_bm_mean)) +
  geom_col() +
  scale_x_continuous(labels = seq(2012, 2022, 4),
                     breaks = seq(2012, 2022, 4)) +
  labs(x = NULL,
       y = NULL,
       title = "Nighttime Lights: Special Economic Zones (Average)") +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
        plot.title = element_text(face = "bold",size = 16),
        strip.text = element_text(face = "bold", size = 10),
        strip.background = element_blank())

ggsave(filename = file.path(fig_dir, "sez_ntl_trends_avg.png"),
       height = 4, width = 6)

# Trends -----------------------------------------------------------------------
ntl_sum_df <- ntl_df %>%
  group_by(Name, lon, lat) %>%
  summarise(ntl_bm_mean_2012 = ntl_bm_mean[date %in% 2012],
            ntl_bm_mean_2022 = ntl_bm_mean[date %in% 2022]) %>%
  ungroup() %>%
  mutate(ntl_bm_mean_22_12_chng = ntl_bm_mean_2022 - ntl_bm_mean_2012,
         ntl_bm_mean_22_12_pc = ntl_bm_mean_22_12_chng / ntl_bm_mean_2012) %>%

  mutate(ntl_bm_mean_22_12_chng_cat = case_when(
    (ntl_bm_mean_22_12_chng >= -120) & (ntl_bm_mean_22_12_chng <= 0) ~ "< 0",
    (ntl_bm_mean_22_12_chng > 0) & (ntl_bm_mean_22_12_chng <= 50) ~ "0 - 50",
    (ntl_bm_mean_22_12_chng > 50) & (ntl_bm_mean_22_12_chng <= 100) ~ "50 - 100",
    (ntl_bm_mean_22_12_chng > 100) & (ntl_bm_mean_22_12_chng <= 150) ~ "100 - 150",
    (ntl_bm_mean_22_12_chng > 150) ~ "> 150"
  ) %>%

    factor(levels = c("< 0",
                      "0 - 50",
                      "50 - 100",
                      "100 - 150",
                      "> 150"))
  )


ggplot() +
  geom_sf(data = adm_sf,
          fill = "gray20") +
  geom_point(data = ntl_sum_df,
             aes(x = lon, y = lat, fill = ntl_bm_mean_22_12_chng),
             pch = 21,
             size = 2) +
  scale_fill_distiller(palette = "Spectral") +
  labs(fill = "Change\nin NTL",
       title = "Change in Nighttime Lights in Special Economic Zones\nfrom 2012 to 2022") +
  theme_void() +
  theme(strip.text = element_text(face = "bold"),
        plot.title = element_text(face = "bold", hjust = 0.5)) +
  facet_wrap(~ntl_bm_mean_22_12_chng_cat)

ggsave(filename = file.path(fig_dir, "sez_ntl_map.png"),
       height = 6, width = 6)
