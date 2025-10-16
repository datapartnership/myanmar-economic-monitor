# Gas Flare Trends

# Load data --------------------------------------------------------------------
roi     <- "adm0"
product <- "VNP46A4"

ntl_df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                            paste0(roi, "_", product, ".Rds")))

# Figure -----------------------------------------------------------------------
ntl_df %>%
  ggplot(aes(x = date, y = ntl_bm_gf_mean)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_x_continuous(labels = 2012:2022,
                     breaks = 2012:2022) +
  labs(x = NULL,
       y = "Nighttime\nLights",
       title = "Nighttime Lights: Gas Flaring Locations") +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
        plot.title = element_text(face = "bold"))

ggsave(filename = file.path(fig_dir, "ntl_gas_flaring_trends.png"),
       height = 3.5, width = 7)
