# Trends

height <- 6
width  <- 8

# Load data --------------------------------------------------------------------
df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                        paste0("adm",1, "_", "VNP46A3", ".Rds")))

# Clean data -------------------------------------------------------------------
zscore <- function(x){
  (x - mean(x))/sd(x)
}

df <- df %>%
  filter(date >= ymd("2020-01-01")) %>%
  dplyr::group_by(NAME_1) %>%
  dplyr::mutate(ntl_bm_mean_z = zscore(ntl_bm_mean)) %>%
  dplyr::ungroup()

# Figures ----------------------------------------------------------------------
df %>%
  ggplot() +
  geom_vline(xintercept = ymd("2020-03-01"),
             color = "red",
             linetype="dashed") +
  geom_vline(xintercept = ymd("2021-02-01"),
             color = "red",
             linetype="dashed") +
  geom_col(aes(x = date, y = ntl_bm_mean),
           fill = "dodgerblue4") +
  facet_wrap(~NAME_1,
             scales = "free") +
  labs(x = NULL,
       y = "NTL",
       title = "Monthly Nighttime Lights",
       caption = "First red line indicates March, 2020 (COVID). Second red line indicates February, 2021 (Coup)") +
  theme_classic2() +
  theme(strip.background = element_blank(),
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(size = 7),
        axis.title.y = element_text(angle = 0, vjust = 0.5))

ggsave(file.path(fig_dir, paste0("ntl_trends_adm1_monthly.png")),
       height = height, width = width)

df %>%
  ggplot() +
  geom_vline(xintercept = ymd("2020-03-01"),
             color = "red",
             linetype="dashed") +
  geom_vline(xintercept = ymd("2021-02-01"),
             color = "red",
             linetype="dashed") +
  geom_col(aes(x = date, y = ntl_bm_mean_z),
           fill = "dodgerblue4") +
  facet_wrap(~NAME_1) +
  labs(x = NULL,
       y = "Z-Score",
       title = "Monthly Nighttime Lights",
       caption = "First red line indicates March, 2020 (COVID). Second red line indicates February, 2021 (Coup)") +
  theme_classic2() +
  theme(strip.background = element_blank(),
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(size = 7),
        axis.title.y = element_text(angle = 0, vjust = 0.5))

ggsave(file.path(fig_dir, paste0("ntl_trends_adm1_monthly_zscore.png")),
       height = height, width = width)
