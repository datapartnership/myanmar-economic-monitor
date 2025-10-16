# RWI and Nighttime Lights Analysis

library(gtools)

# Load data --------------------------------------------------------------------
rwi_df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                            paste0("admrwi", "_", "VNP46A4", ".Rds")))

rwi_df <- rwi_df %>%
  mutate(ntl_bm_sum_ln = log(ntl_bm_sum + 1))

rwi_df$rwi_q4 <- rwi_df$rwi %>%
  quantcut(q = 4, na.rm = TRUE) %>%
  as.numeric() %>%
  factor()

rwi_df$rwi_q5 <- rwi_df$rwi %>%
  quantcut(q = 5, na.rm = TRUE) %>%
  as.numeric() %>%
  factor()

# Correlation ------------------------------------------------------------------
rwi_df %>%
  group_by(date) %>%
  dplyr::summarise(cor = cor(ntl_bm_sum_ln, rwi))

rwi_df %>%
  dplyr::filter(date == 2022) %>%
  ggplot(aes(x = ntl_bm_sum_ln,
             y = rwi)) +
  geom_point() +
  #geom_smooth(method='lm', formula= y~x, se = F, color = "darkorange") +
  labs(x = "Nighttime lights, logged",
       y = "Relative Wealth Index",
       title = "Association between nighttime lights and relative wealth index") +
  theme_classic2()

ggsave(filename = file.path(fig_dir, "rwi_ntl_cor.png"),
       height = 6, width = 10)

# Trends -----------------------------------------------------------------------
rwi_df %>%
  group_by(date, rwi_q4) %>%
  dplyr::summarise(ntl_bm_sum = sum(ntl_bm_sum)) %>%
  ungroup() %>%
  ggplot(aes(x = date,
             y = ntl_bm_sum)) +
  geom_col() +
  facet_wrap(~rwi_q4,
             scales = "free_y") +
  labs(x = NULL,
       y = "Nighttime lights radiance",
       title = "Nighttime lights by RWI quantile") +
  theme_classic2() +
  theme(strip.background = element_blank())

ggsave(filename = file.path(fig_dir, "rwi_q4_ntl_trends.png"),
       height = 6, width = 10)
