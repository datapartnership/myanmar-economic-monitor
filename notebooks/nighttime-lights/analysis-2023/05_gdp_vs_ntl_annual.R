# GDP vs NTL

# Load/prep NTL data -----------------------------------------------------------
roi     <- "adm0"
product <- "VNP46A4"

ntl_df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                            paste0(roi, "_", product, ".Rds")))

ntl_df <- ntl_df %>%
  rename(year = date) %>%
  dplyr::select(year,
                ntl_bm_mean, ntl_bm_median,
                ntl_bm_gf_prop_g2,
                ntl_bm_gf_prop_g5,
                ntl_bm_gf_prop_g10,
                ntl_bm_no_gf_mean)

# Load GDP ---------------------------------------------------------------------
wdi_df <- readRDS(file.path(wdi_dir, "FinalData", "wdi.Rds"))

wdi_df <- wdi_df %>%
  rename(gdp = "NY.GDP.MKTP.CD")

# Merge ------------------------------------------------------------------------
df_wide <- ntl_df %>%
  left_join(wdi_df, by = c("year"))

df_long <- df_wide %>%
  dplyr::select(-c(country, iso2c, iso3c)) %>%
  pivot_longer(cols = -c(year))

# Correlations -----------------------------------------------------------------
df_long_gdp <- df_wide %>%
  dplyr::select(-c(country, iso2c, iso3c)) %>%
  pivot_longer(cols = -c(year, gdp))

cor_df <- df_long_gdp %>%
  filter(!is.na(gdp),
         !is.na(value)) %>%
  group_by(name) %>%
  summarise(cor = cor(gdp, value),
            cor_log = cor(log(gdp), log(value))) %>%
  arrange(-cor)

# Figure -----------------------------------------------------------------------
## Add Percent change from start
df_long <- df_long %>%
  mutate(value_log = log(value+1)) %>%

  group_by(name) %>%
  mutate(value_start = value[year == 2012],
         value_log_start = value_log[year == 2012]) %>%
  ungroup(name) %>%

  mutate(value_pc     = (value     - value_start)    /value_start*100,
         value_log_pc = (value_log - value_log_start)/value_log_start*100)

## Scatter: levels
df_wide %>%
  ggplot(aes(x = gdp, y = ntl_bm_mean)) +
  geom_smooth(method='lm', formula= y~x, se=F, color = "darkorange") +
  geom_point(size = 3) +
  stat_cor(p.accuracy = 0.001, r.accuracy = 0.01) +
  labs(x = "GDP",
       y = "Nighttime\nLights") +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5))

ggsave(filename = file.path(fig_dir, "gdp_ntl_annual_scatter.png"),
       height = 3.5, width = 5)

## Scatter: logs
df_wide %>%
  mutate(gdp = log(gdp),
         ntl_bm_mean = log(ntl_bm_mean)) %>%
  ggplot(aes(x = gdp, y = ntl_bm_mean)) +
  geom_smooth(method='lm', formula= y~x, se=F, color = "darkorange") +
  geom_point(size = 3) +
  stat_cor(p.accuracy = 0.001, r.accuracy = 0.01) +
  labs(x = "GDP, logged",
       y = "Nighttime\nLights, logged") +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5))

ggsave(filename = file.path(fig_dir, "gdp_ntl_annual_scatter_log.png"),
       height = 3.5, width = 5)


## Trends
df_long %>%
  filter(name %in% c("gdp", "ntl_bm_mean")) %>%
  mutate(name = case_when(
    name == "gdp" ~ "GDP",
    name == "ntl_bm_mean" ~ "Nighttime Lights"
  )) %>%
  ggplot(aes(x = year, y = value)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  facet_wrap(~name,
             scales = "free_y",
             ncol = 1) +
  labs(x = NULL,
       y = NULL,
       title = "Annual GDP and Nighttime Lights") +
  scale_x_continuous(labels = 2012:2022,
                     breaks = 2012:2022) +
  theme_classic2() +
  theme(strip.text = element_text(face = "bold", size = 12),
        strip.background = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(filename = file.path(fig_dir, "gdp_ntl_annual_trends.png"),
       height = 4.5, width = 6)

## Trends Log
df_long %>%
  filter(name %in% c("gdp", "ntl_bm_mean")) %>%
  mutate(name = case_when(
    name == "gdp" ~ "GDP (logged)",
    name == "ntl_bm_mean" ~ "Nighttime Lights (logged)"
  )) %>%
  ggplot(aes(x = year, y = value_log)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  facet_wrap(~name,
             scales = "free_y",
             ncol = 1) +
  labs(x = NULL,
       y = NULL,
       title = "Annual GDP and Nighttime Lights (Logged)") +
  scale_x_continuous(labels = 2012:2022,
                     breaks = 2012:2022) +
  theme_classic2() +
  theme(strip.text = element_text(face = "bold", size = 12),
        strip.background = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(filename = file.path(fig_dir, "gdp_ntl_annual_trends_log.png"),
       height = 4.5, width = 6)


## Percent Change Trends
df_long %>%
  filter(name %in% c("gdp", "ntl_bm_mean")) %>%
  mutate(name = case_when(
    name == "gdp" ~ "GDP",
    name == "ntl_bm_mean" ~ "Nighttime Lights"
  )) %>%
  ggplot(aes(x = year, y = value_pc, color = name)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_x_continuous(labels = 2012:2022,
                     breaks = 2012:2022) +
  labs(x = NULL,
       y = "Percent\nChange",
       color = NULL,
       title = "GDP & Nighttime Lights: Percent Change Since 2012") +
  scale_color_manual(values = c("gray20", "darkorange")) +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5),
        plot.title = element_text(face = "bold"))

ggsave(filename = file.path(fig_dir, "gdp_ntl_annual_trends_pc.png"),
       height = 3.5, width = 7)

## Percent Change Trends: Log
df_long_fig <- df_long %>%
  filter(name %in% c("gdp", "ntl_bm_mean")) %>%
  mutate(name_clean = case_when(
    name == "gdp" ~ "GDP\n(logged)",
    name == "ntl_bm_mean" ~ "Nighttime\nLights\n(logged)"
  ))

coeff <- 70

ggplot() +
  geom_line(data = df_long_fig %>% filter(name == "gdp"),
            aes(x = year, y = value_log_pc, color = name_clean),
            size = 1) +
  geom_line(data = df_long_fig %>% filter(name == "ntl_bm_mean"),
             aes(x = year, y = (value_log_pc/coeff), color = name_clean),
             size = 1) +

  geom_point(data = df_long_fig %>% filter(name == "gdp"),
            aes(x = year, y = value_log_pc, color = name_clean),
            size = 2) +
  geom_point(data = df_long_fig %>% filter(name == "ntl_bm_mean"),
            aes(x = year, y = (value_log_pc/coeff), color = name_clean),
            size = 2) +

  scale_y_continuous(

    # Features of the first axis
    name = "Percent\nChange",

    # Add a second axis and specify its features
    sec.axis = sec_axis(~.*coeff, name="Percent\nChange")
  ) +

  scale_x_continuous(labels = 2012:2022,
                     breaks = 2012:2022) +
  labs(x = NULL,
       y = "Percent\nChange",
       color = NULL,
       title = "GDP & Nighttime Lights (Logged): Percent Change Since 2012") +
  scale_color_manual(values = c("gray20", "darkorange")) +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0,
                                    vjust = 0.5,
                                    face = "bold"),
        axis.title.y.right = element_text(angle = 0,
                                          vjust = 0.5,
                                          face = "bold",
                                          color = "darkorange"),
        plot.title = element_text(face = "bold"))

ggsave(filename = file.path(fig_dir, "gdp_ntl_annual_trends_pc_log.png"),
       height = 3.5, width = 7.5)



# Regression -------------------------------------------------------------------
lm1 <- lm(gdp ~ ntl_bm_mean, data = df_wide %>%
            mutate(gdp = gdp / 1000000000))
lm2 <- lm(log(gdp) ~ log(ntl_bm_mean), data = df_wide)

stargazer(lm1,
          lm2,
          dep.var.labels = c("GDP (Billions)", "log(GDP)"),
          covariate.labels = c("NTL", "log(NTL)"),
          omit.stat=c("LL","ser","f"),
          type = "html",
          out = file.path(fig_dir, "reg_gdp_ntl.html"))
