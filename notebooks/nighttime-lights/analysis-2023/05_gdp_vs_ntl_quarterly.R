# GDP vs NTL

# Load/prep NTL data -----------------------------------------------------------
roi     <- "adm0"
product <- "VNP46A3"

ntl_df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                        paste0(roi, "_", product, ".Rds")))

ntl_df <- ntl_df %>%
  mutate(date = floor_date(date, unit = "quarter"),
         month = date %>% month,
         year = date %>% year) %>%
  mutate(quarter = case_when(
    month == 1 ~ "q1",
    month == 4 ~ "q2",
    month == 7 ~ "q3",
    month == 10 ~ "q4"
  )) %>%
  group_by(year, quarter, date) %>%
  summarise(ntl_bm_mean = mean(ntl_bm_mean))

# Load GDP ---------------------------------------------------------------------
gdp_df <- read_xlsx(file.path(gdp_dir, "RawData", "2015based_RealGDP.xlsx"), 2)

gdp_df <- gdp_df %>%
  mutate(quarter = quarter %>% tolower())

# Merge ------------------------------------------------------------------------
df_wide <- ntl_df %>%
  left_join(gdp_df, by = c("year", "quarter"))

df_long <- df_wide %>%
  pivot_longer(cols = -c(year, quarter, date))

# Figure -----------------------------------------------------------------------
## Add Percent change from start
df_long <- df_long %>%
  mutate(value_log = log(value+1)) %>%

  group_by(name) %>%
  mutate(value_start = value[(year == 2012) & (quarter == "q1")],
         value_log_start = value_log[(year == 2012) & (quarter == "q1")]) %>%
  ungroup(name) %>%

  mutate(value_pc = (value - value_start)/value_start*100,
         value_log_pc = (value_log - value_log_start)/value_log_start*100)

## Scatter
df_wide %>%
  ggplot(aes(x = gdp, y = ntl_bm_mean)) +
  geom_smooth(method='lm', formula= y~x, se=F, color = "darkorange") +
  geom_point(size = 3) +
  stat_cor(p.accuracy = 0.001, r.accuracy = 0.01) +
  labs(x = "GDP",
       y = "Nighttime\nLights") +
  theme_classic2() +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5))

ggsave(filename = file.path(fig_dir, "gdp_ntl_quarterly_scatter.png"),
       height = 3.5, width = 5)

## Scatter: Log
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

ggsave(filename = file.path(fig_dir, "gdp_ntl_quarterly_scatter_log.png"),
       height = 3.5, width = 5)

## Trends
df_long %>%
  filter(year >= 2012,
         year <= 2022) %>%
  mutate(name = case_when(
    name == "gdp" ~ "GDP",
    name == "ntl_bm_mean" ~ "Nighttime Lights"
  )) %>%
  ggplot(aes(x = date, y = value)) +
  geom_vline(xintercept = ymd("2016-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2017-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2018-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2019-01-01"), alpha = 0.2) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  facet_wrap(~name, scales = "free_y", ncol = 1) +
  labs(x = NULL,
       y = NULL,
       title = "Quarterly Nighttime Lights") +
  theme_classic2() +
  theme(strip.text = element_text(face = "bold", size = 12),
        strip.background = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(filename = file.path(fig_dir, "gdp_ntl_quarterly_trends.png"),
       height = 4.5, width = 6)

## Trends - Log
df_long %>%
  filter(year >= 2012,
         year <= 2022) %>%
  mutate(name = case_when(
    name == "gdp" ~ "GDP (logged)",
    name == "ntl_bm_mean" ~ "Nighttime Lights (logged)"
  )) %>%
  ggplot(aes(x = date, y = value_log)) +
  geom_vline(xintercept = ymd("2016-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2017-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2018-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2019-01-01"), alpha = 0.2) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  facet_wrap(~name, scales = "free_y", ncol = 1) +
  labs(x = NULL,
       y = NULL,
       title = "Quarterly GDP and Nighttime Lights (Logged)") +
  theme_classic2() +
  theme(strip.text = element_text(face = "bold", size = 12),
        strip.background = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(filename = file.path(fig_dir, "gdp_ntl_quarterly_trends_log.png"),
       height = 4.5, width = 6)

## Trends PC
df_long %>%
  filter(year >= 2012,
         year <= 2022) %>%
  mutate(name = case_when(
    name == "gdp" ~ "GDP",
    name == "ntl_bm_mean" ~ "Nighttime Lights"
  )) %>%
  ggplot(aes(x = date, y = value_pc, color = name)) +
  geom_vline(xintercept = ymd("2016-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2017-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2018-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2019-01-01"), alpha = 0.2) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(x = NULL,
       y = NULL,
       color = NULL,
       title = "Quarterly GDP & Nighttime Lights: Percent Change Since Q1 2015") +
  theme_classic2() +
  theme(plot.title = element_text(face = "bold")) +
  scale_color_manual(values = c("gray20", "darkorange"))

ggsave(filename = file.path(fig_dir, "gdp_ntl_quarterly_trends_pc.png"),
       height = 3.5, width = 7)


## Trends PC - log
df_long %>%
  filter(year >= 2012,
         year <= 2022) %>%
  mutate(name = case_when(
    name == "gdp" ~ "GDP (logged)",
    name == "ntl_bm_mean" ~ "Nighttime Lights (logged)"
  )) %>%
  ggplot(aes(x = date, y = value_log_pc, color = name)) +
  geom_vline(xintercept = ymd("2016-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2017-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2018-01-01"), alpha = 0.2) +
  geom_vline(xintercept = ymd("2019-01-01"), alpha = 0.2) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(x = NULL,
       y = NULL,
       color = NULL,
       title = "Quarterly GDP & Nighttime Lights (Logged): Percent Change Since Q1 2015") +
  theme_classic2() +
  theme(plot.title = element_text(face = "bold")) +
  scale_color_manual(values = c("gray20", "darkorange"))

ggsave(filename = file.path(fig_dir, "gdp_ntl_quarterly_trends_pc_log.png"),
       height = 3.5, width = 7)

# Regression -------------------------------------------------------------------
df_wide <- df_wide %>%
  mutate(gdp_div = gdp / 1000000,
         yr2021_onwards = as.numeric(year >= 2021))

lm1 <- lm(gdp ~ ntl_bm_mean,           data = df_wide %>% filter(year %in% 2012:2020))
lm2 <- lm(log(gdp) ~ log(ntl_bm_mean), data = df_wide %>% filter(year %in% 2012:2020))
lm3 <- lm(gdp ~ ntl_bm_mean + yr2021_onwards,           data = df_wide %>% filter(year %in% 2012:2022))
lm4 <- lm(log(gdp) ~ log(ntl_bm_mean) + yr2021_onwards, data = df_wide %>% filter(year %in% 2012:2022))

stargazer(lm1,
          lm2,
          lm3,
          lm4,
          dep.var.labels = c("GDP (Millions)", "log(GDP)","GDP (Millions)", "log(GDP)"),
          covariate.labels = c("NTL", "log(NTL)", "2020 Onwards"),
          omit.stat=c("LL","ser","f"),
          add.lines=list(c("Start Year", "2012", "2012", "2012", "2012"),
                         c("End Year", "2020", "2020", "2022", "2022")),
          type = "html",
          out = file.path(fig_dir, "reg_gdp_ntl_quarterly.html"))
