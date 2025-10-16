# Clean Gas Flaring Data

# Map
# https://www.worldbank.org/en/programs/gasflaringreduction/global-flaring-data

# Load data --------------------------------------------------------------------
gf_df <- read_xlsx(file.path(gas_flare_dir, "RawData",
                             "GGFR-Flaring-Dashboard-Data-March292023.xlsx"))

# Cleanup ----------------------------------------------------------------------
gf_df <- gf_df %>%
  filter(COUNTRY %in% "Myanmar") %>%
  distinct(Latitude, Longitude) %>%
  clean_names() %>%
  mutate(uid = 1:n())

# Export -----------------------------------------------------------------------
write_csv(gf_df, file.path(gas_flare_dir, "gas_flare_locations.csv"))
saveRDS(gf_df,   file.path(gas_flare_dir, "gas_flare_locations.Rds"))
