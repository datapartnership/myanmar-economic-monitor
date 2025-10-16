# Download WDI

wdi_df <- WDI(country="MM", indicator="NY.GDP.MKTP.CD", start=2012, end=2022)

saveRDS(wdi_df, file.path(wdi_dir, "FinalData", "wdi.Rds"))
