# Append Data

if(F){
  roi = 0
  product <- "VNP46A4"

  for(roi in 0:3){

    df <- readRDS(file.path(ntl_bm_dir, "FinalData", "aggregated",
                            paste0("adm",roi, "_", product, ".Rds")))

    if(roi %in% 0){
      df$NAME_VAR <- df$COUNTRY
      height <- 3
      width <- 5
    }
    if(roi %in% 1){
      df$NAME_VAR <- df$NAME_1
      height <- 6
      width <- 8
    }
    if(roi %in% 2){
      df$NAME_VAR <- df$NAME_2
      height <- 8
      width <- 12
    }
    if(roi %in% 3){
      df$NAME_VAR <- df$NAME_3
      height <- 20
      width <- 25
    }

    df %>%
      group_by(date, NAME_VAR) %>%
      dplyr::summarise(ntl_bm_mean = mean(ntl_bm_mean)) %>%
      ggplot(aes(x = date, y = ntl_bm_mean)) +
      geom_col() +
      theme_classic2() +
      theme(strip.background = element_blank(),
            strip.text = element_text(face = "bold"),
            axis.text.x = element_text(size = 7)) +
      facet_wrap(~NAME_VAR, scales = "free_y") +
      labs(x = NULL, y = "NTL") +
      scale_x_continuous(labels = seq(2012, 2022, 3),
                         breaks = seq(2012, 2022, 3))

    ggsave(file.path(fig_dir, paste0("ntl_trends_adm",roi,".png")),
           height = height, width = width)

  }
}
