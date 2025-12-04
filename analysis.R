library(tidyverse)
library(patchwork)

source('../src/tools.R')
tools$setup()

data <- read.csv('results/test_predictions.csv')

col <- 'fns_cnt'

data %>% 
  as_tibble() %>% 
  filter(city_level %in% c('新一线城市', '一线城市')) %>%
  filter(city == '上海市') %>%
  group_by(l1_term_name, !!sym(col)) %>% 
  summarise(
    n = n(),
    is_renewal = mean(is_renewal),
    pred_probability = mean(pred_probability)
  ) %>% 
  mutate(error = pred_probability - is_renewal) %>% 
  group_by(l1_term_name, !!sym(col)) %>% 
  mutate(
    sum_n = sum(n),
    p = n / sum(n)
    ) %>% 
  # filter(p >= 0.05) %>%
  {
    (
      tibble(.) %>% 
        ggplot() +
        geom_line(aes(l1_term_name, error, group = 1), linewidth = 0.6, col = 'gray50') +
        geom_point(aes(l1_term_name, error), size = 1.5, col = 'black') +
        geom_hline(aes(yintercept = 0), linetype = 'dashed') +
        geom_vline(aes(xintercept = c('2025秋13期')), linetype = 'dashed') +
        # geom_vline(aes(xintercept = c('2025秋11期')), linetype = 'dashed') +
        # geom_vline(aes(xintercept = c('2025秋12期')), linetype = 'dashed') +
        facet_grid(~ .data[[col]]) +
        ylim(-0.1, 0.1) +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 90)) +
        labs(x = NULL)
    ) / (
      tibble(.) %>% 
        ggplot() +
        geom_col(aes(l1_term_name, n), col = 'black', alpha = 0.5) +
        geom_vline(aes(xintercept = c('2025秋13期')), linetype = 'dashed') +
        # geom_vline(aes(xintercept = c('2025秋11期')), linetype = 'dashed') +
        # geom_vline(aes(xintercept = c('2025秋12期')), linetype = 'dashed') +
        facet_grid(~ .data[[col]]) +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 90)) +
        labs(x = NULL)
    )
  } +
  tools$ggcopy()
