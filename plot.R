library(tidyverse)
library(patchwork)

set.seed(1234)

source('../src/tools.R')
tools$setup()

data <- as_tibble(read.csv('results/test_predictions.csv'))

data %>% 
  mutate(l1_term_name = tools$term_name_to_factor(l1_term_name)) %>% 
  filter(between(l1_term_name, '2025秋01期', '2025秋11期')) %>% 
  select(
    l1_term_name,
    city,
    city_level,
    city_score,
    house_price,
    grade,
    refresh_num,
    device,
    is_enable,
    fns_cnt,
    is_renewal
  ) %>% 
  mutate(house_price = round(house_price / 10000, 2)) %>% 
  mutate_at(
    vars(city_score, house_price),
    ~ {
      cut(.x, unique(quantile(.x, seq(0, 1, 0.1), na.rm = T)))
    }
  ) %>% 
  mutate_at(vars(-is_renewal), as_factor) %>% 
  pivot_longer(city:fns_cnt) %>% 
  filter(name != 'city') %>% 
  {
    # browser()
    
    data <- tibble(.) %>% 
      group_by(l1_term_name, name, value) %>% 
      summarise(
        is_renewal = mean(is_renewal), 
        n = n()
      )
    
    plot_data <- function (name, level = NULL) {
      # browser()
      
      data <- data %>% 
        filter(name == {{ name }})
      
      if (is.null(level)) {
        level <- sort(unique(data$value))
      }
      
      data %>% 
        ggplot(aes(factor(value, levels = {{ level }}, ordered = T), is_renewal)) +
        geom_jitter(
          aes(size = n), 
          height = 0, width = 0.3, show.legend = F,
          shape = 21, col = 'white', fill = 'lightblue'
        ) +
        stat_summary(
          fun = 'median',
          col = 'red',
          fun.max = \(x) quantile(x, 0.75),
          fun.min = \(x) quantile(x, 0.25)
        ) +
        ylim(0, 0.2) +
        labs(
          x = NULL,
          y = '各学期续报率',
          title = name
        )
    }
    
    (
      (
        plot_data('fns_cnt', c(0, 1, 2, 3, 4, 5, NA))
      ) + (
        plot_data('is_enable', c(0, 1, NA))
      )
    ) / (
      (
        plot_data('city_level', rev(c('一线城市', '新一线城市', '二线城市', '三线城市', '四线城市', '五线城市', '')))
        ) + (
          plot_data('city_score')
          )
    ) / (
      (
        plot_data('house_price')
      ) + (
        plot_data('device')
      )
    ) / (
      (
        plot_data('grade')
      ) + (
        plot_data('refresh_num', 0:100)
      )
    )
  } +
  tools$ggcopy(width = 1.2, height = 1)
  
