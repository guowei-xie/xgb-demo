library(tidyverse)
library(httr)
library(jsonlite)

# API 地址
url <- 'http://127.0.0.1:8000/predict/batch'

get_pred <- function (users) {
  # browser()

  # 封装 batch 请求格式
  body_json <- toJSON(list(users = users), auto_unbox = TRUE)
  
  # 发送 POST 请求
  res <- POST(
    url,
    add_headers(`Content-Type` = 'application/json'),
    body = body_json
  )
  
  # 解析返回结果
  parsed <- content(res, as = 'parsed', encoding = 'UTF-8')
  
  # 显示结果
  parsed[1]$predictions
}

# 读取原始数据
data <- read_csv('temp/2025秋12期-开始前.csv') %>% 
  group_by(user_id) %>% # 去除重复数据
  filter(n() == 1) %>% 
  ungroup() %>% 
  select(-is_renewal) %>% # 剔除y列
  filter_at(
    vars(
      city,
      city_level,
      city_score,
      house_price,
      grade,
      refresh_num,
      device,
      is_enable,
      fns_cnt
    ),
    ~ { !is.na(.x) } # 去除变量为空的行
  ) %>% 
  inner_join(
    read_csv('temp/2025秋12期-开始后.csv') %>% 
      select(user_id, is_renewal) %>% # 选择y列
      group_by(user_id) %>% # 去除重复数据
      filter(n() == 1) %>% 
      ungroup()
  ) %>% 
  rowwise() %>% 
  mutate(
    users = list( # 构建输入变量
      list(
        city = city,
        city_level = city_level,
        city_score = city_score,
        house_price = house_price,
        grade = grade,
        refresh_num = refresh_num,
        device = device,
        is_enable = is_enable,
        fns_cnt = fns_cnt
      )
    )
  ) %>% 
  ungroup() %>% 
  mutate(pred = get_pred(users)) %>% # 执行预测
  rowwise() %>% 
  mutate(
    pred_probability = pred$probability, # 获取概率
    pred_label = pred$level_tag # 获取标签
  ) %>% 
  ungroup()

# 保存结果
data %>% 
  select(-users, -pred) %>% 
  write_csv('results/test_predictions.csv')
