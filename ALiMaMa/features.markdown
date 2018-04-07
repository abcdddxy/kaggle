# 原始特征
item_price_level, item_sales_level, item_collected_level,
    item_pv_level, user_gender_id, user_age_level, user_occupation_id,
    user_star_level, context_page_id, shop_review_num_level,
    shop_review_positive_rate, shop_star_level, shop_score_service,
    shop_score_delivery, shop_score_description

# item与pred类型特征
* item_category_split_count, item_property_split_count, pred_category_split_count, pred_property_split_count（长度）
* item_pred_category_score, item_pred_property_score（重合度）
* item_pred_category_score_item%, item_pred_property_score_item%, item_pred_category_score_pred%, item_pred_property_score_pred%（重合度/长度）

# 统计特征（user/item）
user_shop_count, user_item_count, user_context_count,
    user_shop_trade_count, user_item_trade_count,
    user_context_trade_count, user_brand_count, user_city_count,
    user_brand_trade_count, user_city_trade_count, item_occupation_count,
    item_age_count, item_gender_count, item_occupation_trade_count,
    item_age_trade_count, item_gender_trade_count

# 统计特征（item/item）
brand_item_count, city_brand_count, city_item_count

# shop、item得分
* item_score = (sales + collected + pv)
* item_score2 = (sales + collected + pv) / price
* item_score3 = ((sales + collected + pv) / price) / (page + 1)
* shop_score = (star + review_num * review_positive) * service * delivery * description

# 火热程度（搜索次数）
* item_hot, user_hot, shop_hot,
    brand_hot, occupation_hot, city_hot, item_trade_hot,
    user_trade_hot, shop_trade_hot, brand_trade_hot,
    occupation_trade_hot, city_trade_hot

# user购买力、成交率
* user_sell_power, user_sell_trade_power（购买力）
* user_sell_power_mean， user_sell_trade_power_mean（平均购买力）
* user_sell_power_mean_item_minus, user_sell_trade_power_mean_item_minus（平均购买力与商品价格差距）

# 查询次数
 user_day_query, user_yesterday_query, user_hour_query,
    user_yesterhour_query, user_minute_query, item_day_query,
    item_hour_query, user_item_day_query, user_item_hour_query,
    item_minute_query, user_item_minute_query

# user在一个时间段出现的次数序号
user_count, user_day_count, user_hour_count, user_minute_count

# 有没有在历史中出现过
user_is_his, brand_is_his, user_is_trade_his, item_is_trade_his,
    brand_is_trade_his

# 成交率
user_trade_percent, occupation_trade_percent

# 历史数据中出现次数
city_brand_count_his, city_item_count_his

# id转vector
item_id_idx, user_id_idx, shop_id_idx, item_brand_id_idx, item_city_id_idx

# 时间特征
* day, hour, minute, yesterday, yesterhour
* next_time_sub, last_time_sub（user上下次点击时间差）
* user_item_next_time_sub, user_item_last_time_sub（user/item上下次点击时间差）
* user_shop_next_time_sub, user_shop_last_time_sub（user/shop上下次点击时间差）
* user_count_first_time_sub, user_item_count_first_time_sub, user_shop_count_first_time_sub（user/item/shop第一次点击时间差）
* is_last_click, is_last_user_item_click（是否是user/item最后一次点击）
* max_click_time_sub, max_user_item_click_time_sub（与最后一次点击时间差）