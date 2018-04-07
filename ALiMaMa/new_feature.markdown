# 基础特征
* id转vector
item_id_idx, user_id_idx, shop_id_idx, item_brand_id_idx, item_city_id_idx（ZERO）

# item与pred类型特征
* item_category_split_count, item_property_split_count, pred_category_split_count, pred_property_split_count（长度）
* item_pred_category_score, item_pred_property_score（重合长度）
* item_pred_category_score_item%, item_pred_property_score_item%, item_pred_category_score_pred%, item_pred_property_score_pred%（重合长度/长度）
（ZERO）
* category_%d, 这一行第2,3个category
* property_%d, 前十个property
* query_category_%d, 用户查询前6个category
* query_col_%d, 商品是否具有用户查询的[0,13]个category
* common_category_prob2, 同common_category_prob, 去除了查询中category未出现在item属性中的部分
* common_property_prob2, 同common_property_prob, 去除了查询中property未出现在item属性中的部分
（CJF）
* （24）使用 starspace 将商品的cat进行embedding（dim=24）
  'item_cat_vec_0', 'item_cat_vec_1', 'item_cat_vec_2', 'item_cat_vec_3', 'item_cat_vec_4', 'item_cat_vec_5', 'item_cat_vec_6', 'item_cat_vec_7', 'item_cat_vec_8', 'item_cat_vec_9', 'item_cat_vec_10', 'item_cat_vec_11', 'item_cat_vec_12', 'item_cat_vec_13', 'item_cat_vec_14', 'item_cat_vec_15', 'item_cat_vec_16', 'item_cat_vec_17', 'item_cat_vec_18', 'item_cat_vec_19', 'item_cat_vec_20', 'item_cat_vec_21', 'item_cat_vec_22', 'item_cat_vec_23',
* （24）使用 starspace 将预测的cat进行embedding（dim=24）
   'pred_cat_mean_0', 'pred_cat_mean_1', 'pred_cat_mean_2', 'pred_cat_mean_3', 'pred_cat_mean_4', 'pred_cat_mean_5', 'pred_cat_mean_6', 'pred_cat_mean_7', 'pred_cat_mean_8', 'pred_cat_mean_9', 'pred_cat_mean_10', 'pred_cat_mean_11', 'pred_cat_mean_12', 'pred_cat_mean_13', 'pred_cat_mean_14', 'pred_cat_mean_15', 'pred_cat_mean_16', 'pred_cat_mean_17', 'pred_cat_mean_18', 'pred_cat_mean_19', 'pred_cat_mean_20', 'pred_cat_mean_21', 'pred_cat_mean_22', 'pred_cat_mean_23', 
* 'item_pred_cat_cos', 'item_pred_ppt_cos' cos(商品的 cat/ppt embedding vector，预测的 cat/ppt embedding vector)
（CJY）

# 统计特征
* user_shop_count, user_item_count, user_context_count,
    user_shop_trade_count, user_item_trade_count,
    user_context_trade_count, user_brand_count, user_city_count,
    user_brand_trade_count, user_city_trade_count, item_occupation_count,
    item_age_count, item_gender_count, item_occupation_trade_count,
    item_age_trade_count, item_gender_trade_count
* brand_item_count, city_brand_count, city_item_count
（ZERO 只针对历史 --）
* 'shop_item_cnt', 'shop_brand_cnt'
（CJY）

# shop、item得分
* item_score = (sales + collected + pv)
* item_score2 = (sales + collected + pv) / price
* item_score3 = ((sales + collected + pv) / price) / (page + 1)
* shop_score = (star + review_num * review_positive) * service * delivery * description
（ZERO ！！）

# 火热程度（搜索次数）
* item_hot, user_hot, shop_hot,
    brand_hot, occupation_hot, city_hot, item_trade_hot,
    user_trade_hot, shop_trade_hot, brand_trade_hot,
    occupation_trade_hot, city_trade_hot
（ZERO --）
'24h_cat_hot', '24h_ppt_hot', 24h_user_seem_times（划窗）
（CJY）

# user购买力、成交率
* user_sell_power, user_sell_trade_power（购买力）
* user_sell_power_mean， user_sell_trade_power_mean（平均购买力）
* user_sell_power_mean_item_minus, user_sell_trade_power_mean_item_minus（平均购买力与商品价格差距）
（ZERO --）

# 查询次数
 user_day_query, user_yesterday_query, user_hour_query,
    user_yesterhour_query, user_minute_query, item_day_query,
    item_hour_query, user_item_day_query, user_item_hour_query,
    item_minute_query, user_item_minute_query
（ZERO ！）

# user在一个时间段出现的次数序号
user_count, user_day_count, user_hour_count, user_minute_count
（ZERO ！）


# 有没有在历史中出现过
user_is_his, brand_is_his, user_is_trade_his, item_is_trade_his,
    brand_is_trade_his
（ZERO --）

# 成交率
user_trade_percent, occupation_trade_percent
（ZERO --）

# 历史数据中出现次数
city_brand_count_his, city_item_count_his
（ZERO --）

# 时间特征
* day, hour, minute, yesterday, yesterhour
* next_time_sub, last_time_sub（user上下次点击时间差）
* user_item_next_time_sub, user_item_last_time_sub（user/item上下次点击时间差）
* user_shop_next_time_sub, user_shop_last_time_sub（user/shop上下次点击时间差）
* user_count_first_time_sub, user_item_count_first_time_sub, user_shop_count_first_time_sub（user/item/shop第一次点击时间差）
* is_last_click, is_last_user_item_click（是否是user/item最后一次点击）
* max_click_time_sub, max_user_item_click_time_sub（与最后一次点击时间差）
（ZERO ！！）

# star用户喜爱程度
   'star_category_max', 'star_category_min', 'star_category_mean', 'star_category_var', 'star_property_max', 'star_property_min', 'star_property_mean', 'star_property_var'（
先计算不同 star level 的用户对各个 cat/ppt 的喜爱程度， 然后计算当前 star level 的用户对预测出的 cat/ppt list 的喜欢程度的 max、min、mean、var）
（CJY）

# 平均值编码
   'item_brand_id+item_price_level_pred_1', 'shop_id_pred_1', 'item_id_pred_1', 'item_sales_level+item_collected_level_pred_1', 'shop_star_level+shop_item_cnt_pred_1', 'shop_review_num_level+shop_item_cnt_pred_1', 'item_price_level+item_pv_level_pred_1', 'item_price_level_pred_1', 'item_sales_level+item_pv_level_pred_1', 'shop_item_cnt+shop_brand_cnt_pred_1', 'user_age_level+user_star_level_pred_1', '24h_user_seem_times_pred_1', 'item_brand_id+shop_brand_cnt_pred_1', 'item_collected_level+item_id_pred_1', 'item_price_level+item_pred_cat_cos_pred_1', 'item_brand_id+hist_hour_ctr_pred_1', 'item_brand_id+shop_score_description_pred_1', 'item_city_id+context_page_id_pred_1', 'item_pv_level+shop_review_positive_rate_pred_1', 'item_id+hist_hour_ctr_pred_1', 'item_price_level+star_property_min_pred_1', 'user_star_level+shop_id_pred_1', 'shop_item_cnt+24h_ppt_hot_pred_1', 'item_brand_id+star_property_max_pred_1', 'user_gender_id+star_category_min_pred_1', 'shop_item_cnt+star_property_max_pred_1', 'item_sales_level+context_page_id_pred_1', 'item_collected_level+shop_score_description_pred_1', 'shop_review_num_level+hist_hour_ctr_pred_1', 'item_collected_level+user_gender_id_pred_1', 'shop_item_cnt+star_property_var_pred_1', '24h_ppt_hot+star_property_max_pred_1', 'user_star_level+star_property_max_pred_1', 'user_age_level+user_id_pred_1'
（CJY）

# 贝叶斯平滑
'item_pv_stair_shop_id_bayes_rate_2',
 'item_city_id_shop_id_bayes_rate_2',
 'item_brand_id_shop_review_num_level_bayes_rate_1',
 'item_sales_stair_user_age_stair_bayes_rate_1',
 'time_slice_shop_review_num_level_bayes_rate_1',
 'item_collected_level_bayes_rate_1'
 （CJF）

# 分级特征
* gender_filled, 用户是否填写了性别 0未填写 1填写

* user_age_stair, 根据用户年龄和转化率分级
* user_occupation_stair, 根据用户职业和转化率分级
* user_star_stair, 根据用户星级和转化率 分级
* context_page_stair, 根据广告page_id和转化率分级
* hour_stair, 根据小时数和转化率分级
* item_price_stair, 根据商品价格level和转化率分级
* item_sales_stair, 根据商品销售level和转化率分级
* item_collected_stair, 根据商品收藏level和转化率分级
* item_pv_stair, 根据商品PVlevel和转化率分级

# 最大最小特征/局部最优特征
* user_query_cheapest, 用户查询所有结果中，item_price_level最小的
* user_query_maxsell, 同上.item_sales_level最大
* user_query_maxcollect, 同.item_collected_level最大
* user_query_maxpv, 同.item_pv_level
* user_query_best_service, 同.shop_score_service
* user_query_best_delivery, 同.shop_score_delivery
* user_query_best_description, 同.shop_score_description
* user_query_maxshopstar, 同.shop_star_level
* user_query_maxreview, 同.shop_review_num_level
* user_query_maxgoodreview, 同.shop_review_positive_rate
* user_query_maxqueryitem_c_similarity, 同.  common_category_prob（自造特征）
* user_query_maxqueryitem_p_similarity, 同.  common_property_prob
* user_query_maxqueryitem_c_similarity2, 同. common_category_prob2
* user_query_maxqueryitem_p_similarity2, 同. common_property_prob2