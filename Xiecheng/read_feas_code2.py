#  用CV筛的
# base_cols = ['orderid', 'nag_masterhotelid_ordadvanceday_mean_sub', 'orderdate_min', 'mroom_totalrooms', 'nag_masterhotelid_ordadvanceday_median_sub', 'pos_hotel_ordadvanceday_mean_sub', 'ordadvanceday', 'arrival_day', 'hotel_totalrooms', 'pos_masterbasicroomid_ordadvanceday_median_sub', 'hotelorder_maxdate_sub', 'masterbasicroomid', 'supplierid_crt_capacity', 'city_crt_capacity', 'room', 'pos_masterhotelid_supplierchannel_var', 'hotelorder_mindate_sub', 'orderdate_hour', 'pos_countryid_ordadvanceday_median_sub', 'pos_city_arrival_mon_mean_sub', 'masterhotelid', 'zq_num', 'pos_hotel_ordadvanceday_var', 'pos_masterhotelid_etd_mon_var', 'pos_hotel_ordadvanceday_median_sub', 'ord_cnt_in_30day_orderdate', 'room_hot', 'pos_hotel_ordadvanceday_max', 'pos_masterhotelid_ordadvanceday_mean', 'mhotel_include_masterbasicroomid_cnt', 'hotel_hot', 'pos_city_ordadvanceday_median_sub', 'pos_masterhotelid_orderdate_mon_mean_sub', 'glon', 'in_city_condi_supplierid', 'nag_city_hotelstar_mean', 'pos_hotel_orderdate_mon_var', 'etd_day', 'nag_masterhotelid_orderdate_mon_var', 'pos_masterhotelid_orderdate_mon_var', 'ord_cnt_in_60day_orderdate', 'pos_hotel_etd_mon_var', 'nag_hotel_ordadvanceday_median_sub', 'countryid_crt_capacity', 'masterhotelid_crt_capacity', 'in_city_condi_star', 'city_capacity_dist', 'pos_masterhotelid_supplierchannel_mean', 'nag_masterhotelid_ordadvanceday_mean', 'pos_masterhotelid_isebookinghtl_mean', 'nag_city_arrival_mon_var', 'hotel', 'arrival_dow', 'supplierid_capacity_dist', 'pos_masterhotelid_hotelbelongto_var', 'ord_cnt_in_1day_orderdate', 'pos_masterhotelid_hotelbelongto_mean_sub', 'pos_hotel_arrival_mon_var', 'in_city_condi_masterhotelid', 'nag_hotel_ordadvanceday_mean_sub', 'pos_masterhotelid_hotelbelongto_mean', 'glat', 'room_num', 'nag_masterhotelid_etd_mon_var', 'city', 'pos_hotel_etd_mon_mean_sub', 'pos_masterhotelid_ordadvanceday_max', 'in_countryid_condi_star', 'pos_city_hotelbelongto_mean', 'etd_dow']
# watch_score = pickle.load(open('../data/watch_scores.pk', 'rb'))

# 用后4天做验证集筛的
basic_cols = ['orderdate_min', 'zone_ordadvanceday_bayes_rate_month', 'orderdate_hour', 'ordadvanceday_masterhotelid_bayes_rate_full', 'hotel_isholdroom_bayes_rate_month', 'ordadvanceday_masterbasicroomid_bayes_rate_full', 'orderid', 'city_ordadvanceday_bayes_rate_month', 'ordadvanceday_hotelstar_bayes_rate_month', 'ordadvanceday_supplierid_bayes_rate_full', 'nag_masterhotelid_ordadvanceday_mean_sub', 'zone_ordadvanceday_bayes_rate_full', 'ordadvanceday_supplierid_bayes_rate_month', 'countryid_supplierid_bayes_rate_month', 'city_ordadvanceday_bayes_rate_full', 'city_supplierid_bayes_rate_month', 'ordadvanceday_masterhotelid_bayes_rate_month', 'countryid_ordadvanceday_bayes_rate_month', 'ordadvanceday_isvendor_bayes_rate_month', 'isholdroom_supplierid_bayes_rate_month', 'zone_masterhotelid_bayes_rate_month', 'countryid_ordadvanceday_bayes_rate_full', 'masterbasicroomid_isvendor_bayes_rate_full', 'hotel_masterbasicroomid_bayes_rate_month', 'nag_masterhotelid_ordadvanceday_median_sub', 'city_masterbasicroomid_bayes_rate_full', 'masterbasicroomid_isvendor_bayes_rate_month', 'hotel_totalrooms', 'pos_hotel_ordadvanceday_mean_sub', 'ordadvanceday_isvendor_bayes_rate_full', 'ordadvanceday_hotelbelongto_bayes_rate_full', 'hotel_masterbasicroomid_bayes_rate_full', 'city_supplierid_bayes_rate_full', 'masterbasicroomid_supplierchannel_bayes_rate_month', 'masterbasicroomid_hotelbelongto_bayes_rate_full', 'city_masterbasicroomid_bayes_rate_month', 'ord_cnt_in_180day_etd', 'ord_cnt_in_1day_etd', 'isholdroom_masterhotelid_bayes_rate_month', 'room_masterhotelid_bayes_rate_full', 'masterbasicroomid_isebookinghtl_bayes_rate_month', 'zq_num', 'room', 'isholdroom_ordadvanceday_bayes_rate_month', 'nag_masterhotelid_orderdate_mon_var', 'city_crt_capacity', 'in_city_condi_masterhotelid', 'masterbasicroomid_isebookinghtl_bayes_rate_full', 'pos_countryid_ordadvanceday_median_sub', 'hotel_isholdroom_bayes_rate_full', 'hotel_room_bayes_rate_month', 'pos_hotel_ordadvanceday_median_sub', 'mhotel_include_masterbasicroomid_cnt', 'ordadvanceday_supplierchannel_bayes_rate_month', 'room_isholdroom_bayes_rate_full', 'ordadvanceday_isebookinghtl_bayes_rate_month', 'supplierid_crt_capacity', 'nag_masterhotelid_ordadvanceday_var', 'masterhotelid_isebookinghtl_bayes_rate_full', 'ordadvanceday_hotelstar_bayes_rate_full']
watch_score = pickle.load(open('../data/watch_scores2.pk', 'rb'))

need_cols = basic_cols + [item[0] for item in watch_score[:-300]]
print(len(need_cols))
full, test_slice, test_a_slice, test_b_slice = get_full(need_cols)
print(full.shape)