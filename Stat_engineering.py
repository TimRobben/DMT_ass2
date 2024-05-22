import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import scipy as stats
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\missing_test_set_country_without_statf.csv", delimiter=',')

numeric_columns = ['prop_starrating',
                   'prop_review_score', 'prop_location_score1', 'prop_location_score2',
                   'prop_log_historical_price', 'price_usd', 'promotion_flag',
                   'srch_length_of_stay',
                   'srch_booking_window','srch_room_count', 'orig_destination_distance',
                   'comp_rate', 'comp_inv', 'comp_rate_percent_diff']

# Group by 'srch_id' and compute statistical features
statistical_features = df.groupby('srch_id')[numeric_columns].agg(['mean'])
#statistical_features = statistical_features.applymap(lambda x: int(round(x)))

# Flatten the multi-level column index
statistical_features.columns = ['{}_{}'.format(col, stat) for col, stat in statistical_features.columns]

# Merge the statistical features back into the original dataframe
df = pd.merge(df, statistical_features, on='srch_id', how='left')

# train set
cols_to_exclude = ['srch_id', 'click_bool', 'booking_bool', 'date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'random_bool', 'srch_destination_id', 'comp_rate', 'comp_inv']

# test set
#cols_to_exclude = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'random_bool', 'srch_destination_id', 'comp_rate', 'comp_inv']

# Columns to be normalized
cols_to_normalize = df.columns.difference(cols_to_exclude)

# Initialize the StandardScaler
scaler = StandardScaler()

# Normalize the data
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Save the updated training data
df.to_csv("E:\VU\VU jaar 1\DMT\Ass_2\missing_test_set_country_with_mean_statf_int.csv", index=False)
print("saved")
print("info:")
df.info()