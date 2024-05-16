# -*- coding: utf-8 -*-
"""feature_engineering.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lqwqN0y9VOCIifVLIaRYgz9GTJc9_caU
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import scipy as stats

df = pd.read_csv("mean_mode_imputed_data.csv", delimiter=',')
df.head()

"""1.  Winnaars gebruiken alle numerical features
2.  Een van de belangrijkste factoren voor consumers is de prijs:
- price rank tov hotels in de omgeving (prop_coutry_id) CHECK
- difference between hotel price and total price CHECK
price different of current price and when booked previously CHECK
3. Historical/composite features:
- average star rating of previous booked hotels BY ADDING MEAN
- average price of previous booked hotels BY ADDING MEAN
- interaction feature: hoeveel kamers*length stay om inzicht te krijgen op hoe “groot” de nodige boeking is CHECK
- rating difference of overall star rating and from customer CHECK
4. hotel position toevoegen als feature omdat dit niet in de test data zit
5. statistical features

PRICE FEATURES
"""

# adds new column 'price_rank_country' to store rank of each hotel's price within the same country/destination
df['price_rank_country'] = df.groupby(['prop_country_id'])['price_usd'].rank(ascending=True)

"""HISTORICAL/COMPOSITE FEATURES

"""

# adds new column to know the total room/length of the stay searched
df['total_room_stay'] = df['srch_room_count'] * df['srch_length_of_stay']

# define numeric columns to apply statistical functions
numeric_columns = ['prop_starrating',
                   'prop_review_score', 'prop_location_score1', 'prop_location_score2',
                   'prop_log_historical_price', 'price_usd', 'promotion_flag',
                   'srch_length_of_stay',
                   'srch_booking_window', 'srch_adults_count', 'srch_children_count',
                   'srch_room_count', 'orig_destination_distance',
                   'comp_rate', 'comp_inv', 'comp_rate_percent_diff']

# Group by 'srch_id' and compute statistical features
statistical_features = df.groupby('srch_id')[numeric_columns].agg(['mean', 'median', 'min', 'max'])

# Flatten the multi-level column index
statistical_features.columns = ['{}_{}'.format(col, stat) for col, stat in statistical_features.columns]

# Merge the statistical features back into the original dataframe
df = pd.merge(df, statistical_features, on='srch_id', how='left')

df.head()

"""ADDED POSITION FEATURE

"""

# Step 1: Compute mean position for each property
mean_position = df.groupby('prop_id')['position'].mean()

# Step 2: Compute standard deviation of position for each property
std_dev_position = df.groupby('prop_id')['position'].std()

# Step 3: Exclude searches with random ordering
filtered_data = df[df['random_bool'] == 0]

# Step 4: Incorporate prior probability of clicking or booking a property
click_prob = filtered_data.groupby('prop_id')['click_bool'].mean() / (filtered_data.groupby('prop_id')['click_bool'].count() - 1)
booking_prob = filtered_data.groupby('prop_id')['booking_bool'].mean() / (filtered_data.groupby('prop_id')['booking_bool'].count() - 1)

# Add the features to the training data
df['mean_position'] = df['prop_id'].map(mean_position)
df['std_dev_position'] = df['prop_id'].map(std_dev_position)
df['click_prob'] = df['prop_id'].map(click_prob)
df['booking_prob'] = df['prop_id'].map(booking_prob)

# Additional feature: Number of previous search results containing the hotel
num_prev_results = filtered_data.groupby('prop_id').size() - 1
df['num_prev_results'] = df['prop_id'].map(num_prev_results)

df.head()

"""NORMALIZE NUMERICAL FEATURES"""

from sklearn.preprocessing import StandardScaler

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with any NaN values
df.dropna(inplace=True)

# Define columns to exclude from normalization
cols_to_exclude = ['srch_id', 'click_bool', 'booking_bool', 'date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'random_bool', 'srch_destination_id', 'comp_rate', 'comp_inv']

# Columns to be normalized
cols_to_normalize = df.columns.difference(cols_to_exclude)

# Initialize the StandardScaler
scaler = StandardScaler()

# Normalize the data
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Save the updated training data
df.to_csv("updated_training_data.csv", index=False)

df.head()
