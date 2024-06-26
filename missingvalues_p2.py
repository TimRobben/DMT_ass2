import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error

df = pd.read_csv('df', delimiter = ',')

binary_vars = ['prop_brand_bool', 'promotion_flag', 'srch_saturday_night_bool', 'random_bool', 'comp_inv', 'booking_bool', 'click_bool']
numeric_vars = ['srch_id', 'site_id','visitor_location_country_id', 'prop_country_id', 'prop_id',
                 'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2',
                'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay',
                 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'orig_destination_distance',
                'comp_rate', 'comp_rate_percent_diff', 'position']

# Impute missing values with mean/mode for all variables
mean_imputer = SimpleImputer(strategy='mean')  # For numeric variables
mode_imputer = SimpleImputer(strategy='most_frequent')  # For binary variables
df_mean_mode = df.copy()
df_mean_mode[numeric_vars] = mean_imputer.fit_transform(df[numeric_vars])
df_mean_mode[binary_vars] = mode_imputer.fit_transform(df[binary_vars])

df_mean_mode.isnull().sum()
# Impute missing values in numerical features using KNNImputer
knn_imputer = KNNImputer(n_neighbors=9)
df_numeric_imputed = pd.DataFrame(knn_imputer.fit_transform(df[numeric_vars]), columns=numeric_vars)

# Impute missing values in binary features using SimpleImputer with 'most_frequent' strategy
mode_imputer = SimpleImputer(strategy='most_frequent')
df_binary_imputed = pd.DataFrame(mode_imputer.fit_transform(df[binary_vars]), columns=binary_vars)

# Combine the imputed numerical and binary features into a single dataframe
df_knn = pd.concat([df_numeric_imputed, df_binary_imputed], axis=1)

df_knn.isnull().sum() # KNN 9 gives best result!

# date_time is deleted from the column so add it back
old_df = pd.read_csv("traning_df", delimiter = ',')
new_df = pd.read_csv("data", delimiter=',') # add mean/mode or knn/mode here
merged_df = pd.merge(new_df, old_df[['srch_id', 'date_time']], on = 'srch_id', how = 'left')

