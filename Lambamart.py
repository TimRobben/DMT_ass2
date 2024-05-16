import snowflake.connector
import csv
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the training dataset
training_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\missing_imputed_country_id.csv")

# Load the testing dataset
testing_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\dmt-2024-2nd-assignment\\test_set_VU_DM.csv")
# Feature engineering
def feature_engineering(data):
    data['search_month'] = pd.to_datetime(data['date_time']).dt.month
    data['search_day'] = pd.to_datetime(data['date_time']).dt.day
    data['search_hour'] = pd.to_datetime(data['date_time']).dt.hour
    data.fillna(-1, inplace=True)
    return data

train_data = feature_engineering(training_data)
test_data = feature_engineering(testing_data)
# Select features and target variable
features = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 
            'prop_starrating', 'prop_review_score', 'prop_brand_bool', 
            'search_month', 'search_day', 'search_hour', 'orig_destination_distance',
            'price_usd', 'srch_room_count', 'srch_saturday_night_bool']
target = 'relevance_score'
# Balance the dataset by upsampling the minority class
df_majority = train_data[train_data[target] == 0]
df_minority = train_data[train_data[target] > 0]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples=len(df_majority), # to match majority class
                                 random_state=123) # reproducible results

train_balanced = pd.concat([df_majority, df_minority_upsampled])

# Split the balanced training data into training and validation sets
X_train, X_val,Y_train, Y_val = train_test_split(train_balanced, test_size=0.2, random_state=42)

# # Extract features (X) and labels (y) from the training data
# X_train = training_data.drop(columns=['booking_bool'])  # Features
# #y_click_train = training_data['click_bool']  # Click labels
# y_booking_train = training_data['booking_bool']  # Booking labels

# # For testing, we'll use the features from the testing dataset
# X_test = testing_data
qids_train = training_data.groupby("srch_id")["srch_id"].count().to_numpy()
# # Split the training dataset into training and validation sets for click prediction
# #X_click_train, X_click_val, y_click_train, y_click_val = train_test_split(X_train, y_click_train, test_size=0.2, random_state=42)

# # Split the training dataset into training and validation sets for booking prediction
# X_booking_train, X_booking_val, y_booking_train, y_booking_val = train_test_split(X_train, y_booking_train, test_size=0.2, random_state=42)
qids_validation = y_booking_val.groupby("srch_id")["srch_id"].count().to_numpy()


model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
)
model.fit(
    X=X_train,
    y=Y_train,
    eval_set=[(X_val, Y_val)],
    eval_at=10,
    verbose=10,
)