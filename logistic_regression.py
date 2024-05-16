import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import ndcg_score

# Load training and test datasets
train_data = pd.read_csv('missing_imputed_country_id.csv', delimiter=',')
test_data = pd.read_csv('test_set_VU_DM.csv', delimiter = ',')

# Feature engineering
def feature_engineering(data):
    data['search_month'] = pd.to_datetime(data['date_time']).dt.month
    data['search_day'] = pd.to_datetime(data['date_time']).dt.day
    data['search_hour'] = pd.to_datetime(data['date_time']).dt.hour
    data.fillna(-1, inplace=True)
    return data

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

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
train_set, val_set = train_test_split(train_balanced, test_size=0.2, random_state=42)

# Standardize features
#scaler = StandardScaler()
#train_set[features] = scaler.fit_transform(train_set[features])
#val_set[features] = scaler.transform(val_set[features])
#test_data[features] = scaler.transform(test_data[features])

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(train_set[features], train_set[target])

# Evaluate the model on the validation set
val_set['predicted_score'] = model.predict_proba(val_set[features])[:, 1]

# Generate predictions for the test dataset
test_data['predicted_score'] = model.predict_proba(test_data[features])[:, 1]

# Create the submission file
submission = test_data[['srch_id', 'prop_id', 'predicted_score']]
submission = submission.sort_values(by=['srch_id', 'predicted_score'], ascending=[True, False])

# Select only the required columns for submission
submission = submission[['srch_id', 'prop_id']]

# score of 0.15
