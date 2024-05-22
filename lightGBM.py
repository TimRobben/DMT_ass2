import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import ndcg_score
from relevance_score import relevance_score

# Load the training dataset
training_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\\train_converted_to_float32.csv")
#print(training_data.columns, training_data.shape, training_data.info())

# Load the testing dataset
testing_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\\test_converted_to_float32.csv")


# Identify columns with float64 data type inferred by pandas (these should be the original float32 columns)
float64_columns_train = training_data.select_dtypes(include=['float64']).columns
float64_columns_test = testing_data.select_dtypes(include=['float64']).columns
# Create a dtype dictionary to enforce float32 for these columns
dtype_dict = {col: 'float32' for col in float64_columns_train}
dtype_dict = {col: 'float32' for col in float64_columns_test}
# Read the CSV file again, this time with the dtype dictionary
training_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\\train_converted_to_float32.csv", dtype=dtype_dict)
testing_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\\test_converted_to_float32.csv", dtype=dtype_dict)
training_data = relevance_score(training_data)
training_data.info()
testing_data.info()
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
features = test_data.columns.tolist()
#features.remove('relevance_score')
features.remove('date_time')
features.remove('predicted_score')
features.remove('promotion_flag')
features.remove('srch_saturday_night_bool')
features.remove('prop_location_score2')
#print(features)
target = 'relevance_score'

# Balance the dataset by upsampling the minority class
df_majority = train_data[train_data[target] == 0]
df_minority = train_data[train_data[target] > 0]

df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minority),  # to match minority class
                                   random_state=123)  # reproducible results

train_balanced = pd.concat([df_majority_downsampled, df_minority])
print("Woohoo we zijn voorbij het voorbereiden!!!!")
# Split the balanced training data into training and validation sets
train_set, val_set = train_test_split(train_balanced, test_size=0.2, random_state=42)

# Prepare data for LightGBM
train_lgb = lgb.Dataset(train_set[features], label=train_set[target], group=train_set.groupby('srch_id').size().to_numpy())
val_lgb = lgb.Dataset(val_set[features], label=val_set[target], group=val_set.groupby('srch_id').size().to_numpy(), reference=train_lgb)



# Set parameters for LightGBM
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5],
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 28,
    'feature_fraction': 0.927,
    'bagging_fraction': 0.985,
    'bagging_freq': 18,
    'verbosity': -1,
    'max_depth': 9
}

# Define a callback for early stopping
early_stopping_callback = lgb.early_stopping(stopping_rounds=50)

# Train the LightGBM model
model = lgb.train(params, train_lgb, valid_sets=[train_lgb, val_lgb], num_boost_round=1000, callbacks=[early_stopping_callback])

# Evaluate the model on the validation set
val_set['predicted_score'] = model.predict(val_set[features], num_iteration=model.best_iteration)


# Generate predictions for the test dataset
test_data['predicted_score'] = model.predict(test_data[features], num_iteration=model.best_iteration)

# Create the submission file
submission = test_data[['srch_id', 'prop_id', 'predicted_score']]
submission = submission.sort_values(by=['srch_id', 'predicted_score'], ascending=[True, False])

# Select only the required columns for submission
submission = submission[['srch_id', 'prop_id']]
submission.to_csv('lambdamart_submission.csv', index=False)