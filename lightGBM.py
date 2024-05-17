import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import ndcg_score
from relevance_score import relevance_score

# Load the training dataset
training_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\missing_imputed_country_id_without_statf_int.csv")
print(training_data.columns, training_data.shape, training_data.info())
training_data = relevance_score(training_data)
# Load the testing dataset
testing_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\missing_test_set_country_without_statf.csv")
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
print(features)
target = 'relevance_score'

# Balance the dataset by upsampling the minority class
df_majority = train_data[train_data[target] == 0]
df_minority = train_data[train_data[target] > 0]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples=len(df_majority), # to match majority class
                                 random_state=123) # reproducible results

train_balanced = pd.concat([df_majority, df_minority_upsampled])
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
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1
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