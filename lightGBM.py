import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from relevance_score import relevance_score

# Load the training dataset
training_data = pd.read_csv("E:/VU/VU jaar 1/DMT/Ass_2/country_id_updated_training_data.csv")
training_data = relevance_score(training_data)

# Feature engineering
def feature_engineering(data):
    data['search_month'] = pd.to_datetime(data['date_time']).dt.month
    data['search_day'] = pd.to_datetime(data['date_time']).dt.day
    data['search_hour'] = pd.to_datetime(data['date_time']).dt.hour
    data.fillna(-1, inplace=True)
    return data

# Select features and target variable
features = training_data.columns.tolist()
features.remove('relevance_score')
target = 'relevance_score'

# Balance the dataset by upsampling the minority class
df_majority = training_data[training_data[target] == 0]
df_minority = training_data[training_data[target] > 0]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,    
                                 n_samples=len(df_majority),
                                 random_state=123)

train_balanced = pd.concat([df_majority, df_minority_upsampled])

# Define batch size
batch_size = 10000

# Split the balanced training data into batches
num_batches = len(train_balanced) // batch_size
for i in range(num_batches):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch_data = train_balanced.iloc[batch_start:batch_end]

    # Prepare data for LightGBM
    batch_group = batch_data.groupby('srch_id').size().to_numpy()
    batch_train_lgb = lgb.Dataset(batch_data[features], label=batch_data[target], group=batch_group)

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

    # Train the LightGBM model
    model = lgb.train(params, batch_train_lgb, num_boost_round=1000)

    # Save the model (optional)
    model.save_model(f'lgb_model_batch_{i}.txt')

# Process the remaining data (if any)
remaining_data = train_balanced.iloc[num_batches * batch_size:]
if not remaining_data.empty:
    remaining_group = remaining_data.groupby('srch_id').size().to_numpy()
    remaining_train_lgb = lgb.Dataset(remaining_data[features], label=remaining_data[target], group=remaining_group)
    model = lgb.train(params, remaining_train_lgb, num_boost_round=1000)
    model.save_model(f'lgb_model_batch_{num_batches}.txt')

# Load the testing dataset
testing_data = pd.read_csv("E:/VU/VU jaar 1/DMT/Ass_2/dmt-2024-2nd-assignment/test_set_VU_DM.csv")
test_data = feature_engineering(testing_data)

# Generate predictions for the test dataset
test_data['predicted_score'] = model.predict(test_data[features], num_iteration=model.best_iteration)

# Create the submission file
submission = test_data[['srch_id', 'prop_id', 'predicted_score']]
submission = submission.sort_values(by=['srch_id', 'predicted_score'], ascending=[True, False])
submission[['srch_id', 'prop_id']].to_csv('lambdamart_submission.csv', index=False)
