import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import ndcg_score
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

# Define batch size
batch_size = 10000

# Initialize a list to store models
models = []

# Define parameters for LightGBM
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

# Initialize a callback for early stopping
early_stopping_callback = lgb.early_stopping(stopping_rounds=50)

# Iterate over the dataset in batches
num_batches = len(training_data) // batch_size
for i in range(num_batches):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch_data = training_data.iloc[batch_start:batch_end]

    # Balance the batch by upsampling the minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=123)
    batch_balanced = pd.concat([df_majority, df_minority_upsampled])

    # Split the batch into training and validation sets
    train_batch, val_batch = train_test_split(batch_balanced, test_size=0.2, random_state=42)

    # Prepare data for LightGBM
    train_lgb = lgb.Dataset(train_batch[features], label=train_batch[target], group=train_batch.groupby('srch_id').size().to_numpy())
    val_lgb = lgb.Dataset(val_batch[features], label=val_batch[target], group=val_batch.groupby('srch_id').size().to_numpy(), reference=train_lgb)

    # Train the LightGBM model
    model = lgb.train(params, train_lgb, valid_sets=[train_lgb, val_lgb], num_boost_round=1000, callbacks=[early_stopping_callback])

    # Append the trained model to the list
    models.append(model)

# Load the testing dataset
testing_data = pd.read_csv("E:/VU/VU jaar 1/DMT/Ass_2/dmt-2024-2nd-assignment/test_set_VU_DM.csv")
test_data = feature_engineering(testing_data)

# Generate predictions for the test dataset using each trained model
test_data['predicted_score'] = 0
for model in models:
    test_data['predicted_score'] += model.predict(test_data[features], num_iteration=model.best_iteration)

# Average the predictions from all models
test_data['predicted_score'] /= len(models)

# Create the submission file
submission = test_data[['srch_id', 'prop_id', 'predicted_score']]
submission = submission.sort_values(by=['srch_id', 'predicted_score'], ascending=[True, False])
submission[['srch_id', 'prop_id']].to_csv('lambdamart_submission.csv', index=False)
