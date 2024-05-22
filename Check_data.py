import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import ndcg_score
from relevance_score import relevance_score

# Load the training dataset
training_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\missing_test_set_country_with_mean_statf_int.csv")

#training_data = relevance_score(training_data)

# Feature engineering
def feature_engineering(data):
    data['search_month'] = pd.to_datetime(data['date_time']).dt.month
    data['search_day'] = pd.to_datetime(data['date_time']).dt.day
    data['search_hour'] = pd.to_datetime(data['date_time']).dt.hour
    data.fillna(-1, inplace=True)
    return data

train_data = feature_engineering(training_data)
#test_data = feature_engineering(testing_data)
train_data.iloc[:, 27:] = train_data.iloc[:, 27:].astype(int)

# Save the DataFrame to a CSV file
train_data.to_csv('E:\VU\VU jaar 1\DMT\Ass_2\missing_test_set_country_with_mean_statf_int.csv', index=False)
print("saved")
train_data.columns
print("info:")
train_data.info()

# features = training_data.columns.tolist()
# features.remove('relevance_score')
# target = 'relevance_score'
# # Balance the dataset by upsampling the minority class
# df_majority = training_data[training_data[target] == 0]
# df_minority = training_data[training_data[target] > 0]
# # Balance the batch by upsampling the minority class
# df_minority_upsampled = resample(df_minority, 
#                                     replace=True,
#                                     n_samples=len(df_majority),
#                                     random_state=123)
# batch_balanced = pd.concat([df_majority, df_minority_upsampled])
