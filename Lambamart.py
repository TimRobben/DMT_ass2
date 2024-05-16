import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from relevance_score import relevance_score
# Load the training dataset
training_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\missing_imputed_country_id.csv")
training_data = relevance_score(training_data)
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
qids_train = train_balanced.groupby("srch_id")["srch_id"].count().to_numpy()
qids_test = test_data.groupby("srch_id")["srch_id"].count().to_numpy()
# Split the balanced training data into training and validation sets
train_set, val_set = train_test_split(train_balanced, test_size=0.2, random_state=42)
print(train_set[features].shape())
print(train_set[target].shape())
print(val_set[features].shape())
print(val_set[target].shape())
# model = lgb.LGBMRanker(
#     objective="lambdarank",
#     metric="ndcg",
#     verbose=10
# )
# model.fit(
#     X=train_set[features],
#     y=train_set[target],
#     group=qids_train,
#     eval_set=[(val_set[features], val_set[target])],
#     eval_group=[qids_test],
#     eval_at=10,
# )