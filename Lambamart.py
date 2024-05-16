import snowflake.connector
import csv
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the training dataset
training_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\missing_imputed_country_id.csv")

# Load the testing dataset
testing_data = pd.read_csv("E:\VU\VU jaar 1\DMT\Ass_2\dmt-2024-2nd-assignment\\test_set_VU_DM.csv")

# Extract features (X) and labels (y) from the training data
X_train = training_data.drop(columns=['booking_bool'])  # Features
#y_click_train = training_data['click_bool']  # Click labels
y_booking_train = training_data['booking_bool']  # Booking labels

# For testing, we'll use the features from the testing dataset
X_test = testing_data
qids_train = training_data.groupby("srch_id")["srch_id"].count().to_numpy()
# Split the training dataset into training and validation sets for click prediction
#X_click_train, X_click_val, y_click_train, y_click_val = train_test_split(X_train, y_click_train, test_size=0.2, random_state=42)

# Split the training dataset into training and validation sets for booking prediction
X_booking_train, X_booking_val, y_booking_train, y_booking_val = train_test_split(X_train, y_booking_train, test_size=0.2, random_state=42)
qids_validation = y_booking_val.groupby("srch_id")["srch_id"].count().to_numpy()


model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
)
model.fit(
    X=X_booking_train,
    y=y_booking_train,
    group=qids_train,
    eval_set=[(X_booking_val, y_booking_val)],
    eval_group=[qids_validation],
    eval_at=10,
    verbose=10,
)