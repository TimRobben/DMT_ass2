import pandas as pd

# Load your DataFrame (example path used here)
df = pd.read_csv("E:\\VU\\VU jaar 1\\DMT\\Ass_2\\missing_test_set_country_with_mean_statf_int.csv", delimiter=',')


# Identify columns with float64 data type
float64_columns = df.select_dtypes(include=['float64']).columns

# Convert float64 columns to float32
df[float64_columns] = df[float64_columns].astype('float32')

# Optional: Save the updated DataFrame to a new CSV file
df.to_csv("E:\\VU\\VU jaar 1\\DMT\\Ass_2\\test_converted_to_float32.csv", index=False)
df.info()