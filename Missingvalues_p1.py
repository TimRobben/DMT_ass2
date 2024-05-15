import pandas as pd
import numpy as np

df = pd.read_csv("training_set_VU_DM.csv", delimiter=',')

#Convert columns to numeric data type
comp_inv_columns = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']
df[comp_inv_columns] = df[comp_inv_columns].apply(pd.to_numeric, errors='coerce')

# Combine competitor availability fields
df['comp_inv'] = np.where(df[comp_inv_columns].sum(axis=1) > 0, 1, 0)

# Combine competitor rate fields
comp_rate_columns = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']
df['comp_rate'] = df[comp_rate_columns].min(axis=1)

# Combine competitor rate percent diff fields
comp_rate_percent_diff_columns = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
df['comp_rate_percent_diff'] = df[comp_rate_percent_diff_columns].mean(axis=1)

# Optionally, you can drop the original competitor fields if you don't need them anymore
df.drop(columns=comp_inv_columns + comp_rate_columns + comp_rate_percent_diff_columns, inplace=True)


# Define columns with high proportions of missing values
columns_with_high_missing = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_query_affinity_score',
                             'gross_bookings_usd']
# Drop columns with high proportions of missing values
df.drop(columns_with_high_missing, axis=1, inplace=True)
