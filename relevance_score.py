import pandas as pd

# Create a new column with initial value 0
df['relevance_score'] = 0

# Set relevance score to 5 where booking_bool is 1
df.loc[df['booking_bool'] == 1, 'relevance_score'] = 5

# Set relevance score to 1 where click_bool is 1 and booking_bool is 0
df.loc[(df['click_bool'] == 1) & (df['booking_bool'] == 0), 'relevance_score'] = 1

# Calculate the percentage of rows with relevance_score above zero
percentage_positive_relevance = (df['relevance_score'] > 0).mean() * 100

# Display the percentage
print("Percentage of rows with relevance_score above zero:", percentage_positive_relevance)
