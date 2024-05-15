df = pd.read_csv('df', delimiter = ',')

# Impute missing values in numerical features using KNNImputer
knn_imputer = KNNImputer(n_neighbors=9)
df_numeric_imputed = pd.DataFrame(knn_imputer.fit_transform(df[numeric_vars]), columns=numeric_vars)

# Impute missing values in binary features using SimpleImputer with 'most_frequent' strategy
mode_imputer = SimpleImputer(strategy='most_frequent')
df_binary_imputed = pd.DataFrame(mode_imputer.fit_transform(df[binary_vars]), columns=binary_vars)

# Combine the imputed numerical and binary features into a single dataframe
df_knn = pd.concat([df_numeric_imputed, df_binary_imputed], axis=1)

df_knn.isnull().sum()
