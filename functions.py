import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv('iris/iris.data', header=None, names=column_names)

"""basic missing data generation function randomly assigns missing values without following a specific structural pattern.
"""


# MCAR function
def apply_mcar(df, column_name, missing_percentage):
    num_rows = len(df)
    num_missing = int(num_rows * missing_percentage)
    num_missing = min(num_missing, num_rows)

    missing_indices = np.random.choice(df.index, num_missing, replace=False)

    df.loc[missing_indices, column_name] = np.nan

    return df


column_name = 'sepal.length'
missing_percentage = 0.1

data_with_missing = apply_mcar(data.copy(), column_name, missing_percentage)

data_with_missing.head(10)

"""Structural Missing Data Generation designed to introduce missing values based on a condition applied to a column. For example, you can make 'petal_length' missing in 40% of cases where 'petal_length' is greater than 1.5.
"""


def apply_mar(df, condition_column, condition, column_name, missing_percentage):
    condition_met_indices = df[condition_column].apply(condition)
    eligible_indices = df[condition_met_indices].index
    num_eligible = len(eligible_indices)
    num_missing = int(num_eligible * missing_percentage)
    num_missing = min(num_missing, num_eligible)
    missing_indices = np.random.choice(eligible_indices, num_missing, replace=False)
    df.loc[missing_indices, column_name] = np.nan

    return df


condition_column = 'petal.length'

"""Now Let's try some Advanced Structural Patterns:
scenarios where missingness in one variable depends on the values of multiple other variables, simulating real-world complexities.
"""


def advanced_missingness(df, missingness_rules):
    for rule in missingness_rules:
        condition_mask = rule['condition'](df)
        affected_rows = df[condition_mask]

        for column in rule['columns']:
            if 'missing_percentage' in rule:
                missing_count = int(len(affected_rows) * rule['missing_percentage'])
                missing_indices = np.random.choice(affected_rows.index, size=missing_count, replace=False)
                df.loc[missing_indices, column] = np.nan
            else:
                df.loc[condition_mask, column] = np.nan

    return df


missingness_rules = [
    {
        'columns': ['sepal_width'],
        'condition': lambda df: df['sepal_length'] > 5.8,
        'missing_percentage': 0.3,
    },
    {
        'columns': ['petal_length'],
        'condition': lambda df: df['class'] == 'Iris-setosa',
        'missing_percentage': 0.5,
    },
    {
        'columns': ['petal_width'],
        'condition': lambda df: df['class'] == 'Iris-versicolor',
        # if no 'missing_percentage' then it means apply to all rows meeting the condition
    }
]

modified_iris_df = advanced_missingness(data.copy(), missingness_rules)
modified_iris_df.head(20)

"""##Seqeuntial missingness
importance of simulating realistic missingness patterns that often occur in longitudinal studies or medical experiments. In such datasets, missingness isn't random but follows certain patterns due to the nature of the study or participant responses. Here are key points from your supervisor's guidance:

Sequential Missingness: In longitudinal studies, if a data point is missing at one time point (e.g., a follow-up visit), subsequent data points (later visits) might also be missing. This pattern reflects scenarios like participants dropping out of a study.
"""


def apply_mnar(df, start_col, prob_missing=0.1, prob_continuation=0.8):
    for index, row in df.iterrows():
        if np.random.rand() < prob_missing:
            for col in df.columns[df.columns.get_loc(start_col):]:
                df.at[index, col] = np.nan
                if np.random.rand() > prob_continuation:
                    break
    return df


df_with_missingness = apply_mnar(data.copy(), 'sepal_length')
df_with_missingness.head(20)


def generate_synthetic_iris(size=150):
    # generate features, standard deviations and means
    sepal_length = np.random.normal(loc=5.84, scale=0.83, size=size)
    sepal_width = np.random.normal(loc=3.05, scale=0.43, size=size)
    petal_length = np.random.normal(loc=3.76, scale=1.76, size=size)
    petal_width = np.random.normal(loc=1.20, scale=0.76, size=size)

    synthetic_iris = pd.DataFrame({
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    })

    synthetic_iris.loc[
        np.random.choice(synthetic_iris.index, size=int(size * 0.1), replace=False), 'sepal_width'] = np.nan

    return synthetic_iris


synthetic_iris = generate_synthetic_iris()

synthetic_iris.head(20)


def generate_synthetic_dataset(original_df):
    synthetic_df = pd.DataFrame()

    # Handling numerical data with simplified KDE
    numerical_columns = original_df.select_dtypes(include=['float64', 'int64']).columns
    for column in original_df.select_dtypes(include=['float64', 'int64']):
        data = original_df[column].dropna()
        kde = gaussian_kde(data, bw_method='scott')
        synthetic_df[column] = kde.resample(original_df.shape[0])[0]

    # Handling categorical data
    for column in original_df.select_dtypes(include=['object']):
        probabilities = original_df[column].value_counts(normalize=True)
        synthetic_df[column] = np.random.choice(probabilities.index, size=original_df.shape[0], p=probabilities.values)

    if not numerical_columns.empty:
        pca = PCA(n_components=len(numerical_columns))
        pca.fit(original_df[numerical_columns].dropna())  # Fit PCA to remove missing values
        transformed_data = pca.transform(original_df[numerical_columns].fillna(original_df[numerical_columns].mean()))
        covariance_matrix = np.cov(transformed_data, rowvar=False)
        multivariate_normal = np.random.multivariate_normal(mean=np.mean(transformed_data, axis=0),
                                                            cov=covariance_matrix,
                                                            size=original_df.shape[0])
        synthetic_numerical_data = pca.inverse_transform(multivariate_normal)
        synthetic_numerical_df = pd.DataFrame(synthetic_numerical_data, columns=numerical_columns)
        synthetic_df[numerical_columns] = synthetic_numerical_df
    return synthetic_df
