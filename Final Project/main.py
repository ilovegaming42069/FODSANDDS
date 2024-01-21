import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

# Load the dataset
df = pd.read_csv('test.csv')
# Function to calculate total sum of squares for ANOVA
def total_sum_of_squares(df, cont_var):
    overall_mean = df[cont_var].mean()
    return ((df[cont_var] - overall_mean) ** 2).sum()

# Pearson correlation function with t-statistic
def calculate_pearson_correlation(df, var1, var2):
    correlation, p_value = stats.pearsonr(df[var1], df[var2])
    n = len(df)
    t_stat = correlation * ((n - 2) / (1 - correlation**2))**0.5
    return correlation, t_stat, p_value

# T-test for two independent samples
def perform_ttest(df, cat_var, cont_var):
    group1 = df[df[cat_var] == df[cat_var].unique()[0]][cont_var]
    group2 = df[df[cat_var] == df[cat_var].unique()[1]][cont_var]
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return t_stat, p_value

# ANOVA test with total sum of squares
def perform_anova(df, cat_var, cont_var):
    groups = [group[cont_var].values for name, group in df.groupby(cat_var)]
    f_stat, p_value = stats.f_oneway(*groups)
    total_ss = total_sum_of_squares(df, cont_var)
    return total_ss, f_stat, p_value

# Calculating Pearson correlations, T-tests, and ANOVA tests
pearson_results = {
    'Well-being score vs Age': calculate_pearson_correlation(df, 'Well-being score', 'Age'),
    'Anxiety score vs Age': calculate_pearson_correlation(df, 'anxiety score', 'Age'),
    'Well-being score vs Media use': calculate_pearson_correlation(df, 'Well-being score', 'Media use(hrs/day)'),
    'Anxiety score vs Media use': calculate_pearson_correlation(df, 'anxiety score', 'Media use(hrs/day)'),
'Media use vs Age': calculate_pearson_correlation(df, 'Age', 'Media use(hrs/day)')
}

ttest_results = {
    'Well-being score vs Sex': perform_ttest(df, 'Sex', 'Well-being score'),
    'Anxiety score vs Sex': perform_ttest(df, 'Sex', 'anxiety score')
}

anova_results = {
    'Well-being score vs Qualifications': perform_anova(df, 'Qualifications', 'Well-being score'),
    'Anxiety score vs Qualifications': perform_anova(df, 'Qualifications', 'anxiety score'),
    'Well-being score vs Habitat': perform_anova(df, 'Habitat', 'Well-being score'),
    'Anxiety score vs Habitat': perform_anova(df, 'Habitat', 'anxiety score')
}

# Output the results
print("Pearson Correlation Results:")
for test, result in pearson_results.items():
    print(f"{test}: Correlation = {result[0]}, T-statistic = {result[1]}, P-value = {result[2]}")

print("\nT-Test Results:")
for test, result in ttest_results.items():
    print(f"{test}: T-statistic = {result[0]}, P-value = {result[1]}")

print("\nANOVA Test Results:")
for test, result in anova_results.items():
    print(f"{test}: Total Sum of Squares = {result[0]}, F-statistic = {result[1]}, P-value = {result[2]}")

correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', annot_kws={'size':12})  # Increase annotation text size
plt.title('Correlation Heatmap of Dataset Features')  # Increase title font size
plt.xticks(fontsize=6)  # Increase x-axis tick font size
plt.yticks(fontsize=6)  # Increase y-axis tick font size
plt.show()




# Modify 'Sex', 'Qualifications', and 'Habitat' features
# Modify 'Sex', 'Qualifications', and 'Habitat' features
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
qualification_mapping = {'UG': 0, 'PG': 1, 'Phd/M.phil': 2}
df['Qualifications'] = df['Qualifications'].map(qualification_mapping)
habitat_mapping = {'Rural               ': 0, 'Urban Municipal Area': 1, 'Metropolitan City   ': 2}
df['Habitat'] = df['Habitat'].map(habitat_mapping)



def plot_enhanced_boxplot(column, title):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, y=column)

    median = df[column].median()
    lower_quartile = df[column].quantile(0.25)
    upper_quartile = df[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    lower_whisker = df[df[column] >= (lower_quartile - 1.5 * iqr)][column].min()
    upper_whisker = df[df[column] <= (upper_quartile + 1.5 * iqr)][column].max()
    outliers = df[(df[column] < lower_whisker) | (df[column] > upper_whisker)][column]

    plt.title(title)
    plt.text(x=0.5, y=median, s=f"Median: {median}", ha='center', va='center', color='black',
             bbox=dict(facecolor='yellow', alpha=0.5))
    plt.text(x=0.5, y=lower_quartile, s=f"LQ: {lower_quartile}", ha='center', va='center', color='black',
             bbox=dict(facecolor='blue', alpha=0.5))
    plt.text(x=0.5, y=upper_quartile, s=f"UQ: {upper_quartile}", ha='center', va='center', color='black',
             bbox=dict(facecolor='green', alpha=0.5))
    plt.text(x=0.5, y=lower_whisker, s=f"Min: {lower_whisker}", ha='center', va='center', color='black',
             bbox=dict(facecolor='grey', alpha=0.5))
    plt.text(x=0.5, y=upper_whisker, s=f"Max: {upper_whisker}", ha='center', va='center', color='black',
             bbox=dict(facecolor='grey', alpha=0.5))
    plt.show()


plot_enhanced_boxplot('Well-being score', 'Distribution of Well-being Scores')
plot_enhanced_boxplot('anxiety score', 'Distribution of Anxiety Scores')

# Create a new target variable for depression detection
df['Depression'] = ((df['Well-being score'] < df['Well-being score'].median()) &
                    (df['anxiety score'] > df['anxiety score'].median())).astype(int)
df.drop(['Well-being score', 'Well-being category', 'anxiety score', 'anxiety category', 'Media category',
         'Age Category'], axis=1, inplace=True)

# Balance and enlarge the dataset
df_depressed = df[df['Depression'] == 1]
df_not_depressed = df[df['Depression'] == 0]
depression_counts_before_balancing = df['Depression'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(depression_counts_before_balancing, labels=['Non-Depressed', 'Depressed'], autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Depression Before Balancing the Dataset')
plt.show()
df_not_depressed_downsampled = resample(df_not_depressed, replace=False, n_samples=len(df_depressed), random_state=42)
df_balanced = pd.concat([df_depressed, df_not_depressed_downsampled])
df_enlarged = pd.concat([df_balanced] * 2).reset_index(drop=True)
df_enlarged = df_enlarged.sample(frac=1, random_state=42).reset_index(drop=True)

# Splitting the enlarged dataset into features and target variable
X_enlarged = df_enlarged.drop('Depression', axis=1)
y_enlarged = df_enlarged['Depression']
X_train, X_test, y_train, y_test = train_test_split(X_enlarged, y_enlarged, test_size=0.25, random_state=42)

# Update numerical features
numerical_features = ['Age', 'Media use(hrs/day)', 'Sex', 'Qualifications', 'Habitat']

# Create column transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features)
])

# Fit the preprocessor to your training data
preprocessor.fit(X_train)

# Define models, hyperparameters, and feature importance methods
# Define models with expanded hyperparameters
models = {
    'Logistic Regression': (
        LogisticRegression(),
        {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['lbfgs', 'saga', 'newton-cg', 'liblinear'],
            'classifier__penalty': ['l2']
        },
        'coefficients'
    ),
    'Random Forest': (
        RandomForestClassifier(),
        {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [10, 20, 30, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'feature_importances'
    ),
    'SVM': (
        SVC(),
        {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': [0.01, 0.1, 1, 'scale', 'auto'],
            'classifier__kernel': ['linear', 'rbf', 'poly']
        },
        'permutation'
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(),
        {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__learning_rate': [0.01, 0.1, 0.2, 0.5],
            'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
            'classifier__max_depth': [3, 5, 7, 9]
        },
        'feature_importances'
    ),
    'K-Nearest Neighbors': (
        KNeighborsClassifier(),
        {
            'classifier__n_neighbors': [3, 5, 7, 10, 15],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'classifier__leaf_size': [10, 30, 50, 70]
        },
        'permutation'
    ),
    'Naive Bayes': (
        GaussianNB(),
        {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'coefficients'
    )
}

# Train, tune, and evaluate each model; calculate feature importances
for model_name, (model, params, importance_method) in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    grid_search = GridSearchCV(pipeline, params, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Model: {model_name}")
    print(report)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    # Feature importance evaluation
    if importance_method == 'coefficients':
        if hasattr(grid_search.best_estimator_.named_steps['classifier'], 'coef_'):
            feature_importances = grid_search.best_estimator_.named_steps['classifier'].coef_[0]
            sns.barplot(x=feature_importances, y=preprocessor.get_feature_names_out())
            plt.title(f'Feature Importances in {model_name}', fontsize=10 )
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel('Importance', fontsize=10)
            plt.ylabel('Features', fontsize=10)
            plt.tight_layout()
            plt.show()
    elif importance_method == 'feature_importances':
        feature_importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
        sns.barplot(x=feature_importances, y=preprocessor.get_feature_names_out())
        plt.title(f'Feature Importances in {model_name}', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Importance', fontsize=10)
        plt.ylabel('Features', fontsize=10)
        plt.tight_layout()
        plt.show()
    elif importance_method == 'permutation':
        result = permutation_importance(grid_search.best_estimator_, X_test, y_test, n_repeats=10, random_state=42)
        perm_sorted_idx = result.importances_mean.argsort()
        sns.boxplot(data=result.importances[perm_sorted_idx].T, orient="h", palette="vlag")
        plt.title(f'Permutation Importances in {model_name}', fontsize=10)
        plt.yticks(perm_sorted_idx, preprocessor.get_feature_names_out()[perm_sorted_idx], fontsize=10)
        plt.xlabel('Importance Score', fontsize=10)
        plt.tight_layout()
        plt.show()





