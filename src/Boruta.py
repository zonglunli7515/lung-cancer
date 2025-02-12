from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from boruta import BorutaPy
import pandas as pd
import numpy as np
from statsmodels.imputation.mice import MICEData
from fancyimpute import IterativeImputer
from collections import Counter
from scipy.stats import t
from sklearn.metrics import roc_auc_score, roc_curve

dir = "../data/LungCA_smoking_diabetes_upd.xlsx"
data = pd.read_excel(dir)

df = data.sort_values(['VOL IDs FOR EXPERIMENTAL PURPOSES','Age @ sample (yrs, 2 decimal)']).groupby('VOL IDs FOR EXPERIMENTAL PURPOSES').head(10)
df = df.reset_index(drop=True)

nrow = df.shape[0]

markers = df.columns.to_list()[28:]

df.drop(columns=['CA19-9', 'CEA'], inplace=True)

"""row check"""
NA_list = []
rows_to_remove = []
for i in range(nrow):
    marker_epr = df.iloc[i].tolist()[28:]
    num_NA = np.sum([pd.isna(elt) for elt in marker_epr])
    if num_NA > 50:
        rows_to_remove.append(i)
        #print(i)
    NA_list.append(num_NA)

df.drop(rows_to_remove, axis=0, inplace=True)

IDs = list(df["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i=df.loc[df["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df["Age @ sample (yrs, 2 decimal)"] = times

markers = df.columns.to_list()[28:]
data_to_impute = df[markers]

imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(data_to_impute)
df[markers] = imputed_data


"""
df.drop(columns=['CYFRA21-1'], inplace=True)
df['CA125 '] = df['CA125 '].fillna(15)
"""

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/10*7)

def run_Boruta_analysis(seed_num):
    np.random.seed(seed_num)
    train_idx = list(np.random.choice(IDs_new, train_size, replace = False))
    test_idx = list(set(IDs_new) - set(train_idx)) 

    df_train = df[df['VOL IDs FOR EXPERIMENTAL PURPOSES'] == train_idx[0]]
    for idx in train_idx:
        if idx != train_idx[0]:
            this_df = df[df['VOL IDs FOR EXPERIMENTAL PURPOSES'] == idx]
            df_train = pd.concat([df_train, this_df])
    df_train.reset_index(drop=True, inplace=True)
    df_test = df[df['VOL IDs FOR EXPERIMENTAL PURPOSES'] == test_idx[0]]
    for idx in test_idx:
        if idx != test_idx[0]:
            this_df = df[df['VOL IDs FOR EXPERIMENTAL PURPOSES'] == idx]
            df_test = pd.concat([df_test, this_df])
    df_test.reset_index(drop=True, inplace=True)

    df_last_train = df_train.drop_duplicates(subset='VOL IDs FOR EXPERIMENTAL PURPOSES', keep='last')
    markers = df_last_train.columns.to_list()[28:]
    X_train = df_last_train[markers]
    y_train = df_last_train['CASE CONTROL STATUS'].to_numpy()

    df_last_test = df_test.drop_duplicates(subset='VOL IDs FOR EXPERIMENTAL PURPOSES', keep='last')
    X_test = df_last_test[markers]
    y_test = df_last_test['CASE CONTROL STATUS'].to_numpy()

    # Initialize Boruta
    # Step 2: Apply Boruta with the best max_depth
    best_max_depth = 50 # None
    rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=best_max_depth, random_state=seed_num)
    boruta_selector = BorutaPy(estimator=rf_boruta, n_estimators='auto', random_state=seed_num)

    # Fit Boruta
    boruta_selector.fit(X_train.values, y_train)

    # Get selected features
    selected_features = X_train.columns[boruta_selector.support_]

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    final_model = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=best_max_depth, random_state=seed_num)
    final_model.fit(X_train_selected, y_train)
    y_pred = final_model.predict(X_test_selected)
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - 0.8))
    target_sensitivity = tpr[idx]
    return selected_features.tolist(), roc_auc, target_sensitivity

a,b,c = run_Boruta_analysis(0)

feature_list = []
auc_list = []
sensitivity_list = []
for i in range(20):
    print(i)
    features, auc, sensitivity = run_Boruta_analysis(i)
    feature_list += features
    auc_list.append(auc)
    sensitivity_list.append(sensitivity)

frequency = Counter(feature_list)
frequency = frequency.most_common()

def compute_ci(sample, confidence=0.95):
    N = len(sample)
    sample = np.array(sample)
    m = sample.mean()
    s = sample.std()
    dof = N-1
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    return m, (m-s*t_crit/np.sqrt(N), m+s*t_crit/np.sqrt(N)) 

"""
# Define the hyperparameters to tune
param_grid = {
    'max_depth': [5, 10, 20, 30, 40, 50]    # Maximum depth of each tree
}
# Initialize a random forest model
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=42)
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_max_depth = grid_search.best_params_['max_depth']
print("Best max_depth:", best_max_depth)
"""

