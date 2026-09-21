"""
Variant of LASSO.py: best C is still chosen by cross-validation on the training
set, but the L1 logistic regression that produces the non-zero coefficients is
refitted on the *test* split of each iteration. Over 100 iterations the
frequency of each selected marker is accumulated.

Note: features selected on the test split cannot also be evaluated on that same
split without leakage, so the AUC / sensitivity reported here are still the
honest train-fit -> test-predict numbers from the original script.
"""

from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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

# Create and fit the imputer
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data_array = imputer.fit_transform(data_to_impute)

# Convert back to DataFrame to preserve column names
imputed_data = pd.DataFrame(imputed_data_array, columns=markers, index=df.index)
df[markers] = imputed_data

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/10*7)


def run_LASSO_analysis(seed_num):
    max_iter_num = 100
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

    # Standardize Training and Test Data (One-Time Standardization)
    scaler = StandardScaler()
    X_train_raw = X_train.to_numpy()
    X_test_raw = X_test.to_numpy()
    X_train = scaler.fit_transform(X_train_raw)   # Fit and transform on training set
    X_test = scaler.transform(X_test_raw)         # Transform only the test set

    # Define Logistic Regression model and parameter grid
    logreg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=max_iter_num)
    param_grid = {'C': np.logspace(-1, 1, 20)}  # Range of values for C

    # Grid Search to find the best C based on ROC AUC -- training set only
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best C value
    best_C = grid_search.best_params_['C']
    print(f"Best C (based on ROC AUC): {best_C}")

    # --- Selection on the TEST set, applying the same best C ---
    # Standardize the test set on its own so the penalty acts on comparably
    # scaled columns within this split.
    test_scaler = StandardScaler()
    X_test_own = test_scaler.fit_transform(X_test_raw)

    test_logreg = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=max_iter_num)
    test_logreg.fit(X_test_own, y_test)
    selected_features = np.where(test_logreg.coef_ != 0)[1]
    test_features = [markers[i] for i in selected_features]
    print(f"Selected features (test fit): {test_features}")

    # Performance: fit on train, predict on test, using the train-selected features
    train_logreg = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=max_iter_num)
    train_logreg.fit(X_train, y_train)
    train_selected = np.where(train_logreg.coef_ != 0)[1]

    X_train_selected = X_train[:, train_selected]
    X_test_selected = X_test[:, train_selected]

    final_model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=max_iter_num)
    final_model.fit(X_train_selected, y_train)

    y_pred = final_model.predict_proba(X_test_selected)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - 0.8))
    target_sensitivity = tpr[idx]

    return test_features, roc_auc, target_sensitivity


feature_list = []
auc_list = []
sensitivity_list = []
for i in range(100):
    print(i)
    features, auc, sensitivity = run_LASSO_analysis(i)
    feature_list += features
    auc_list.append(auc)
    sensitivity_list.append(sensitivity)

frequency = Counter(feature_list)
frequency = frequency.most_common()
print(frequency)



