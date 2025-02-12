from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
    X_train = scaler.fit_transform(X_train)  # Fit and transform on training set
    X_test = scaler.transform(X_test)  # Transform only the test set

    # Define Logistic Regression model and parameter grid
    logreg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=max_iter_num)
    param_grid = {'C': np.logspace(-1, 1, 20)}  # Range of values for C
    param_grid = {'C': np.linspace(0.05, 0.15, 10)}

    # Grid Search to find the best C based on ROC AUC
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best C value
    best_C = grid_search.best_params_['C']
    print(f"Best C (based on ROC AUC): {best_C}")

    best_logreg = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=max_iter_num)
    best_logreg.fit(X_train, y_train)
    selected_features = np.where(best_logreg.coef_ != 0)[1]
    print(f"Selected features: {[markers[i] for i in selected_features]}")

    # Reduce dataset to selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Retrain Logistic Regression with selected features
    final_model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=max_iter_num)
    final_model.fit(X_train_selected, y_train)

    y_pred = final_model.predict_proba(X_test_selected)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - 0.8))
    target_sensitivity = tpr[idx]

    return [markers[i] for i in selected_features], roc_auc, target_sensitivity

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

def compute_ci(sample, confidence=0.95):
    N = len(sample)
    sample = np.array(sample)
    m = sample.mean()
    s = sample.std()
    dof = N-1
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    return m, (m-s*t_crit/np.sqrt(N), m+s*t_crit/np.sqrt(N)) 