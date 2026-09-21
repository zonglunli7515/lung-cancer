import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Create and fit the imputer
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data_array = imputer.fit_transform(data_to_impute)

# Convert back to DataFrame to preserve column names
imputed_data = pd.DataFrame(imputed_data_array, columns=markers, index=df.index)

# Replace original columns with imputed data
df[markers] = imputed_data

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/10*7)

selected_features = ['smoking_status', 'CEACAM5', 'MUC-16', 'CXL17', 'WFDC2', 'hK14', 'CPE']
selected_features = ['CEACAM5', 'MUC-16']
#selected_features = ['CEACAM5', 'MUC-16', 'CXL17', 'WFDC2', 'hK14']


def run_logit_bayes(seed_num):
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

    columns = ["VOL IDs FOR EXPERIMENTAL PURPOSES", "Age @ sample (yrs, 2 decimal)", "CASE CONTROL STATUS", ] + selected_features
    df_train = df_train[columns]
    df_test = df_test[columns]

    df_last_train = df_train.drop_duplicates(subset='VOL IDs FOR EXPERIMENTAL PURPOSES', keep='last')
    df_last_test = df_test.drop_duplicates(subset='VOL IDs FOR EXPERIMENTAL PURPOSES', keep='last')
    X_train = df_last_train[selected_features]
    X_test = df_last_test[selected_features]
    y_train = df_last_train['CASE CONTROL STATUS'].to_numpy()
    y_test = df_last_test['CASE CONTROL STATUS'].to_numpy()

    def objective(trial):
        C = trial.suggest_loguniform('C', 1e-3, 1e1)
        #solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga', 'newton-cg'])
        solver = trial.suggest_categorical('solver', ['liblinear'])

        model = LogisticRegression(C=C, solver=solver, max_iter=500)

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_train)[:,1]
        
        return roc_auc_score(y_train, y_pred)
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    # Get the best parameters
    best_C = study.best_params['C']
    best_solver = study.best_params['solver']

    #final_model = LogisticRegression(C=best_C, solver=best_solver, max_iter=500)
    final_model = LogisticRegression()
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - 0.8))
    idx2 = np.argmin(np.abs(specificity - 0.9))
    target_sensitivity = tpr[idx]
    target_sensitivity2 = tpr[idx2]
    
    return roc_auc, target_sensitivity, target_sensitivity2

def run_simulation(n_iters):
    res_list = Parallel(n_jobs=-1)(delayed(run_logit_bayes)(i) for i in range(n_iters))
    return res_list

a = run_simulation(100)
roc_aucs = [b[0] for b in a]
target_sensitivity = [b[1] for b in a]
target_sensitivity2 = [b[2] for b in a]

def compute_ci(sample, confidence=0.95):
    N = len(sample)
    sample = np.array(sample)
    m = sample.mean()
    s = sample.std(ddof=1)  # Use sample standard deviation
    dof = N - 1
    t_crit = t.ppf(1 - (1-confidence)/2, dof)  # Compute positive t-critical value
    margin_of_error = s * t_crit / np.sqrt(N)
    
    #return m, (m - margin_of_error, m + margin_of_error)
    return m, s

def bootstrap_ci(sample, confidence=0.95, n_bootstrap=1000):
    sample = np.array(sample)
    medians = [np.median(np.random.choice(sample, size=len(sample), replace=True)) for _ in range(n_bootstrap)]
    lower_bound = np.percentile(medians, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(medians, (1 + confidence) / 2 * 100)
    return np.median(sample), (lower_bound, upper_bound)

print(compute_ci(roc_aucs))
print(compute_ci(target_sensitivity))
print(compute_ci(target_sensitivity2))