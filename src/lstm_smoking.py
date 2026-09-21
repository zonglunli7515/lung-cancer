import pandas as pd
import numpy as np
from rnn_class import VectorizeData, build_rnn, scaling, LSTM, sort_batch
from make_input import make_input
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from fancyimpute import IterativeImputer
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import t

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

imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(data_to_impute)
df[markers] = imputed_data

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})
df['smoking_status'] = df['smoking_status'].fillna('missing')
one_hot = pd.get_dummies(df['smoking_status'])
df = pd.concat([df, one_hot], axis=1)

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/10*7)

selected_features = ['CEACAM5', 'MUC-16', 'CXL17', 'WFDC2', 'hK14', 'CPE']
selected_features = ['CEACAM5', 'MUC-16', 'ever smoker', 'missing', 'never smoker']

np.random.seed(42)
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