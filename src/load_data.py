import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from my_feature_selection import lasso_feature_selection, forward_stepwise_selection, backward_stepwise_selection, my_scaling, get_distance, run_lowess_classification
from rnn_class import VectorizeData, build_rnn, scaling
from make_input import make_input
import json
from torch.utils.data import DataLoader

dir = "../data/LungCA_smoking_diabetes_upd.xlsx"
data = pd.read_excel(dir)

df = data.sort_values(['VOL IDs FOR EXPERIMENTAL PURPOSES','Age @ sample (yrs, 2 decimal)']).groupby('VOL IDs FOR EXPERIMENTAL PURPOSES').head(10)
df = df.reset_index(drop=True)

nrow = df.shape[0]

markers = df.columns.to_list()[28:]

"""column check"""

"""
NA_list = []
for marker in markers:
    marker_epr = df[marker].tolist()
    num_NA = np.sum([pd.isna(elt) for elt in marker_epr])
    NA_list.append(num_NA)
    if num_NA > 200:
        print(marker)

plt.figure() 
sns.histplot(NA_list)
"""

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

"""
plt.figure() 
sns.histplot(NA_list)
"""

df.drop(rows_to_remove, axis=0, inplace=True)

"""
markers = df.columns.to_list()[28:]
for marker in markers:
    count = df[marker].isna().sum()
    if  count > 0:
        print(marker, count)
"""

df.drop(columns=['CYFRA21-1'], inplace=True)
df['CA125 '] = df['CA125 '].fillna(15.75)

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/2)

np.random.seed(44)
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

markers = df_train.columns.to_list()[28:]
IDs = list(df_train["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i=df_train.loc[df_train["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df_train["Age @ sample (yrs, 2 decimal)"] = times

IDs = list(df_test["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i=df_test.loc[df_test["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df_test["Age @ sample (yrs, 2 decimal)"] = times


"""
#df_train = my_scaling(df_train, markers)

df0 = df_train.loc[df_train["CASE CONTROL STATUS"] == 0]
df1 = df_train.loc[df_train["CASE CONTROL STATUS"] == 1]

dis_scores = []
for feature in markers:
    dis_scores.append(get_distance(df0, df1, feature))

idx_list = np.argsort(dis_scores)
dis_scores = np.array(dis_scores)[idx_list]
marker_ordered = [markers[elm] for elm in idx_list]
print(marker_ordered[:5])
"""

"""
a,b = run_lowess_classification(df_train, df_test, 'KLK13')
print(a, b)
"""

df_last_train = df_train.drop_duplicates(subset='VOL IDs FOR EXPERIMENTAL PURPOSES', keep='last')
markers = df_last_train.columns.to_list()[28:]
X_train = df_last_train[markers]
y_train = df_last_train['CASE CONTROL STATUS'].to_numpy()

selected_features = lasso_feature_selection(X_train, y_train, C=0.1)
print("Selected features:", selected_features)

columns = ["VOL IDs FOR EXPERIMENTAL PURPOSES", "Age @ sample (yrs, 2 decimal)", "CASE CONTROL STATUS", ] + selected_features
df_train = df_train[columns]
df_test = df_test[columns]

df_train, df_test = scaling(df_train, df_test, selected_features)

train_data, train_times, train_targets = make_input(df_train, selected_features)   
test_data, test_times, test_targets = make_input(df_test, selected_features) 

ds_train = VectorizeData(train_data, train_times, train_targets)
ds_test = VectorizeData(test_data, test_times, test_targets)

dir_params = "../data/params_to_select.json"
with open(dir_params) as data_file:
    args = json.load(data_file)

params = args['models_grid']['model_0']
train_dl = DataLoader(ds_train, batch_size=params['batch size'])
test_dl = DataLoader(ds_test, batch_size=len(ds_test))

input_size = len(selected_features)
n_out = 2

rnn_model = build_rnn(input_size, n_out, params=params)
m = rnn_model.get_model(network_type='bilstm')
roc_auc = rnn_model.fit(m, train_dl, test_dl)

"""
counts = df_last_entries.groupby('CASE CONTROL STATUS')['smoking_status'].value_counts(dropna=False)
print(counts)
counts = df_last_entries.groupby('CASE CONTROL STATUS')['diabetes'].value_counts(dropna=False)
print(counts)
"""

"""
#df_last_train, df_last_test = train_test_split(df_last_entries, test_size=0.5, random_state=40)
df_last_train = df_train.drop_duplicates(subset='VOL IDs FOR EXPERIMENTAL PURPOSES', keep='last')
X_train = df_last_train[markers]
y_train = df_last_train['CASE CONTROL STATUS'].to_numpy()


selected_features = lasso_feature_selection(X_train, y_train, C=0.05)
print("Selected features:", selected_features)
"""

"""
forward_selected_features, forward_best_aic = forward_stepwise_selection(X_train, y_train)
#print(forward_selected_features)
backward_selected_features, backward_best_aic = backward_stepwise_selection(X_train, y_train)
#print(backward_selected_features)
common_features = set(forward_selected_features).intersection(backward_selected_features)
print(common_features)
"""