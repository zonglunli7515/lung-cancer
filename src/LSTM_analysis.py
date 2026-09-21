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

df.drop(columns=['CA19-9', 'CEA'], inplace=True)

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
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values - df_i["Age @ sample (yrs, 2 decimal)"].values[0])
    times += times_i
df["Age @ sample (yrs, 2 decimal)"] = times

df.drop(columns=['CYFRA21-1'], inplace=True)
df['CA125 '] = df['CA125 '].fillna(15.75)

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/2)



def run_lstm_analysis(seed_num):
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

    selected_features = lasso_feature_selection(X_train, y_train, C=0.1)
    #print("Selected features:", selected_features)

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
    return selected_features, roc_auc

for i in range(10):
    print(run_lstm_analysis(i))