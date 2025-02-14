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

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/10*7)

seed_num = 42
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

selected_features = ['CEACAM5', 'MUC-16', 'CXL17', 'WFDC2', 'hK14', 'CPE']

columns = ["VOL IDs FOR EXPERIMENTAL PURPOSES", "Age @ sample (yrs, 2 decimal)", "CASE CONTROL STATUS", ] + selected_features
df_train = df_train[columns]
df_test = df_test[columns]

#df_train, df_test = scaling(df_train, df_test, selected_features)

train_data, train_times, train_targets = make_input(df_train, selected_features)   
test_data, test_times, test_targets = make_input(df_test, selected_features)

ds_train = VectorizeData(train_data, train_times, train_targets)
ds_test = VectorizeData(test_data, test_times, test_targets)

# Define objective function for Optuna
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample hyperparameters
    lstm_units = trial.suggest_categorical("lstm_units", [8, 12, 16, 20, 24, 28, 32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 20])
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    num_epochs = trial.suggest_int("epochs", 1, 20)

    train_dl = DataLoader(ds_train, batch_size=batch_size)
    model = LSTM(input_size=len(selected_features), hidden_size=lstm_units, num_layers=1, output_size=2, dropout=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y, lengths in train_dl:
            batch_X, batch_y, lengths = sort_batch(batch_X, batch_y, lengths)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X, lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate on training data (for demonstration, ideally use a validation set)
    model.eval()
    with torch.no_grad():
        train_dl_all = DataLoader(ds_train, len(train_data))
        for batch_X, batch_y, lengths in train_dl_all:
            batch_X, batch_y, lengths = sort_batch(batch_X, batch_y, lengths)
            outputs = model(batch_X.to(device), lengths)  # Get predicted probabilities
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.cpu().numpy().squeeze()
            roc_auc = roc_auc_score(batch_y, outputs[:,1])  # Compute AUC score

    return roc_auc  # Optuna will maximize this metric

# Run Bayesian Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Print best hyperparameters
print("Best Hyperparameters:", study.best_params)
print("Best AUC:", study.best_value)

best_params = study.best_params
# Re-initialize the model using best parameters
best_model = LSTM(
    input_size=len(selected_features), 
    hidden_size=best_params["lstm_units"], 
    num_layers=1, 
    output_size=2, 
    dropout=best_params["dropout_rate"]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])
train_dl = DataLoader(ds_train, best_params["batch_size"])

best_model.train()
for epoch in range(best_params["epochs"]):
    for batch_X, batch_y, lengths in train_dl:
        batch_X, batch_y, lengths = sort_batch(batch_X, batch_y, lengths)
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = best_model(batch_X, lengths)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

best_model.eval()
target_specificity = 0.9
with torch.no_grad():
    test_dl = DataLoader(ds_test, len(test_data))
    for batch_X, batch_y, lengths in test_dl:
        batch_X, batch_y, lengths = sort_batch(batch_X, batch_y, lengths)
        outputs = best_model(batch_X.to(device), lengths)  # Get predicted probabilities
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.cpu().numpy().squeeze()
        roc_auc = roc_auc_score(batch_y, outputs[:,1])  # Compute AUC score
        fpr, tpr, thresholds = roc_curve(batch_y, outputs[:,1])
        print(roc_auc)