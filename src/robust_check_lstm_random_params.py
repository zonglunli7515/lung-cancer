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
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import t
import matplotlib.pyplot as plt
import os

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

selected_features = ['CEACAM5', 'MUC-16', 'CXL17', 'WFDC2', 'hK14', 'CPE']
selected_features = ['CEACAM5', 'MUC-16']


def run_lstm_random(seed_num):
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

    #df_train, df_test = scaling(df_train, df_test, selected_features)

    train_data, train_times, train_targets = make_input(df_train, selected_features)   
    test_data, test_times, test_targets = make_input(df_test, selected_features)

    ds_train = VectorizeData(train_data, train_times, train_targets)
    ds_test = VectorizeData(test_data, test_times, test_targets)

    # Randomly sample one hyperparameter combination from the same search space
    # that Optuna explores in robust_check_lstm.py, instead of tuning.
    best_params = {
        "lstm_units": int(np.random.choice([8, 12, 16, 20, 24, 28, 32])),
        "dropout_rate": float(np.random.uniform(0.0, 0.4)),
        "batch_size": int(np.random.choice([8, 12, 16, 20])),
        "learning_rate": float(10 ** np.random.uniform(np.log10(1e-3), np.log10(1e-2))),
        "epochs": int(np.random.randint(1, 21)),
    }

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
    target_specificity = 0.8
    target_specificity2 = 0.9
    with torch.no_grad():
        test_dl = DataLoader(ds_test, len(test_data))
        for batch_X, batch_y, lengths in test_dl:
            batch_X, batch_y, lengths = sort_batch(batch_X, batch_y, lengths)
            outputs = best_model(batch_X.to(device), lengths)  # Get predicted probabilities
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.cpu().numpy().squeeze()
            roc_auc = roc_auc_score(batch_y, outputs[:,1])  # Compute AUC score
            fpr, tpr, thresholds = roc_curve(batch_y, outputs[:,1])
            specificity = 1 - fpr
            idx = np.argmin(np.abs(specificity - target_specificity))
            idx2 = np.argmin(np.abs(specificity - target_specificity2))
            target_sensitivity = tpr[idx]
            target_sensitivity2 = tpr[idx2]

    return roc_auc, target_sensitivity, target_sensitivity2

def run_simulation(n_iters):
    res_list = Parallel(n_jobs=-1)(delayed(run_lstm_random)(i) for i in range(n_iters))
    return res_list

a = run_simulation(500)
roc_aucs = [b[0] for b in a]
target_sensitivity = [b[1] for b in a]
target_sensitivity2 = [b[2] for b in a]

"""Violin plot of each score across the random-parameter iterations."""
# Mean score of the Optuna-tuned model, drawn on each panel as a reference marker.
# Placeholder -- replace with your own numbers.
optuna_means = {
    "ROC AUC": 0.844,
    "Sensitivity(0.8)": 0.688,
    "Sensitivity(0.9)": 0.557,
}

scores = {
    "ROC AUC": roc_aucs,
    "Sensitivity(0.8)": target_sensitivity,
    "Sensitivity(0.9)": target_sensitivity2,
}

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
for ax, (label, vals) in zip(axes, scores.items()):
    vals = np.asarray(vals)

    parts = ax.violinplot(vals, showextrema=False, widths=0.7)
    for body in parts["bodies"]:
        body.set_facecolor("#7fa8d1")
        body.set_edgecolor("#33475b")
        body.set_alpha(0.65)

    ax.hlines(vals.mean(), 0.65, 1.35, color="#33475b", linewidth=2, zorder=4)
    ax.axhline(optuna_means[label], color="#c0392b", linestyle="--", linewidth=1.6, zorder=2)

    ax.set_title(label)
    ax.set_ylabel("score")
    ax.set_xticks([])

fig.tight_layout()
os.makedirs("../results", exist_ok=True)
fig.savefig("../results/random_params_scores_violin.png", dpi=300, bbox_inches="tight")
print("Figure written to ../results/random_params_scores_violin.png")
