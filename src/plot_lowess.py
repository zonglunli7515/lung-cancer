import pandas as pd
import numpy as np 
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

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

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

IDs = list(df["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i=df.loc[df["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df["Age @ sample (yrs, 2 decimal)"] = times


# df.drop(columns=['CYFRA21-1'], inplace=True)

""""
###### for CYFRA21-1 ######
df_sub = df[["VOL IDs FOR EXPERIMENTAL PURPOSES", "Age @ sample (yrs, 2 decimal)", "CASE CONTROL STATUS", "CYFRA21-1"]]
nrow = df_sub.shape[0]
rows_to_remove = []
for i in range(nrow):
    if pd.isna(df_sub.iloc[i]["CYFRA21-1"]):
        rows_to_remove.append(i)
"""

markers = df.columns.to_list()[28:]

df0 = df.loc[df['CASE CONTROL STATUS'] == 0]
df1 = df.loc[df['CASE CONTROL STATUS'] == 1]
df0 = df0.reset_index(drop=True)
df1 = df1.reset_index(drop=True)

def get_times_values(data, feature_name):
    return data["Age @ sample (yrs, 2 decimal)"].values, data[feature_name].values

def get_unique(x_array, y_array):
    indices = []
    x_unique = np.unique(x_array)
    for elt in x_unique:
        indices.append(np.where(x_array==elt)[0][0])
    return x_unique, y_array[indices]

def plot_idv(data, feature):
    if feature == "CYFRA21-1":
        nrow = data.shape[0]
        to_remove = []
        for i in range(nrow):
            if pd.isna(data.iloc[i]["CYFRA21-1"]):
                to_remove.append(i)
        data.drop(to_remove, axis=0, inplace=True)
       
    X, y = get_times_values(data, feature)
    sorted_idx = np.argsort(X)
    X = X[sorted_idx]
    y = y[sorted_idx]
    lowess_score = lowess(y, X, frac=0.6)
    lx = lowess_score[:,0]
    ly = lowess_score[:,1]
    lx, ly = get_unique(lx, ly)
    # Bootstrapping to estimate confidence intervals
    n_boot = 100
    y_boot = np.zeros((n_boot, len(lx)))
    for i in range(n_boot):
        sample_idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
        x_sample = X[sample_idx]
        y_sample = y[sample_idx]
        lowess_boot = lowess(y_sample, x_sample, frac=0.6)
        y_boot[i] = np.interp(lx, lowess_boot[:, 0], lowess_boot[:, 1])
    # Calculate the confidence intervals
    lower_ci = np.percentile(y_boot, 2.5, axis=0)
    upper_ci = np.percentile(y_boot, 97.5, axis=0)
    return lx, ly, lower_ci, upper_ci

fig, axs = plt.subplots(3, 2, figsize=(18, 18), dpi=300)
axs = axs.flatten()

sub_markers = markers[20:40]
sub_markers = ['CEACAM5', 'MUC-16', 'CXL17', 'WFDC2', 'hK14', 'CPE']

for j, feature in enumerate(sub_markers):
    print(feature)
    res1 = plot_idv(df1, feature)
    res0 = plot_idv(df0, feature)
    axs[j].plot(res1[0], res1[1], color='red', label='case')
    axs[j].fill_between(res1[0], res1[2], res1[3], color='red', alpha=0.2)
    axs[j].plot(res0[0], res0[1], color='blue', label='control')
    axs[j].fill_between(res0[0], res0[2], res0[3], color='blue', alpha=0.2)
    #axs[j].legend()
    axs[j].set_title(feature)
    if j>=16:
        axs[j].set_xlabel('Reverse time')

"""
for ax in axs[13:]:
    ax.axis('off')
"""