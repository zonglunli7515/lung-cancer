import pandas as pd
import numpy as np 
import seaborn as sns
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

markers = df.columns.to_list()[28:]

df0 = df.loc[df['CASE CONTROL STATUS'] == 0]
df1 = df.loc[df['CASE CONTROL STATUS'] == 1]
df0 = df0.reset_index(drop=True)
df1 = df1.reset_index(drop=True)

sub_markers = ['CEACAM5', 'MUC-16', 'CXL17', 'WFDC2', 'hK14', 'CPE']

IDs = list(df1["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i=df1.loc[df1["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
    times_i = list(df_i["Age @ Dx (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df1["Age @ sample (yrs, 2 decimal)"] = times

"""
# Set seaborn style
sns.set(style="whitegrid")

# Create subplots: 2 rows x 3 columns
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
# Flatten axes array for easy iteration
axes = axes.flatten()

for j, marker in enumerate(sub_markers):
    values0 = df0[marker].values

    a = 0
    b = 1
    values_list = []
    for i in range(5):
        df1_this = df1.loc[(df1["Age @ sample (yrs, 2 decimal)"] > a) & (df1["Age @ sample (yrs, 2 decimal)"] <= b)]
        values1_this = df1_this[marker].values
        values_list.append(values1_this)
        a += 1
        b += 1

    values_list.append(values0)

    df_data = pd.DataFrame({
        'Value': [item for sublist in values_list for item in sublist],
        'Stage': [f'Y{i+1}' for i, sublist in enumerate(values_list) for _ in sublist]
    })

    df_data['Stage'] = df_data['Stage'].replace({'Y6': 'Control'})
    sns.boxplot(x='Stage', y='Value', data=df_data, palette='Set3', ax=axes[j])
    axes[j].set_title(marker)
    if j < 3:
        axes[j].set_xlabel('')
    if j!=0 and j!=3:
        axes[j].set_ylabel('')

# Adjust layout
plt.tight_layout()
plt.show()
"""


IDs = list(df0["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i=df0.loc[df0["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df0["Age @ sample (yrs, 2 decimal)"] = times

# Set seaborn style
sns.set(style="whitegrid")

# Create subplots: 2 rows x 3 columns
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
# Flatten axes array for easy iteration
axes = axes.flatten()

for j, marker in enumerate(sub_markers):

    a = 0
    b = 1
    values_list = []
    for i in range(5):
        df0_this = df0.loc[(df0["Age @ sample (yrs, 2 decimal)"] >= a) & (df0["Age @ sample (yrs, 2 decimal)"] < b)]
        values1_this = df0_this[marker].values
        values_list.append(values1_this)
        a += 1
        b += 1

    df_data = pd.DataFrame({
        'Value': [item for sublist in values_list for item in sublist],
        'Stage': [f'Y{i+1}' for i, sublist in enumerate(values_list) for _ in sublist]
    })

    sns.boxplot(x='Stage', y='Value', data=df_data, palette='Set3', ax=axes[j])
    axes[j].set_title(marker)
    if j < 3:
        axes[j].set_xlabel('')
    if j!=0 and j!=3:
        axes[j].set_ylabel('')

# Adjust layout
plt.tight_layout()
plt.show()