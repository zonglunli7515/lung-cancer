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
df = df.reset_index(drop=True)

IDs = list(df["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i=df.loc[df["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df["Age @ sample (yrs, 2 decimal)"] = times

nrow = df.shape[0]
rows_to_remove = []
for i in range(nrow):
    if pd.isna(df.iloc[i]["CYFRA21-1"]):
        rows_to_remove.append(i)
df.drop(rows_to_remove, axis=0, inplace=True)

df1 = df[df['CASE CONTROL STATUS'] == 1]
df0 = df[df['CASE CONTROL STATUS'] == 0]
IDs1 = list(df1["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
IDs0 = list(df0["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())

"""
marker = "CEACAM5"

fig, ax = plt.subplots()  # Correct function to create subplots
for idx in IDs1[10:15]:
    df_i = df1[df1["VOL IDs FOR EXPERIMENTAL PURPOSES"] == idx]
    x = df_i["Age @ sample (yrs, 2 decimal)"].values
    y = df_i[marker].values
    ax.plot(x, y, '.--', linewidth=1.0, markersize=5)  # Removed linestyle since marker is used

# Add labels, title, and legend for better visualization
ax.set_xlabel("Reverse time")
ax.set_ylabel(marker)
#ax.set_title("CYFRA21-1 Levels vs. Age")
#ax.legend()
plt.show()
"""

# Create subplots: 1 rows x 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
# Flatten axes array for easy iteration
axes = axes.flatten()

sub_markers = ["CEACAM5", "MUC-16"]
for j, marker in enumerate(sub_markers):
    for idx in IDs1[10:15]:
        df_i = df1[df1["VOL IDs FOR EXPERIMENTAL PURPOSES"] == idx]
        x = df_i["Age @ sample (yrs, 2 decimal)"].values
        y = df_i[marker].values
        axes[j].plot(x, y, '.--', linewidth=2.0, markersize=10)
        axes[j].set_title(marker)
        axes[j].set_xlabel('Reversed time (year)')
        if j == 0:
            axes[j].set_ylabel('Value')

# Adjust layout
plt.tight_layout()
plt.show()

"""
fig, ax = plt.subplots()  # Correct function to create subplots
for idx in IDs0:
    df_i = df0[df0["VOL IDs FOR EXPERIMENTAL PURPOSES"] == idx]
    x = df_i["Age @ sample (yrs, 2 decimal)"].values
    y = df_i[marker].values
    ax.plot(x, y, '.--', linewidth=0.5, markersize=2)  # Removed linestyle since marker is used

# Add labels, title, and legend for better visualization
ax.set_xlabel("Reverse time")
ax.set_ylabel(marker)
#ax.set_title("CYFRA21-1 Levels vs. Age")
#ax.legend()
plt.show()
"""