import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from my_feature_selection import lasso_feature_selection, forward_stepwise_selection, backward_stepwise_selection, bootstrap_stepwise
from make_input import make_input
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter

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

df.drop(columns=['CYFRA21-1'], inplace=True)
df['CA125 '] = df['CA125 '].fillna(15.75)

df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

df.reset_index(drop=True, inplace=True)

IDs = list(df["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
times = []
for idx in IDs:
    df_i=df.loc[df["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
    times_i = list(df_i["Age @ sample (yrs, 2 decimal)"].values[-1] - df_i["Age @ sample (yrs, 2 decimal)"].values)
    times += times_i
df["Age @ sample (yrs, 2 decimal)"] = times

df_last = df.drop_duplicates(subset='VOL IDs FOR EXPERIMENTAL PURPOSES', keep='last')
markers = df_last.columns.to_list()[28:]
X = df_last[markers]
y = df_last['CASE CONTROL STATUS'].to_numpy()

boost_features = bootstrap_stepwise(X, y, 50)
boost_features = [item for sub_list in boost_features for item in sub_list]
frequency = Counter(boost_features)
frequency = frequency.most_common()
print(frequency)