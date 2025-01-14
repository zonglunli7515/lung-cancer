import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from my_feature_selection import lasso_feature_selection, forward_stepwise_selection, backward_stepwise_selection
from make_input import make_input
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve

dir = "../data/LungCA_smoking_diabetes.xlsx"
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
df['CASE CONTROL STATUS'] = df['CASE CONTROL STATUS'].map({'Control': 0, 'LungCA': 1})

IDs_new = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
train_size = int(len(IDs_new)/2)
np.random.seed(39)
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

selected_features = lasso_feature_selection(X_train, y_train, C=0.05)
print("Selected features:", selected_features)

X_train = df_last_train[selected_features].values

df_last_test = df_test.drop_duplicates(subset='VOL IDs FOR EXPERIMENTAL PURPOSES', keep='last')
X_test = df_last_test[selected_features].values
y_test = df_last_test['CASE CONTROL STATUS'].to_numpy()

clf = LogisticRegression() 
clf = make_pipeline(StandardScaler(), clf)
clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, preds)
fpr, tpr, thresholds = roc_curve(y_test, preds)
specificity = 1 - fpr
idx = np.argmin(np.abs(specificity - 0.9))
target_sensitivity = tpr[idx]
print(roc_auc, target_sensitivity)