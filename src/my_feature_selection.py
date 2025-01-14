import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, roc_curve

def lasso_feature_selection(X, y, C=1.0):
    """
    Perform Lasso feature selection using logistic regression.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target variable.
        C (float): Inverse of regularization strength (smaller values = stronger regularization).

    Returns:
        list: Selected features.
    """
    # Standardize features to ensure regularization works correctly
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic regression with L1 regularization
    model = LogisticRegression(penalty='l1', solver='liblinear', C=C)
    model.fit(X_scaled, y)

    # Get coefficients and select features with non-zero weights
    selected_features = np.array(X.columns)[model.coef_[0] != 0]
    return selected_features.tolist()

# AIC calculation function for logistic regression
def calculate_aic(model, X, y):
    # Log-likelihood calculation
    log_likelihood = -log_loss(y, model.predict_proba(X), normalize=False)
    k = X.shape[1] + 1  # Number of features + 1 for the intercept
    return 2 * k - 2 * log_likelihood

# Forward stepwise selection
def forward_stepwise_selection(X, y):
    remaining_features = list(X.columns)
    selected_features = []
    best_aic = np.inf
    while remaining_features:
        aic_with_candidates = []
        for feature in remaining_features:
            current_features = selected_features + [feature]
            model = LogisticRegression(penalty='l2', solver='liblinear')
            model.fit(X[current_features], y)
            aic = calculate_aic(model, X[current_features], y)
            aic_with_candidates.append((feature, aic))
        
        # Select the feature with the best (lowest) AIC
        best_feature, best_aic_candidate = min(aic_with_candidates, key=lambda x: x[1])
        
        if best_aic_candidate < best_aic:
            selected_features.append(best_feature)
            best_aic = best_aic_candidate
            remaining_features.remove(best_feature)
        else:
            break  # If no improvement in AIC, stop
        
    return selected_features, best_aic

def backward_stepwise_selection(X, y):
    selected_features = list(X.columns)
    best_aic = np.inf
    while selected_features:
        aic_with_candidates = []
        for feature in selected_features:
            current_features = [f for f in selected_features if f != feature]
            model = LogisticRegression(penalty='l2', solver='liblinear')
            model.fit(X[current_features], y)
            aic = calculate_aic(model, X[current_features], y)
            aic_with_candidates.append((feature, aic))
        
        # Select the feature whose removal gives the best (lowest) AIC
        worst_feature, worst_aic_candidate = min(aic_with_candidates, key=lambda x: x[1])
        
        if worst_aic_candidate < best_aic:
            selected_features.remove(worst_feature)
            best_aic = worst_aic_candidate
        else:
            break  # If no improvement in AIC, stop
        
    return selected_features, best_aic

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

def get_unique(x_array, y_array):
    indices = []
    x_unique = np.unique(x_array)
    for elt in x_unique:
        indices.append(np.where(x_array==elt)[0][0])
    return x_unique, y_array[indices]

def get_times_values(data, feature_name):
    return data["Age @ sample (yrs, 2 decimal)"].values, data[feature_name].values

def get_lowess_inerpolations(data, feature):
    X, y = get_times_values(data, feature)
    sorted_idx = np.argsort(X)
    X = X[sorted_idx]
    y = y[sorted_idx]
    lowess_score = lowess(y, X, frac=0.3)
    lx = lowess_score[:,0]
    ly = lowess_score[:,1]
    lx, ly = get_unique(lx, ly)
    interp_func = interp1d(lx, ly, kind='linear')
    return interp_func, np.max(lx)

def my_scaling(df, col_names):
    scaler = StandardScaler()
    df[col_names] = scaler.fit_transform(df[col_names])
    return df

def get_distance(df0, df1, feature):
    
    interp_func0, max0 = get_lowess_inerpolations(df0, feature)
    interp_func1, max1 = get_lowess_inerpolations(df1, feature)
    x_space = np.linspace(0, min(max0, max1)-1, 100)
    y0_interpolated = interp_func0(x_space)
    y1_interpolated = interp_func1(x_space)
    y0_interpolated = np.diff(y0_interpolated)
    y1_interpolated = np.diff(y1_interpolated)
    return cosine_similarity(y0_interpolated, y1_interpolated)

def get_lowess_interpolations(data, feature):
    X, y = get_times_values(data, feature)
    lowess_score = lowess(y, X, frac=0.3)
    lx = lowess_score[:,0]
    ly = lowess_score[:,1]
    lx, ly = get_unique(lx, ly)
    interp_func = interp1d(lx, ly, kind='linear', fill_value="nan") # fill_value=nan
    return interp_func, np.max(lx)

def run_lowess_classification(df_train, df_test, variables):
    cutoff = 0.6
    target_specificity = 0.9
    df0 = df_train.loc[df_train["CASE CONTROL STATUS"] == 0]
    df1 = df_train.loc[df_train["CASE CONTROL STATUS"] == 1]
    test_IDs = list(df_test["VOL IDs FOR EXPERIMENTAL PURPOSES"].unique())
    interp_func0, max0 = get_lowess_interpolations(df0, variables)
    interp_func1, max1 = get_lowess_interpolations(df1, variables) 
    time_thres = min(max0, max1)-cutoff
    y = []
    y_pred = []
    for idx in test_IDs:
        df_i=df_test.loc[df_test["VOL IDs FOR EXPERIMENTAL PURPOSES"]==idx]
        y_i = df_i["CASE CONTROL STATUS"].values[0]
        y.append(y_i)
        times_i = df_i["Age @ sample (yrs, 2 decimal)"].values
        feature_values = df_i[variables].values
        times_i = times_i[times_i<time_thres]
        feature_values = feature_values[len(feature_values)-len(times_i):]
        y0_interpolated = interp_func0(times_i)
        y1_interpolated = interp_func1(times_i)
        diff0 = cosine_similarity(feature_values, y0_interpolated)
        diff1 = cosine_similarity(feature_values, y1_interpolated)
        y_pred.append(diff1/ (diff0 + diff1))
    roc_auc = roc_auc_score(y, y_pred)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - target_specificity))
    target_sensitivity = tpr[idx]
    return roc_auc, target_sensitivity



if __name__ == "__main__":
    dir = "../data/LungCA_smoking_diabetes.xlsx"
    df = pd.read_excel(dir)
