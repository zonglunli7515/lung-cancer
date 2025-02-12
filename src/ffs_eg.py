import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

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

# Example usage with a dataset
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = (data.target == 0).astype(int)  # Binary classification for simplicity

# Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Forward stepwise selection
forward_selected_features, forward_best_aic = forward_stepwise_selection(X_train, y_train)
print(forward_selected_features)