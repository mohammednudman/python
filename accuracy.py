# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.model_selection import cross_val_score

# Perform k-fold cross-validation and evaluate the model's performance
scores = cross_val_score(model, X, y, cv=5)

# %%
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)

# %%
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# %%
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true, y_pred)

# %%
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
auc = roc_auc_score(y_true, anomaly_scores)

# %%
# Visualize anomalies on top of the data
plt.scatter(X[:, 0], X[:, 1], s=100, edgecolors='k', c=y_pred, cmap=plt.cm.coolwarm)
plt.title("Anomalies Detected")
plt.show()

# %% [markdown]
# HYPER TUNING    

# %%
from sklearn.model_selection import GridSearchCV

# %%
from sklearn.ensemble import IsolationForest

model = IsolationForest()

# %%
param_grid = {
    'contamination': [0.01, 0.05, 0.1, 0.2, 0.3]
}

# %%
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# %%
grid_search.fit(X, y)

# %%
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# %%
best_model = best_estimator
y_pred = best_model.predict(X_test)


