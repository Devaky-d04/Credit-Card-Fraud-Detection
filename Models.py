# Define models with low computational complexity
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),  # Limit depth for simplicity
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'LightGBM': LGBMClassifier(max_depth=3, num_leaves=20)  # Limited complexity for efficiency
}

# Logistic Regression
lr = models['Logistic Regression']
lr.fit(X_train_smote, y_train_smote)
y_pred_lr = lr.predict(X_test)
y_pred_lr_proba = lr.predict_proba(X_test)[:, 1]

print("\nModel: Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
roc_auc_lr = roc_auc_score(y_test, y_pred_lr_proba)
print("ROC AUC Score:", roc_auc_lr)
fpr, tpr, _ = roc_curve(y_test, y_pred_lr_proba)
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})")

# Naive Bayes
nb = models['Naive Bayes']
nb.fit(X_train_smote, y_train_smote)
y_pred_nb = nb.predict(X_test)
y_pred_nb_proba = nb.predict_proba(X_test)[:, 1]

print("\nModel: Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))
roc_auc_nb = roc_auc_score(y_test, y_pred_nb_proba)
print("ROC AUC Score:", roc_auc_nb)
fpr, tpr, _ = roc_curve(y_test, y_pred_nb_proba)
plt.plot(fpr, tpr, label=f"Naive Bayes (AUC = {roc_auc_nb:.2f})")

# Decision Tree
dt = models['Decision Tree']
dt.fit(X_train_smote, y_train_smote)
y_pred_dt = dt.predict(X_test)
y_pred_dt_proba = dt.predict_proba(X_test)[:, 1]

print("\nModel: Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
roc_auc_dt = roc_auc_score(y_test, y_pred_dt_proba)
print("ROC AUC Score:", roc_auc_dt)
fpr, tpr, _ = roc_curve(y_test, y_pred_dt_proba)
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {roc_auc_dt:.2f})")

# LightGBM
lgbm = models['LightGBM']
lgbm.fit(X_train_smote, y_train_smote)
y_pred_lgbm = lgbm.predict(X_test)
y_pred_lgbm_proba = lgbm.predict_proba(X_test)[:, 1]

print("\nModel: LightGBM")
print("Accuracy:", accuracy_score(y_test, y_pred_lgbm))
print("Classification Report:\n", classification_report(y_test, y_pred_lgbm))
roc_auc_lgbm = roc_auc_score(y_test, y_pred_lgbm_proba)
print("ROC AUC Score:", roc_auc_lgbm)
fpr, tpr, _ = roc_curve(y_test, y_pred_lgbm_proba)
plt.plot(fpr, tpr, label=f"LightGBM (AUC = {roc_auc_lgbm:.2f})")


# Initialize LinearSVC with optimized parameters for faster computation
svm = LinearSVC(max_iter=1000, tol=1e-3, dual=False)  # dual=False for faster computation with large datasets
svm.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_svm = svm.predict(X_test)
y_pred_svm_proba = svm.decision_function(X_test)

# Display evaluation metrics
print("Model: LinearSVC")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Compute ROC AUC score and plot ROC curve
roc_auc_svm = roc_auc_score(y_test, y_pred_svm_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_svm_proba)

plt.plot(fpr, tpr, label=f"LinearSVC (AUC = {roc_auc_svm:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LinearSVC")
plt.legend()
plt.show()

# Optimized Autoencoder Architecture with fewer layers and smaller dimensions
autoencoder = Sequential()

# Encoding layer (compressed representation)
autoencoder.add(Dense(64, input_dim=X_train_smote.shape[1], activation='relu'))  # Reduced neurons

# Bottleneck layer (lowest dimensionality)
autoencoder.add(Dense(32, activation='relu'))  # Smaller size

# Decoding layer (reconstruction)
autoencoder.add(Dense(64, activation='relu'))  # Symmetric layer
autoencoder.add(Dense(X_train_smote.shape[1], activation='sigmoid'))  # Output layer with original input size

# Compile the model with Adam optimizer for faster convergence
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
autoencoder.fit(X_train_smote, X_train_smote, epochs=10, batch_size=32, validation_data=(X_test, X_test))

import matplotlib.pyplot as plt
import numpy as np

# Training the autoencoder and capturing the training history
history = autoencoder.fit(X_train_smote, X_train_smote, epochs=10, batch_size=32, validation_data=(X_test, X_test))

# 1. Visualize the Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 2. Visualize Original vs Reconstructed Data (For the first sample in the test set)
# Get the reconstructed data by passing the test data through the autoencoder
reconstructed = autoencoder.predict(X_test)

# Visualizing the first sample
sample_idx = 0
plt.figure(figsize=(8, 4))

# Plot original data
plt.subplot(1, 2, 1)
plt.plot(X_test[sample_idx], label='Original Data')
plt.title('Original Data')
plt.xlabel('Feature Index')
plt.ylabel('Value')

# Plot reconstructed data
plt.subplot(1, 2, 2)
plt.plot(reconstructed[sample_idx], label='Reconstructed Data')
plt.title('Reconstructed Data')
plt.xlabel('Feature Index')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Train Random Forest and get predictions
rf = RandomForestClassifier(n_estimators=100).fit(X_train_smote, y_train_smote)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Print Accuracy and Classification Report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Calculate and plot ROC AUC Curve
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb.predict(X_test)
y_pred_xgb_proba = xgb.predict_proba(X_test)[:, 1]

# Evaluation Metrics
print("\nModel: XGBoost")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# ROC AUC Score and Curve
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb_proba)
print("ROC AUC Score:", roc_auc_xgb)
fpr, tpr, _ = roc_curve(y_test, y_pred_xgb_proba)
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_xgb:.2f})")

