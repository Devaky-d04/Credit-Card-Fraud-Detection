# Correlation Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Box Plot of Numerical Features
plt.figure(figsize=(15, 8))
features = ['Amount'] + [f'V{i}' for i in range(1, 29)]
sns.boxplot(data=data[features], orient='h')
plt.title("Box Plot for Numerical Features (Outlier Analysis)")
plt.xlabel("Value")
plt.ylabel("Features")
plt.show()

# Pairplot for Selected Features
sample_data = data.sample(5000, random_state=42)  # Sampling for readability
sns.pairplot(sample_data, vars=['V1', 'V2', 'V3', 'Amount'], hue='Class', palette={0: "blue", 1: "red"}, markers=["o", "D"])
plt.suptitle("Pairplot of Selected Features for Fraud and Non-Fraud Transactions")
plt.show()

# Splitting Features and Target
X = data.drop(columns=['Class'])  # Drop the target column
y = data['Class']  # Target column

# Scaling Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_smote)
plt.title('Class Distribution After SMOTE (0: Not Fraud, 1: Fraud)')
plt.show()

# PCA Visualization of SMOTE Data
pca = PCA(n_components=2)
X_train_smote_2D = pca.fit_transform(X_train_smote)
smote_df = pd.DataFrame(X_train_smote_2D, columns=['PCA1', 'PCA2'])
smote_df['Class'] = y_train_smote

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Class', data=smote_df, palette={0: "blue", 1: "red"}, alpha=0.5)
plt.title('SMOTE Resampled Data in 2D PCA Space')
plt.legend(['Not Fraud', 'Fraud'])
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(data[features].sample(5000, random_state=42))  # Sampling to speed up
tsne_df = pd.DataFrame(tsne_data, columns=['TSNE1', 'TSNE2'])
tsne_df['Class'] = data['Class'].sample(5000, random_state=42).values

plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Class', data=tsne_df, palette={0: "blue", 1: "red"}, alpha=0.6)
plt.title("t-SNE Plot of Fraud and Non-Fraud Transactions")
plt.legend(['Not Fraud', 'Fraud'])
plt.show()

