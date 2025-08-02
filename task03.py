# ==============================================================================
# Step 1: Import Libraries and Load Data
# ==============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# Load the dataset from a local file
# Ensure 'bank-full.csv' is in the same folder as your script
try:
    df = pd.read_csv('bank-full.csv', sep=';')
except FileNotFoundError:
    print("Error: 'bank-full.csv' not found. Please download the dataset and place it in the same folder as the script.")
    exit()


# ==============================================================================
# Step 2: Data Exploration and Preprocessing
# ==============================================================================

print("--- Dataset Information ---")
df.info()

# Create a copy for processing
df_processed = df.copy()

# Identify categorical columns to be encoded
categorical_cols = df_processed.select_dtypes(include=['object']).columns

# Apply LabelEncoder to convert each categorical column to numbers
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])

print("\n--- First 5 Rows of Processed Data ---")
print(df_processed.head())


# ==============================================================================
# Step 3: Define Features (X) and Target (y)
# ==============================================================================

# X contains all the feature columns (everything except the target 'y')
X = df_processed.drop('y', axis=1)

# y contains only the target column
y = df_processed['y']


# ==============================================================================
# Step 4: Split Data into Training and Testing Sets
# ==============================================================================

# Split the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# ==============================================================================
# Step 5: Build and Train the Decision Tree Model
# ==============================================================================

# Create the Decision Tree model with improvements
# max_depth prevents overfitting
# class_weight='balanced' helps with the imbalanced dataset
# ccp_alpha introduces cost-complexity pruning
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    class_weight='balanced',
    ccp_alpha=0.001  # Added hyperparameter for pruning
)

# Train the model using the training data
model.fit(X_train, y_train)

print("\n✅ Decision Tree model has been trained successfully.")


# ==============================================================================
# Step 6: Evaluate the Model's Performance
# ==============================================================================

# Make predictions on the unseen test data
y_pred = model.predict(X_test)

# --- 1. Accuracy Score ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# --- 2. Confusion Matrix ---
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix for better interpretation
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- 3. Classification Report ---
# Provides detailed metrics like precision, recall, and f1-score
class_report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
print("\nClassification Report:")
print(class_report)


# ==============================================================================
# Step 7: Visualize the Decision Tree
# ==============================================================================

print("\nDisplaying the trained Decision Tree...")
plt.figure(figsize=(25, 15))
plot_tree(model,
          feature_names=X.columns,
          class_names=['No', 'Yes'],  # Corresponds to 0 and 1 from LabelEncoder
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Bank Marketing Prediction", fontsize=20)
plt.show()
