from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings("ignore")
# Load the dataset
data = load_breast_cancer()

# Access features (data)
X = data.data

# Access target variable (cancer diagnosis)
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to compare
# Create and train the classifiers
lr = LogisticRegression()
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Evaluate the classifiers
models = {'Logistic Regression': lr, 'Decision Tree': dt, 'Random Forest': rf}
metrics = ['accuracy', 'precision', 'recall', 'f1']

all_results = []
for model_name, model in models.items():
  # Make predictions on test data
  y_pred = model.predict(X_test)

  # Calculate evaluation metrics
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')
  f1 = f1_score(y_test, y_pred, average='weighted')

  # Store results
  results = {
      'model_name': model_name,
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1': f1
  }
  all_results.append(results)

# Print the comparison of results
print("Model Evaluation Results:")
for result in all_results:
  print(f"{result['model_name']}: Accuracy - {result['accuracy']:.4f}, Precision - {result['precision']:.4f}, Recall - {result['recall']:.4f}, F1 - {result['f1']:.4f}")