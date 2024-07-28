import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the raw data
data = pd.read_csv('IRIS.csv')

# Data pre-processing
# Drop any missing values
data.dropna(inplace=True)

# Data cleaning
# Remove any duplicate rows
data.drop_duplicates(inplace=True)

# Data transformation
# Convert categorical variables into numerical format using one-hot encoding
data = pd.get_dummies(data)
print(data)
# Prepare data for AI model
# Split the data into input features and target variable
X = data.iloc[:,:5]
y = data.iloc[:,5:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply AI algorithm (Random Forest classifier)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print('Model Accuracy:', accuracy)