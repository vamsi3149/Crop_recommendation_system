import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, f1_score

# Load the crop data
crop_data = pd.read_csv("E:\python_testing\Crop_recommendation.csv")

# Split the data into features and target
X = crop_data[['temperature', 'humidity', 'N', 'P', 'K']]
y = crop_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a gradient boosting classifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

accuracy = np.mean(y_pred == y_test)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('F1 Score:', f1)
# Make predictions for new data
new_data = pd.DataFrame([[25, 60, 100, 50, 50]], columns=['temperature', 'humidity', 'N', 'P', 'K'])
new_pred = clf.predict(new_data)

# Print the prediction
print('Predicted crop:', new_pred[0])
