import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from the pickle file
data_dict = pickle.load(open(r"C:\Users\krish\final_yoga_data.pickle", 'rb'))

# Convert data and labels to numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions that how many images are classified as correct 
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Calculating the average confidence score
probabilities = model.predict_proba(x_test)
confidence_scores = []

# Map the predicted class to its respective index in `model.classes_`
for i in range(len(y_predict)):
    predicted_label = y_predict[i]
    class_index = np.where(model.classes_ == predicted_label)[0][0]  # Get index of the predicted class
    confidence = probabilities[i][class_index]  # Confidence score for the predicted class
    confidence_scores.append(confidence)

average_confidence = np.mean(confidence_scores)
print('Average confidence score: {:.2f}%'.format(average_confidence * 100))

# Save the model to a pickle file
with open('final_yoga_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
