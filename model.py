from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.metrics import accuracy_score
df = pd.read_csv('Stress Data_C.csv')
df.head()
# Select dependent and independent variables
X = df[['snoring_rate', 'body_temperature','blood_oxygen','respiration_rate', 'sleeping_hours', 'heart_rate','Headache','Working_hours']]
y = df['stress_level']
# Split the dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
xgboost_classifier = xgb.XGBClassifier(random_state=42)
# Fit the model
xgboost_classifier.fit(X_train, y_train)
# Predict and evaluate accuracy
y_pred_xgb = xgboost_classifier.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")
# Save the model
pickle.dump(xgboost_classifier, open('XGBoost_model.pkl', 'wb'))
mlp_classifier = MLPClassifier(random_state=42)
# Fit the model
mlp_classifier.fit(X_train, y_train)
# Predict and evaluate accuracy
y_pred_mlp = mlp_classifier.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Accuracy: {accuracy_mlp:.4f}")
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
# Save the model
pickle.dump(random_forest, open('random_forest_model.pkl', 'wb'))
loaded_model = pickle.load(open('random_forest_model.pkl', 'rb'))
snoring_rate = 90 #@param {type:"number"}
body_temperature = 98.6 #@param {type:"number"}
blood_oxygen = 95 #@param {type:"number"}
respiration_rate = 15 #@param {type:"number"}
sleeping_hours = 7 #@param {type:"number"}
heart_rate = 70 #@param {type:"number"}
Headache = 1 #@param {type:"number"}
Working_hours = 8 #@param {type:"number"}


# Create a DataFrame from the input features
input_data = pd.DataFrame({
    'snoring_rate': [snoring_rate],
    'body_temperature': [body_temperature],
    'blood_oxygen': [blood_oxygen],
    'respiration_rate': [respiration_rate],
    'sleeping_hours': [sleeping_hours],
    'heart_rate': [heart_rate],
    'Headache': [Headache],
    'Working_hours': [Working_hours]
})

# Make prediction
prediction = loaded_model.predict(input_data)

# Print the prediction
print(f"Predicted stress level: {prediction[0]}")
#import modules
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import joblib
# Load the dataset
df = pd.read_csv('Stress Data_C.csv')

# Select dependent and independent variables
X = df[['snoring_rate', 'body_temperature', 'blood_oxygen', 'respiration_rate', 'sleeping_hours', 'heart_rate', 'Headache', 'Working_hours']]
y = df['stress_level']

# Split the dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Instantiate model
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
random_forest_classifier.fit(X_train, y_train)

# Predict stress levels on the test data
predictions = random_forest_classifier.predict(X_test)

# Define function for personalized stress management recommendations
def generate_recommendations(stress_level, user_features):
    """
    Generate personalized stress management recommendations based on the user's stress level
    and input features.
    """
    recommendations = []

    # Stress level-based recommendations
    if stress_level == 0:  # Low stress
        recommendations.append("You are doing great! Continue your current routine and focus on maintaining a balanced lifestyle.")
        recommendations.append("Ensure you are getting sufficient sleep and manage your working hours.")
    elif stress_level == 1:  # Moderate stress
        recommendations.append("Consider engaging in regular physical activities, like walking or yoga.")
        recommendations.append("Try incorporating relaxation techniques such as meditation or deep breathing exercises.")
        recommendations.append("Monitor your sleeping hours and ensure you are getting enough rest.")
    elif stress_level == 2:  # High stress
        recommendations.append("It's important to take a break and prioritize self-care. Consider a short vacation or time away from work.")
        recommendations.append("Start practicing mindfulness meditation or consider seeing a mental health professional for guidance.")
        recommendations.append("Try reducing your working hours and focus on rest and recovery.")

    # Personalized recommendations based on features (e.g., sleep, heart rate, working hours)
    if user_features['sleeping_hours'] < 6:
        recommendations.append("Try to increase your sleeping hours. Aim for at least 7-8 hours per night to manage stress better.")
    if user_features['heart_rate'] > 90:
        recommendations.append("Your heart rate seems elevated. Try engaging in some calming activities like deep breathing.")
    if user_features['Working_hours'] > 8:
        recommendations.append("Consider reducing your working hours to prevent burnout and stress overload.")

    return recommendations

# Example: Generate recommendations for the first test data point
user_features = X_test.iloc[0].to_dict()
stress_level_pred = predictions[0]
recommendations = generate_recommendations(stress_level_pred, user_features)

# Print the recommendations
print(f"Predicted Stress Level: {stress_level_pred}")
print("Personalized Recommendations:")
for rec in recommendations:
    print(f"- {rec}")

# Save the model as a pickle file
pickle.dump(random_forest_classifier, open('model.pklz', 'wb'))