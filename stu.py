import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# study hours, attendance % → pass or fail
X = np.array([
    [2, 40], [3, 55], [5, 70], [6, 80], [8, 90],
    [1, 30], [4, 60], [7, 85], [9, 95], [2, 35],
    [5, 65], [6, 75], [8, 88], [3, 50], [7, 80],
    [1, 25], [4, 55], [6, 78], [9, 92], [3, 48]
])
y = np.array([0,0,1,1,1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation='relu',
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.1f}%")

new_student = np.array([[6, 75]])
result = model.predict(new_student)[0]
prob = model.predict_proba(new_student)[0][1]

print(f"Prediction: {'PASS' if result == 1 else 'FAIL'}")
print(f"Pass probability: {prob:.0%}")