import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_excel("student_performance.csv.xlsx")  # Excel input
print("Dataset Preview:\n", data.head())

X = data.drop("result", axis=1)
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))


try:
    new_students = pd.read_excel("new_students.xlsx")
    predictions = dt_model.predict(new_students)
    new_students['predicted_result'] = ["PASS ✅" if p==1 else "FAIL ❌" for p in predictions]

    new_students.to_excel("new_students_predictions.xlsx", index=False)
    print("\nPredictions saved to 'new_students_predictions.xlsx'")
except FileNotFoundError:
    print("\nNo 'new_students.xlsx' file found. Skipping new student predictions.")

data['predicted_result'] = dt_model.predict(X)
data['predicted_result'] = data['predicted_result'].apply(lambda x: "PASS ✅" if x==1 else "FAIL ❌")
data.to_excel("student_performance_with_predictions.xlsx", index=False)
print("Predictions for original dataset saved to 'student_performance_with_predictions.xlsx'")
