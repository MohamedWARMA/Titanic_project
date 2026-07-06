import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('titanic_clean.csv')

#Separer les features (X) et cinle (Y)

x = df.drop(columns=['survived'])
y= df['survived']

#Split train / test
X_train , X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))


print(classification_report(y_test, y_pred))


print("Accuracy : ", round(accuracy, 3))


print("Train : ", X_train.shape)
print("Test : ", X_test.shape)