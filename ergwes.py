import pandas as pd
import sqlite3
import socket
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


conn = sqlite3.connect('server_logs_250000rows.db')


data = pd.read_sql_query("SELECT * FROM logs", conn)


conn.close()


data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
data = data.dropna(subset=['Timestamp'])
data['Timestamp'] = data['Timestamp'].astype('int64') // 10**9

def encode_ip(ip):
    try:
        return int.from_bytes(socket.inet_aton(ip), 'big')
    except OSError:
        return None

data['IP'] = data['IP'].apply(encode_ip)
data = data.dropna(subset=['IP'])

data['IP'] = data['IP'].astype('int64')


X = data.drop(columns=['Attack Type'])
y = data['Attack Type']


categorical_columns = ['HTTP Method', 'URL', 'User-Agent']
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)


param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


best_model.fit(X_train, y_train)


y_pred_train = best_model.predict(X_train)


y_pred_test = best_model.predict(X_test)


accuracy_train = accuracy_score(y_train, y_pred_train) * 100  # Умножение на 100 для получения процентов
classification_rep_train = classification_report(y_train, y_pred_train, target_names=label_encoder.classes_)


accuracy_test = accuracy_score(y_test, y_pred_test) * 100  # Умножение на 100 для получения процентов
classification_rep_test = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_)


print(f"Точность модели на обучающей выборке: {accuracy_train:.2f}%")
print("Отчет о классификации на обучающей выборке:\n", classification_rep_train)

print(f"Точность модели на тестовой выборке: {accuracy_test:.2f}%")
print("Отчет о классификации на тестовой выборке:\n", classification_rep_test)
