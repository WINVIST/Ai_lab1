import pandas as pd
import sqlite3
import socket
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train) * 100
classification_rep_train = classification_report(y_train, y_pred_train, target_names=label_encoder.classes_)

accuracy_test = accuracy_score(y_test, y_pred_test) * 100
classification_rep_test = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_)

print(f"Точность модели на обучающей выборке: {accuracy_train:.2f}%")
print("Отчет о классификации на обучающей выборке:\n", classification_rep_train)

print(f"Точность модели на тестовой выборке: {accuracy_test:.2f}%")
print("Отчет о классификации на тестовой выборке:\n", classification_rep_test)
