import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_score, completeness_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Укажите количество логических ядер


# 1. Загрузка данных из базы данных
connection = sqlite3.connect('server_logs_600k.db')
data_query = "SELECT * FROM logs"
data = pd.read_sql(data_query, connection)
connection.close()

# 2. Предварительная обработка данных
# Конвертация временной метки и извлечение признаков времени
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month
data['Year'] = data['Timestamp'].dt.year
data['Weekday'] = data['Timestamp'].dt.weekday

# Кодирование целевой переменной (Attack Type)
attack_mapping = {"DDoS": 1, "SQL Injection": 2, "XSS": 3, "Brute Force": 4, "No Attack": 0}
data['AttackType'] = data['Attack Type'].map(attack_mapping)

# Исключение нерелевантных столбцов
data.drop(columns=['Timestamp', 'IP', 'URL', 'User-Agent', 'Attack Type'], inplace=True)

# One-hot encoding для категориальных данных
categorical_columns = ['HTTP Method', 'Status']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Масштабирование данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['AttackType']))

# 3. Кластеризация данных
n_clusters = len(attack_mapping)  # Используем количество атак как число кластеров
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# 4. Вычисление метрик кластеризации
homogeneity = homogeneity_score(data['AttackType'], data['Cluster'])
completeness = completeness_score(data['AttackType'], data['Cluster'])

print(f"Homogeneity Score: {homogeneity:.2f}")
print(f"Completeness Score: {completeness:.2f}")

# 5. Визуализация кластеров
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
data['PCA1'] = pca_data[:, 0]
data['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(10, 6))
for cluster_label in range(n_clusters):
    cluster_points = data[data['Cluster'] == cluster_label]
    plt.scatter(cluster_points['PCA1'], cluster_points['PCA2'], label=f'Cluster {cluster_label}')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Cluster Visualization')
plt.legend()
plt.show()
