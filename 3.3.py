import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Подключение к базе данных и загрузка данных
# Используем меньший объем данных для ускорения выполнения
db_path = 'server_logs_600k.db'  # Укажите путь к вашей базе данных
conn = sqlite3.connect(db_path)
data = pd.read_sql_query("SELECT * FROM logs LIMIT 100000", conn)
conn.close()

# Исключение целевой переменной и обработка признаков
y = data['Attack Type']
X = data.drop(columns=['Attack Type', 'User-Agent'])  # Исключаем 'User-Agent' из-за высокой кардинальности

# Обработка категориальных данных
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Кодировка целевой переменной
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Применение SMOTE для балансировки классов
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Настройки для эксперимента
hidden_layer_sizes = [(50,), (100,)]  # Сокращаем количество параметров
optimizer = 'adam'  # Используем только один оптимизатор
results = []

# Эксперимент с различными параметрами нейронной сети
for hidden_size in hidden_layer_sizes:
    model = MLPClassifier(hidden_layer_sizes=hidden_size, solver=optimizer, max_iter=300, tol=1e-4, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({'hidden_size': hidden_size, 'accuracy': accuracy})

# Построение графика зависимости точности от параметров
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.plot(results_df['hidden_size'].astype(str), results_df['accuracy'], marker='o', label='Accuracy')
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')
plt.title('Model Accuracy by Hidden Layer Sizes')
plt.legend()
plt.grid()
plt.show()

# Использование лучшей модели для оценки метрик
best_model_params = results_df.loc[results_df['accuracy'].idxmax()]
best_model = MLPClassifier(hidden_layer_sizes=best_model_params['hidden_size'],
                           solver=optimizer,
                           max_iter=300,
                           tol=1e-4,
                           random_state=42)
best_model.fit(X_train_scaled, y_train)

# Предсказания и оценка лучшей модели
y_pred = best_model.predict(X_test_scaled)
y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

# ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

# Precision-Recall кривая
precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=1)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()

# Краткий отчет о классификации
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)
