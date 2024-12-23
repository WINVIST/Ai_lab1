import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import numpy as np
import logging
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

def get_metrics(cls, name, y_true, X_test):
    logger.info("Generating metrics...")
    y_pred = cls.predict(X_test)
    logger.debug(f"Predictions: {y_pred}")
    r = np.flip(confusion_matrix(y_true, y_pred))
    print("\nМатрица ошибок")
    print("[[FF FP]\n [TF TT]]")
    print(r)

    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {acc}")
    print("\nAccuracy: ", acc)

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    logger.info(f"Precision: {precision}")
    print("Precision: ", precision)

    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    logger.info(f"Recall: {recall}")
    print("Recall: ", recall)

    # ROC and Precision-Recall for each class
    y_scores = cls.predict_proba(X_test)
    roc_auc_scores = []
    for i, class_label in enumerate(cls.classes_):
        logger.info(f"Calculating ROC and Precision-Recall for class {class_label}...")
        fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve for class {class_label}')
        plt.legend()
        plt.grid()
        plt.show()

        precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{name} Precision-Recall Curve for class {class_label}')
        plt.legend()
        plt.grid()
        plt.show()

    # Calculate and log the average ROC AUC score
    avg_roc_auc = roc_auc_score(pd.get_dummies(y_true), y_scores, multi_class='ovr')
    logger.info(f"Average ROC AUC: {avg_roc_auc}")
    print(f"\nAverage ROC AUC: {avg_roc_auc}")

    # Classification Report
    report = classification_report(y_true, y_pred)
    logger.info("Classification report generated.")
    print("\nClassification Report:")
    print(report)

def experiment(X, Y):
    logger.info("Starting experiment with different parameters...")
    fig, asx = plt.subplots(1, 2)
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    solvers = ['lbfgs', 'adam']
    names = [[''] * 3]*2
    plots_layers = [[0] * 3]*2
    scores = [[[0.]*10] * 3] * 2

    for solver_num, solver in enumerate(solvers):
        logger.info(f'Predict with solver = {solver}')
        logger.info('Processing layers')
        for layers in range(3):
            for i in range(10, 110, 10):
                mpl = MLPClassifier(hidden_layer_sizes=[i]*(layers+1), solver=solver, max_iter=1000, alpha=0.001, random_state=42)
                mpl.fit(x_train, y_train)
                score = mpl.score(x_test, y_test)
                logger.debug(f"Layers: {layers + 1}, Neurons: {i}, Score: {score}")
                scores[solver_num][layers][i//10 - 1] = score
                logger.info(f"Solver: {solver}, Layers: {layers + 1}, Neurons per Layer: {i}, Score: {score}")
            plots_layers[solver_num][layers] = asx[solver_num].plot(range(10, 110, 10), scores[solver_num][layers])[0]
            names[solver_num][layers] = f"{layers + 1} layers"

    for i in range(2):
        asx[i].set_title(f'Точность [{solvers[i]}]')
        asx[i].legend(plots_layers[i], names[i])

    plt.tight_layout()
    plt.show()
    logger.info("Experiment completed.")

def predict(X, Y):
    logger.info("Starting prediction...")
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    logger.debug("Training data scaled.")
    x_test_scaled = scaler.transform(x_test)
    logger.debug("Test data scaled.")

    mpl = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=[60, 60], solver='lbfgs', max_iter=1000, alpha=0.001, random_state=42))
    logger.info("Fitting MLP model...")

    with tqdm(total=x_train_scaled.shape[0], desc="Training Progress") as pbar:
        for i in range(1):  # Simulated loop for better visualization
            mpl.fit(x_train_scaled, y_train)
            pbar.update(x_train_scaled.shape[0] // 10)

    logger.info("Model fitting completed. Generating metrics...")
    plt.figure(figsize=(6, 4))
    plt.bar(['Train', 'Test'], [mpl.score(x_train_scaled, y_train), mpl.score(x_test_scaled, y_test)], color=['blue', 'orange'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.show()

    get_metrics(mpl, "MLP", y_test, x_test_scaled)
    logger.info("Prediction completed.")

if __name__ == '__main__':
    logger.info("Connecting to the database...")
    db_path = 'server_logs_250000rows.db'
    conn = sqlite3.connect(db_path)

    try:
        data = pd.read_sql_query("SELECT * FROM logs", conn)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    finally:
        conn.close()

    logger.info("Preparing data...")
    y = data['Attack Type']
    X = data.drop(columns=['Attack Type', 'User-Agent'])

    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    logger.info("Data preparation completed.")

    # Uncomment the desired action below:
    # experiment(X_resampled, y_resampled)
    predict(X_resampled, y_resampled)
