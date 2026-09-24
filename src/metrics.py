from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from train import train_logistic_regression, train_random_forest


# CARREGAMENTO DOS DADOS

df = pd.read_csv("data/processed/churn_modelagem.csv")


# TREINAMENTO DOS MODELOS

resultados_logistic = train_logistic_regression(df)

resultados_rf = train_random_forest(df)


# LOGISTIC REGRESSION

(
    model_logistic,
    preprocessor_logistic,
    y_teste_logistic,
    y_pred_logistic,
    y_prob_logistic
) = resultados_logistic

acc_logistic = accuracy_score(
    y_teste_logistic,
    y_pred_logistic
)

precision_logistic = precision_score(
    y_teste_logistic,
    y_pred_logistic,
    pos_label="Yes"
)

recall_logistic = recall_score(
    y_teste_logistic,
    y_pred_logistic,
    pos_label="Yes"
)

f1_logistic = f1_score(
    y_teste_logistic,
    y_pred_logistic,
    pos_label="Yes"
)

roc_auc_logistic = roc_auc_score(
    y_teste_logistic.map({
        "No": 0,
        "Yes": 1
    }),
    y_prob_logistic
)


matriz_logistic = confusion_matrix(
    y_teste_logistic,
    y_pred_logistic,
    labels=["No", "Yes"]
)

tn_logistic, fp_logistic, fn_logistic, tp_logistic = (
    matriz_logistic.ravel()
)

# RANDOM FOREST

(
    model_rf,
    preprocessor_rf,
    y_teste_rf,
    y_pred_rf,
    y_prob_rf
) = resultados_rf


acc_rf = accuracy_score(
    y_teste_rf,
    y_pred_rf
)

precision_rf = precision_score(
    y_teste_rf,
    y_pred_rf,
    pos_label="Yes"
)

recall_rf = recall_score(
    y_teste_rf,
    y_pred_rf,
    pos_label="Yes"
)

f1_rf = f1_score(
    y_teste_rf,
    y_pred_rf,
    pos_label="Yes"
)

roc_auc_rf = roc_auc_score(
    y_teste_rf.map({
        "No": 0,
        "Yes": 1
    }),
    y_prob_rf
)


matriz_rf = confusion_matrix(
    y_teste_rf,
    y_pred_rf,
    labels=["No", "Yes"]
)

tn_rf, fp_rf, fn_rf, tp_rf = (
    matriz_rf.ravel()
)

# MATRIZ DE CONFUSÃO - LOGISTIC REGRESSION

plt.figure(figsize=(6, 5))

sns.heatmap(
    matriz_logistic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Logistic Regression")

plt.tight_layout()
plt.show()

# MATRIZ DE CONFUSÃO - RANDOM FOREST

plt.figure(figsize=(6, 5))

sns.heatmap(
    matriz_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Random Forest")

plt.tight_layout()
plt.show()

# RESULTADOS - LOGISTIC REGRESSION

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

print(f"Accuracy: {acc_logistic:.4f}")
print(f"Precision: {precision_logistic:.4f}")
print(f"Recall: {recall_logistic:.4f}")
print(f"F1-score: {f1_logistic:.4f}")
print(f"ROC-AUC: {roc_auc_logistic:.4f}")

print(f"\nTrue Negative: {tn_logistic}")
print(f"False Positive: {fp_logistic}")
print(f"False Negative: {fn_logistic}")
print(f"True Positive: {tp_logistic}")

# RESULTADOS - RANDOM FOREST

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

print(f"Accuracy: {acc_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-score: {f1_rf:.4f}")
print(f"ROC-AUC: {roc_auc_rf:.4f}")

print(f"\nTrue Negative: {tn_rf}")
print(f"False Positive: {fp_rf}")
print(f"False Negative: {fn_rf}")
print(f"True Positive: {tp_rf}")