from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
from train import train_model
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/churn_modelagem.csv")

## Treinamento
model, y_teste, y_pred, y_prob = train_model(df)

## Métricas
acc = accuracy_score (
    y_teste,
    y_pred
)

precision = precision_score (
    y_teste,
    y_pred,
    pos_label='Yes'
)

recall = recall_score (
    y_teste,
    y_pred,
    pos_label='Yes'
)

f1 = f1_score (
    y_teste,
    y_pred,
    pos_label='Yes'
)

roc_auc = roc_auc_score(
    y_teste.map({
        "No": 0,
        "Yes": 1
    }),
    y_prob
)

# Matriz de confusão
matriz_confusao = confusion_matrix(
    y_teste,
    y_pred,
    labels=["No", "Yes"]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    matriz_confusao,
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


tn, fp, fn, tp = matriz_confusao.ravel()


# Resultados
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nMatriz de confusão:")
print(matriz_confusao)

print(f"\nTrue Negative: {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"True Positive: {tp}")