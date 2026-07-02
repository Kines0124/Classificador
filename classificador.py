import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Carrega o arquivo (dataset original fica intacto aqui)
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv.csv', sep=',')

# 2. Cria as variáveis X e y copiando os dados para não afetar o dataset original
y = dataset['Churn'].copy()
x = dataset.copy()

# 3. Remove as colunas que não vão para o treino do X
x = x.drop(['customerID', 'Churn'], axis=1)

# Listas de colunas (Removi o 'Churn' daqui pois já separamos ele no y)
colunas_label = [
    'gender', 
    'Partner', 
    'Dependents', 
    'PhoneService', 
    'PaperlessBilling'
]

colunas_one_hot = [
    'MultipleLines', 
    'InternetService', 
    'OnlineSecurity', 
    'OnlineBackup', 
    'DeviceProtection', 
    'TechSupport', 
    'StreamingTV', 
    'StreamingMovies', 
    'Contract', 
    'PaymentMethod'
]

# 4. Trata o TotalCharges direto no X
x['TotalCharges'] = pd.to_numeric(x['TotalCharges'], errors='coerce').fillna(0)

# 5. Aplica o LabelEncoder na variável alvo y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# 6. Aplica o LabelEncoder nas colunas binárias de X
for i in colunas_label:
    labelencoder1 = LabelEncoder()
    x[i] = labelencoder1.fit_transform(x[i])    
    
# 7. Aplica o One-Hot Encoding nas colunas do X
x = pd.get_dummies(x, columns=colunas_one_hot, drop_first=True, dtype=int)

# 8. Separação dos dados em treino e teste

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

# 9. Normalização dos dados necessarios

colunas_numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']

sc = StandardScaler()
x_treino[colunas_numericas] = sc.fit_transform(x_treino[colunas_numericas])
x_teste[colunas_numericas] = sc.transform(x_teste[colunas_numericas])
                                                   

    
# MODELO 1: Regressão Logística

modelo_logistica = LogisticRegression(class_weight='balanced', random_state=42)

modelo_logistica.fit(x_treino, y_treino)

resultado_logistica = modelo_logistica.predict(x_teste)

# Modelo 2: Random Forest

modelo_rf = RandomForestClassifier(class_weight='balanced', max_depth=5, random_state=42)

modelo_rf.fit(x_treino, y_treino)

resultado_rf = modelo_rf.predict(x_teste)

# ========================================================
# 10. Comparação dos Resultados
# ========================================================

print("\n" + "="*40)
print("AVALIAÇÃO DA REGRESSÃO LOGÍSTICA")
print("="*40)
print(f"Acurácia Geral: {accuracy_score(y_teste, resultado_logistica):.4f}")
print("\nRelatório Detalhado:")
print(classification_report(y_teste, resultado_logistica))

print("\n" + "="*40)
print("AVALIAÇÃO DO RANDOM FOREST")
print("="*40)
print(f"Acurácia Geral: {accuracy_score(y_teste, resultado_rf):.4f}")
print("\nRelatório Detalhado:")
print(classification_report(y_teste, resultado_rf))

# ========================================================
# 11. Gráfico de Importância das Variáveis (Random Forest)
# ========================================================
importancias = modelo_rf.feature_importances_

df_importancias = pd.DataFrame({
    'Variável': x.columns,
    'Importância': importancias
}).sort_values(by='Importância', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(df_importancias['Variável'][::-1], df_importancias['Importância'][::-1], color='teal')
plt.title('Top 10 Variáveis que Mais Afetam o Churn', fontsize=14)
plt.xlabel('Grau de Importância', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ========================================================
# 12. Gráfico da Matriz de Confusão (Regressão Logística)
# ========================================================
cm_logistica = confusion_matrix(y_teste, resultado_logistica)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_logistica, 
    display_labels=['Ficou (0)', 'Churn (1)']
)

disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Regressão Logística")
plt.show()