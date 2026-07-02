# Previsão de Churn de Clientes (Telecom)

Modelo de Machine Learning para prever se um cliente de uma empresa de telecomunicações vai cancelar o serviço (churn), com comparação entre dois algoritmos de classificação.

## Objetivo

O objetivo deste projeto foi colocar em prática, de ponta a ponta, os conhecimentos de Machine Learning com Scikit-Learn: desde a limpeza e preparação dos dados até o treinamento, avaliação e comparação de diferentes modelos de classificação — analisando não só qual modelo "acerta mais", mas qual realmente resolve melhor o problema de negócio.

## Dataset

Os dados utilizados são do dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), disponível publicamente no Kaggle. Contém informações reais (anonimizadas) de clientes de uma operadora de telecom, incluindo dados demográficos, tipo de contrato, serviços contratados e se o cliente cancelou ou não o serviço.

> O arquivo CSV não está incluído neste repositório. Para reproduzir o projeto, baixe o dataset no link acima e salve como `WA_Fn-UseC_-Telco-Customer-Churn.csv` na raiz do projeto.

## Metodologia

### 1. Preparação dos dados
- Tratamento da coluna `TotalCharges`, que continha valores inconsistentes (strings vazias) e foi convertida para numérica, preenchendo os casos inválidos com 0.
- Codificação de variáveis categóricas:
  - **Label Encoding** para colunas binárias (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`).
  - **One-Hot Encoding** para colunas com múltiplas categorias (`Contract`, `InternetService`, `PaymentMethod`, entre outras).
- Padronização (`StandardScaler`) das variáveis numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`), ajustada apenas nos dados de treino e aplicada nos dados de teste, para evitar vazamento de informação (data leakage).
- Separação treino/teste com `train_test_split` (70/30, `random_state=42` para reprodutibilidade).

### 2. Balanceamento de classes
O dataset é desbalanceado: a maioria dos clientes não cancela o serviço. Isso faz com que um modelo "ingênuo" possa alcançar alta acurácia geral apenas prevendo "não vai cancelar" na maior parte dos casos — o que é inútil na prática, já que o objetivo é justamente identificar quem vai cancelar.

Para lidar com isso, os dois modelos foram treinados com o parâmetro `class_weight='balanced'`. Em vez de duplicar ou gerar novos dados, essa técnica ajusta internamente a função de perda do modelo, penalizando mais os erros cometidos na classe minoritária (clientes que cancelam). O resultado foi um ganho expressivo de **recall** na classe de churn, ao custo de um pouco de precisão — um trade-off aceitável e até desejável neste tipo de problema, onde deixar passar um cliente que vai cancelar (falso negativo) tende a ser mais custoso para o negócio do que prever um cancelamento que não ocorre (falso positivo).

### 3. Modelos treinados
Dois modelos foram treinados e comparados:
- **Logistic Regression**
- **Random Forest Classifier**

Para o Random Forest, o hiperparâmetro `max_depth` foi ajustado empiricamente. Foram testados valores mais altos e mais baixos de profundidade da árvore; valores altos levaram a overfitting (o modelo memorizava padrões do treino que não generalizavam bem para o teste), enquanto valores muito baixos perdiam capacidade de capturar relações relevantes nos dados. O valor `max_depth=5` apresentou o melhor equilíbrio entre generalização e desempenho durante os testes, e foi o adotado na versão final.

## Resultados

| Métrica (classe Churn) | Logistic Regression | Random Forest |
|---|---|---|
| Acurácia geral | 75,72% | 75,06% |
| Precision | 0,53 | 0,53 |
| Recall | 0,84 | 0,84 |
| F1-score | 0,65 | 0,65 |

Relatório completo de classificação:

**Regressão Logística**
```
              precision    recall  f1-score   support
           0       0.92      0.73      0.81      1539
           1       0.53      0.84      0.65       574
    accuracy                           0.76      2113
   macro avg       0.73      0.78      0.73      2113
weighted avg       0.82      0.76      0.77      2113
```

**Random Forest**
```
              precision    recall  f1-score   support
           0       0.92      0.72      0.81      1539
           1       0.53      0.84      0.65       574
    accuracy                           0.75      2113
   macro avg       0.73      0.78      0.73      2113
weighted avg       0.82      0.75      0.76      2113
```

### Matriz de confusão (Regressão Logística)

![Matriz de Confusão](imagens/MatrizDeConfusão.png)

Dos 574 clientes que realmente cancelaram o serviço no conjunto de teste, o modelo identificou corretamente 481 (recall de 84%), errando em apenas 93 casos. Em contrapartida, 420 clientes que não cancelaram foram classificados como possível churn (falsos positivos) — um resultado esperado e aceitável dado o balanceamento aplicado, já que o foco do projeto foi priorizar a identificação de clientes em risco.

### Importância das variáveis (Random Forest)

![Importância das Variáveis](imagens/Rf-mais-relevantes.png)

## Principais Insights

- **`tenure` (tempo de contrato) é a variável mais relevante** para prever churn, seguida por contratos de 2 anos (`Contract_Two year`) e o valor total já pago (`TotalCharges`). Isso indica que clientes mais novos, sem vínculo de longo prazo, são o grupo de maior risco.
- Clientes com **internet via fibra óptica** e que pagam por **boleto eletrônico (Electronic check)** também aparecem entre as variáveis mais influentes, sugerindo perfis de clientes específicos com maior propensão ao cancelamento.
- Os dois modelos apresentaram **precision e recall praticamente idênticos na classe de churn** (0,53 / 0,84), variando apenas na acurácia geral. Isso mostra que, neste problema, o balanceamento de classes teve impacto tão relevante quanto (ou mais que) a escolha do algoritmo em si — e que um modelo mais simples (Regressão Logística) performou de forma equivalente a um modelo mais complexo (Random Forest).

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-Learn
- Matplotlib

## Como executar

```bash
pip install -r requirements.txt
python classificador.py
```

> Certifique-se de ter baixado o dataset do Kaggle e salvo como `WA_Fn-UseC_-Telco-Customer-Churn.csv` na raiz do projeto antes de rodar o script.
