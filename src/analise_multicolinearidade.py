from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd


df = pd.read_csv("data/processed/churn_modelagem.csv")


# Separação dos atributos e do alvo
x = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# Separação treino/teste
x_train, x_teste, y_train, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# MonthlyCharges isoladamente
scaler_monthly = StandardScaler()

x_monthly_train = scaler_monthly.fit_transform(
    x_train[["MonthlyCharges"]]
)

model_monthly = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model_monthly.fit(
    x_monthly_train,
    y_train
)


# TotalCharges isoladamente
scaler_total = StandardScaler()

x_total_train = scaler_total.fit_transform(
    x_train[["TotalCharges"]]
)

model_total = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model_total.fit(
    x_total_train,
    y_train
)


print(
    "Coeficiente MonthlyCharges:",
    model_monthly.coef_[0][0]
)

print(
    "Coeficiente TotalCharges:",
    model_total.coef_[0][0]
)




# tenure + MonthlyCharges

scaler_monthly_tenure = StandardScaler()

x_monthly_tenure_train = scaler_monthly_tenure.fit_transform(
    x_train[["tenure", "MonthlyCharges"]]
)

model_monthly_tenure = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model_monthly_tenure.fit(
    x_monthly_tenure_train,
    y_train
)


print(
    "\nCoeficientes - tenure + MonthlyCharges:"
)

print(
    "tenure:",
    model_monthly_tenure.coef_[0][0]
)

print(
    "MonthlyCharges:",
    model_monthly_tenure.coef_[0][1]
)

## Tenure x totalCharges
scaler_TotalC_tenure = StandardScaler()

x_TotalC_tenure_train = scaler_TotalC_tenure.fit_transform(
    x_train[["tenure", "TotalCharges"]]
)

model_TotalC_tenure = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model_TotalC_tenure.fit(
    x_TotalC_tenure_train,
    y_train
)

print(
    "\nCoeficientes - tenure + TotalCharges:"
)

print(
    "tenure:",
    model_TotalC_tenure.coef_[0][0]
)

print(
    "TotalCharges:",
    model_TotalC_tenure.coef_[0][1]
)



## Tenure x MonthlyCharges x totalCharges
scaler_TotalC_Monthly_tenure = StandardScaler()

x_TotalC_Monthly_tenure_train = scaler_TotalC_Monthly_tenure.fit_transform(
    x_train[["tenure", "TotalCharges","MonthlyCharges"]]
)

model_TotalC_Monthly_tenure = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model_TotalC_Monthly_tenure.fit(
    x_TotalC_Monthly_tenure_train,
    y_train
)

print(
    "\nCoeficientes - tenure + TotalCharges:"
)

print(
    "tenure:",
    model_TotalC_Monthly_tenure.coef_[0][0]
)

print(
    "TotalCharges:",
    model_TotalC_Monthly_tenure.coef_[0][1]
)

print(
    "MonthlyCharges:",
    model_TotalC_Monthly_tenure.coef_[0][2]
)
