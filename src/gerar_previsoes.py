import pandas as pd
from sklearn.linear_model import LogisticRegression
from preprocessing import create_preprocessor


ARQUIVO_ENTRADA = "data/processed/churn_tratado.csv"
ARQUIVO_SAIDA = "data/processed/previsoes_churn.csv"

VARIAVEIS_REMOVER = [
    "gender",
    "PhoneService",
    "MultipleLines",
    "Partner",
    "SeniorCitizen",
    "Dependents",
    "PaperlessBilling"
]

THRESHOLD = 0.50


def gerar_previsoes(df):
    customer_id = df["customerID"].copy()

    x = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    x = x.drop(columns=VARIAVEIS_REMOVER)

    preprocessor = create_preprocessor(x)

    x_transformado = preprocessor.fit_transform(x)

    model = LogisticRegression(
        C=1,
        class_weight="balanced",
        max_iter=1000
    )

    model.fit(x_transformado, y)

    y_prob = model.predict_proba(x_transformado)[:, 1]

    y_pred = pd.Series(
        y_prob >= THRESHOLD,
        index=df.index
    ).map({
        False: "No",
        True: "Yes"
    })

    previsoes = df.copy()

    previsoes["probabilidade_churn"] = y_prob
    previsoes["previsao_churn"] = y_pred

    return previsoes


if __name__ == "__main__":
    df = pd.read_csv(ARQUIVO_ENTRADA)

    previsoes = gerar_previsoes(df)

    previsoes.to_csv(
        ARQUIVO_SAIDA,
        index=False
    )

    print(f"Previsões geradas com sucesso.")
    print(f"Arquivo salvo em: {ARQUIVO_SAIDA}")
    print(f"Total de clientes: {len(previsoes)}")
    print(f"Clientes classificados como churn: {(previsoes['previsao_churn'] == 'Yes').sum()}")
    print(f"Clientes classificados como não churn: {(previsoes['previsao_churn'] == 'No').sum()}")