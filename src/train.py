import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess

## Treinamento modelo de regressão logistica
def train_logistic_regression(df, t_size=0.2):

    x_train, x_teste, y_train, y_teste, preprocessor = preprocess(
        df,
        t_size=t_size
    )

    x_train_transformed = preprocessor.fit_transform(x_train)

    x_teste_transformed = preprocessor.transform(x_teste)

    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    )

    model.fit(
        x_train_transformed,
        y_train
    )

    y_pred = model.predict(x_teste_transformed)
    
    y_prob = model.predict_proba(
    x_teste_transformed
    )[:, 1]

    return model, preprocessor, y_teste, y_pred, y_prob

## Treinamento Random forest
def train_random_forest(df, t_size=0.2):

    x_train, x_teste, y_train, y_teste, preprocessor = preprocess(
        df,
        t_size=t_size
    )

    x_train_transformado = preprocessor.fit_transform(x_train)

    x_teste_transformado = preprocessor.transform(x_teste)

    model = RandomForestClassifier(
        class_weight="balanced",
        max_depth=10,
        random_state=42
    )

    model.fit(
        x_train_transformado,
        y_train
    )

    y_pred = model.predict(x_teste_transformado)

    y_prob = model.predict_proba(
        x_teste_transformado
    )[:, 1]

    return model, preprocessor, y_teste, y_pred, y_prob


if __name__ == "__main__":

    df = pd.read_csv("data/processed/churn_modelagem.csv")

    train_logistic_regression(df)
    train_random_forest(df)