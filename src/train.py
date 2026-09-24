import pandas as pd

from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess


def train_model(df, t_size=0.2):

    x_train, x_teste, y_train, y_teste, preprocessor = preprocess(
        df,
        t_size=t_size
    )

    x_train_normalized = preprocessor.fit_transform(x_train)

    x_teste_normalized = preprocessor.transform(x_teste)

    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    )

    model.fit(
        x_train_normalized,
        y_train
    )

    y_pred = model.predict(x_teste_normalized)

    return model, y_teste, y_pred


if __name__ == "__main__":

    df = pd.read_csv("../data/processed/churn_modelagem.csv")

    model, y_teste, y_pred = train_model(df)