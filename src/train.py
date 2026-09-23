import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess


def train_model(df):

    x_train, x_teste, y_train, y_teste, preprocessor = preprocess(
        df,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000
                )
            )
        ]
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_teste)

    return model, y_teste, y_pred


if __name__ == "__main__":

    df = pd.read_csv("../data/processed/churn_modelagem.csv")

    model, y_teste, y_pred = train_model(df)