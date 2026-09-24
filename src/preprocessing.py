import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess(df, t_size=0.2, target='Churn'):
    
    ## Sepração dos atributos  que queremos em x e y
    x = df.drop(columns=[target])
    y = df[target]

    ## Separação do dataset em treino e teste na proporção 80:20
    x_train, x_teste, y_train, y_teste = train_test_split(
        x,
        y,
        test_size=t_size,
        random_state=42,
        stratify=y
    )

    # Identificação das colunas numéricas e categóricas
    colunas_numericas = x.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    colunas_categoricas = x.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), colunas_numericas),
            ('cat', OneHotEncoder(handle_unknown="ignore"), colunas_categoricas)
        ]
    )
    
    return x_train,x_teste,y_train,y_teste,preprocess

def create_preprocessor(x):

    colunas_numericas = x.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    colunas_categoricas = x.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                colunas_numericas
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                colunas_categoricas
            )
        ]
    )

    return preprocessor