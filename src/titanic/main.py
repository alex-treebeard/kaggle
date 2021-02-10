import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy="constant")

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


def run():
    DATA_DIR = "data"
    OUTPUT_DIR = "output"

    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    te = pd.read_csv(f"{DATA_DIR}/test.csv")

    numerical_fields = ["Pclass", "Age", "SibSp", "Parch"]
    categorical_fields = ["Sex"]
    fields = [*numerical_fields, *categorical_fields]

    X = tr[fields]
    y = tr[["Survived"]].to_numpy().reshape((len(tr),))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_fields),
            ("cat", categorical_transformer, categorical_fields),
        ]
    )

    my_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(max_depth=1, random_state=0)),
        ]
    )
    scores = -1 * cross_val_score(
        my_pipeline, X, y, cv=5, scoring="neg_mean_absolute_error"
    )

    print("Average score (across experiments):")
    print(1 - scores.mean())

    my_pipeline.fit(X, y)

    y_p = my_pipeline.predict(te[fields])

    out = pd.DataFrame({"PassengerId": te["PassengerId"], "Survived": y_p})
    out.to_csv(f"{OUTPUT_DIR}/out.csv", index=False)


if __name__ == "__main__":
    run()
