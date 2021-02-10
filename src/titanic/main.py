import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

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

    estimator = XGBClassifier(
        learning_rate=0.05, use_label_encoder=False, eval_metric="logloss"
    )
    # estimator = RandomForestClassifier(max_depth=1, random_state=0)

    my_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                estimator,
            ),
        ]
    )

    param_grid = {"model__n_estimators": [250, 500, 1000]}
    s = GridSearchCV(
        my_pipeline, param_grid, cv=5, scoring="neg_mean_absolute_error"
    ).fit(X, y)

    print("Best score:")
    print(1 - (-1 * s.best_score_))
    print(f"Best params:\n{s.best_params_}")

    s.best_estimator_.fit(X, y)

    y_p = s.best_estimator_.predict(te[fields])

    out = pd.DataFrame({"PassengerId": te["PassengerId"], "Survived": y_p})
    out.to_csv(f"{OUTPUT_DIR}/out.csv", index=False)


if __name__ == "__main__":
    run()
