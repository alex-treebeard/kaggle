import pandas as pd


def run():
    DATA_DIR = "data"
    OUTPUT_DIR = "output"

    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    te = pd.read_csv(f"{DATA_DIR}/test.csv")

    fields = [
        "Pclass",
        #     'Age',
        #     'Fare'
    ]

    from sklearn.impute import SimpleImputer

    my_imputer = SimpleImputer()

    def get_input(dfx):
        s = pd.get_dummies(dfx[["Sex"]])
        onehot = pd.concat([dfx[fields], s], axis=1)
        return my_imputer.fit_transform(onehot)

    X = get_input(tr)
    y = tr[["Survived"]]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(max_depth=1, random_state=0)
    clf = clf.fit(X_train, y_train)

    from sklearn.metrics import mean_absolute_error

    y_pred = clf.predict(X_test)
    1 - mean_absolute_error(y_test, y_pred)

    clf_final = RandomForestClassifier(max_depth=1, random_state=0)
    clf_final.fit(X, y)

    y_p = clf_final.predict(get_input(te))

    out = pd.DataFrame({"PassengerId": te["PassengerId"], "Survived": y_p})

    out.to_csv(f"{OUTPUT_DIR}/suby.csv", index=False)


if __name__ == "__main__":
    run()
