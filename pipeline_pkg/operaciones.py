from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from .decoradores import controlTiempo

class Operaciones:

    @staticmethod
    @controlTiempo
    def procesamiento_datos(wine_df, wine_data):
        X = wine_df[wine_data.feature_names].copy()
        y = wine_df["target"].copy()
        return X, y

    @controlTiempo
    @staticmethod
    def instanciar_escalas(X):
        scaler = StandardScaler()

        if hasattr(X, "values"):
            X = X.values

        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    @staticmethod
    @controlTiempo
    def instanciar_modelos():
        return (
            LogisticRegression(),
            SVC(),
            DecisionTreeClassifier()
        )

    @staticmethod
    @controlTiempo
    def entrenar_modelos(logreg, svm, tree, X_train, y_train):
        logreg.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        tree.fit(X_train, y_train)
        return logreg, svm, tree