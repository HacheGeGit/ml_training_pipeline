from sklearn.datasets import load_wine
from sklearn.svm import SVC
from pipeline_pkg.pipeline import Operaciones


def test_svm_prediccion():

    data = load_wine()
    X = data.data
    y = data.target

    X_scaled, _ = Operaciones.instanciar_escalas(X)

    model = SVC()
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)

    assert len(preds) == len(y)