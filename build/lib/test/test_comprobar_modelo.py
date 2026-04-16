from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pipeline_pkg import Modelo

def test_comprobar_modelo_returns_predictions():
    data = load_wine()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC()
    model.fit(X_scaled, y)

    preds = Modelo.comprobar_modelo(X_scaled, model)

    assert len(preds) == len(y)
