from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def test_wine_dataset_loads():
    data = load_wine()
    assert data is not None
    assert len(data.data) > 0

def test_scaler_preserves_shape():
    data = load_wine()
    X = data.data

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X.shape == X_scaled.shape

def test_svm_training_and_prediction():
    data = load_wine()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    model = SVC()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    assert len(preds) == len(y_test)
