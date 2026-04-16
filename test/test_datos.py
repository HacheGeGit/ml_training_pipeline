from sklearn.datasets import load_wine
from pipeline_pkg.pipeline import CargarDatos


def test_load_wine():
    data = CargarDatos.cargar_datos()
    assert data is not None


def test_df_creation():
    data = load_wine()
    df = CargarDatos.carga_df(data)

    assert len(df.columns) == len(data.feature_names)