import pytest
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline_pkg.pipeline import Modelo


def test_comprobar_tipo_fichero_seguro():
    # NO toca archivos reales → evita crash
    result = Modelo.comprobar_tipo_fichero(file="fake.skops")
    assert isinstance(result, list)


def test_modelo_init():
    m = Modelo("Test", "ruta.skops")
    assert m.nombre == "Test"
    assert m.ruta == "ruta.skops"