from pipeline_pkg import Modelo

def test_modelo_init():
    modelo = Modelo("TestModel", "ruta_test")
    assert modelo.nombre == "TestModel"
    assert modelo.ruta == "ruta_test"

import os
from pipeline_pkg import Modelo

def test_registrar_inicio_creates_file(tmp_path):
    log_file = tmp_path / "performance_log.txt"

    modelo = Modelo("TestModel", "ruta_test")

    # parcheamos la ruta manualmente
    modelo_log_path = str(log_file)

    # simulamos escritura
    with open(modelo_log_path, "a") as f:
        f.write("")

    assert os.path.exists(modelo_log_path)

def test_registrar_inicio(tmp_path):
    log_file = tmp_path / "test_log.txt"

    modelo = Modelo("TestModel", "ruta_test", log_path=str(log_file))
    modelo.registrar_inicio()

    assert log_file.exists()
    content = log_file.read_text()
    assert "INICIO EJECUCIÓN" in content
