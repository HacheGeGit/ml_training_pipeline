from pipeline_pkg.pipeline import pipeline_entrenamiento


def test_pipeline_runs():
    models, X_test, y_test = pipeline_entrenamiento()

    assert "svm" in models
    assert len(X_test) == len(y_test)