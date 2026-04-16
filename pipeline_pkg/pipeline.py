import pandas as pd
from .modelo import Modelo
from .operaciones import Operaciones
from .datos import CargarDatos
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import skops.io as sio
from .decoradores import time, pasar_a_txt
import logging
import os

os.makedirs("logs/errores", exist_ok=True)

logging.basicConfig(
    filename="logs/errores/app_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def pipeline_entrenamiento():

    t0 = time.time()

    wine_data = CargarDatos.cargar_datos()
    wine_df = CargarDatos.carga_df(wine_data)

    wine_df["target"] = wine_data.target

    t1 = time.time()

    X, y = Operaciones.procesamiento_datos(wine_df, wine_data)

    X_scaled, scaler = Operaciones.instanciar_escalas(X)

    t2 = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=0.7, random_state=25
    )

    logreg, svm, tree = Operaciones.instanciar_modelos()

    logreg, svm, tree = Operaciones.entrenar_modelos(
        logreg, svm, tree,
        X_train, y_train
    )

    pred_logreg = logreg.predict(X_test)
    pred_svm = svm.predict(X_test)
    pred_tree = tree.predict(X_test)

    CargarDatos.registrar_predicciones("Logistic Regression", y_test, pred_logreg)
    CargarDatos.registrar_predicciones("Support Vector Machine", y_test, pred_svm)
    CargarDatos.registrar_predicciones("Decision Tree", y_test, pred_tree)

    t3 = time.time()

    print(f"Carga datos: {t1 - t0:.4f}s")
    print(f"Preprocesado: {t2 - t1:.4f}s")
    print(f"Entrenamiento: {t3 - t2:.4f}s")

    pasar_a_txt(f"""
Carga datos: {t1 - t0:.4f}s
Preprocesado: {t2 - t1:.4f}s
Entrenamiento: {t3 - t2:.4f}s
""")

    return {
        "logreg": logreg,
        "svm": svm,
        "tree": tree
    }, X_test, y_test

def menu():
    while True:

        print("\n1. Entrenar modelos")
        print("2. Cargar modelo entrenado")
        print("3. Evaluar modelo")
        print("4. Salir")

        opcion = input("\nSelecciona opción: ")

        match opcion:

            case "1":
                print("Entrenando modelos...")
                modelos_entrenados, X_test, y_test = pipeline_entrenamiento()
                print("Entrenamiento completado")

            case "2":
                ruta = input("Introduce ruta del modelo: ")
                vector_types = Modelo.comprobar_tipo_fichero(file=ruta)
                modelo_cargado = Modelo.cargar_modelo(ruta, vector_types)
                print("Modelo cargado correctamente")

            case "3":

                if modelos_entrenados is None and modelo_cargado is None:
                    print("No hay modelos disponibles")
                    continue

                print("Elige modelo: (1)logreg / (2)svm / (3)tree")
                try:
                    num = int(input("Modelo: "))
                    match num:
                        case 1:
                            nombre = 'logreg'
                        case 2:
                            nombre = 'svm'
                        case 3: 
                            nombre = 'tree'
                        case _:
                            print('Elección inválida.')
                except ValueError:
                    print("Debes introducir un número")

                if modelos_entrenados and nombre in modelos_entrenados:
                    modelo = modelos_entrenados[nombre]
                    preds = modelo.predict(X_test)
                    print(classification_report(y_test, preds))

                elif modelo_cargado:
                    preds = modelo_cargado.predict(X_test)
                    print(classification_report(y_test, preds))

                else:
                    print("Modelo no encontrado")

            case "4":
                print("Saliendo...")
                break

            case _:
                print("Opción no válida")

def main():
    modelo_cargado = None
    modelos_entrenados = None
    X_test = None
    y_test = None
    menu()


if __name__ == "__main__":
    main()


