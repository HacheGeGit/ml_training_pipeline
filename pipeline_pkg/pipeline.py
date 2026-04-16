from .modelo import Modelo
from .operaciones import Operaciones
from .datos import CargarDatos
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .decoradores import time, pasar_a_txt
from .explorador import seleccionar_modelo
from datetime import datetime
import logging
import os

os.makedirs("logs/errores", exist_ok=True)

logging.basicConfig(
    filename="logs/errores/app_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def pipeline_entrenamiento():
    modelo_wine = Modelo("Modelo Wine","./skopsModels/VectorMachine.skops") 
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.makedirs("logs", exist_ok=True)

    with open("logs/performance_log.txt", "a") as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"INICIO EJECUCIÓN: {now} Modelo Wine\n")
        f.write("="*50 + "\n\n")
    t0 = time.perf_counter()

    wine_data = CargarDatos.cargar_datos()
    wine_df = CargarDatos.carga_df(wine_data)

    wine_df["target"] = wine_data.target

    t1 = time.perf_counter()

    X, y = Operaciones.procesamiento_datos(wine_df, wine_data)

    X_scaled, scaler = Operaciones.instanciar_escalas(X)

    t2 = time.perf_counter()

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
    print("Registrando predicciones:")
    modelo_wine.resgistrar_svm_preds(pred_svm, y_test)
    CargarDatos.registrar_predicciones("Logistic Regression", y_test, pred_logreg)
    CargarDatos.registrar_predicciones("Support Vector Machine", y_test, pred_svm)
    CargarDatos.registrar_predicciones("Decision Tree", y_test, pred_tree)

    t3 = time.perf_counter()

    print(f"Carga datos: {t1 - t0:.4f}s")
    print(f"Preprocesado: {t2 - t1:.4f}s")
    print(f"Entrenamiento: {t3 - t2:.4f}s")

    duracion_total = t3 - t0

    resumen = (
    f"[{now}]Tiempo de realización de la tarea cargar_datos: {t1 - t0:.6f} segundos.\n"
    f"[{now}]Tiempo de realización de la tarea procesamiento_datos: {t2 - t1:.6f} segundos.\n"
    f"[{now}]Tiempo de realización de la tarea entrenar_modelos: {t3 - t2:.6f} segundos.\n"
    f"[{now}]Tiempo de realización de la tarea pipeline_total: {duracion_total:.6f} segundos.\n"
)

    print(resumen)

    pasar_a_txt(resumen)

    return {
        "logreg": logreg,
        "svm": svm,
        "tree": tree,
        "scaler": scaler
    }, X_test, y_test


def menu():
    modelo_cargado = None
    modelos_entrenados = None
    X_test = None
    y_test = None
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
                ruta = seleccionar_modelo()
                
                vector_types = Modelo.comprobar_tipo_fichero(file=ruta)
                modelo_cargado = Modelo.cargar_modelo(ruta, vector_types)
                print("Modelo cargado correctamente")

            case "3":

                if modelos_entrenados is None and modelo_cargado is None:
                    print("No hay modelos disponibles")
                    continue

                
                if modelo_cargado and X_test is None:
                    wine_data = CargarDatos.cargar_datos()
                    wine_df = CargarDatos.carga_df(wine_data)
                    wine_df["target"] = wine_data.target

                    X, y = Operaciones.procesamiento_datos(wine_df, wine_data)
                    X_scaled, scaler = Operaciones.instanciar_escalas(X)

                    _, X_test, _, y_test = train_test_split(
                        X_scaled, y, train_size=0.7, random_state=25
                    )

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
    menu()


if __name__ == "__main__":
    main()


