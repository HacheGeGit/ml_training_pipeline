import time
from functools import wraps
from datetime import datetime

def controlTiempo(fn_base):
    @wraps(fn_base)
    def inicio(*args, **kwargs):
        start_time = time.time()
        resultado = fn_base(*args, **kwargs)
        end_time = time.time()
        cadena = f'Tiempo de realización de la tarea {fn_base.__name__}: {end_time - start_time:.6f} segundos.\n'
        print(cadena)
        pasar_a_txt(cadena)
        return resultado
    return inicio

def pasar_a_txt(text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('logs/performance_log.txt', 'a') as f:
        f.write(f'[{now}]{text}')