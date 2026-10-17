"""Pruebas rápidas del entorno, la red y la recolección de datos.

Uso:
    python pruebas.py

Si el entrenamiento se comporta raro, córrelo para descartar que el problema
esté en el entorno, en la red o en la recolección.
"""
import time
import traceback

import numpy as np
import torch

from entrenar import (
    NUM_ENVS,
    acciones_aleatorias,
    calcular_ventajas,
    crear_entorno,
    crear_estado,
    recolectar,
)

PASOS = 1000


def probar_red(estado):
    """Comprueba las formas de salida de la red y la entropía inicial."""
    print("\n--- Prueba de la red ---")
    with torch.no_grad():
        accion, log_prob, entropia, valor = estado.red.obtener_accion_y_valor(estado.next_obs)

    print("Forma acción:  ", accion.shape)
    print("Forma log_prob:", log_prob.shape)
    print("Forma entropía:", entropia.shape)
    print("Forma valor:   ", valor.shape)
    print("Acciones:", accion)
    print("Entropía promedio:", entropia.mean().item())


def probar_entorno(env):
    """Juega con acciones aleatorias y cuenta episodios y victorias."""
    print("\n--- Prueba del entorno ---")
    episodios = np.zeros(NUM_ENVS, dtype=int)
    victorias_p1 = np.zeros(NUM_ENVS, dtype=int)

    inicio = time.time()
    for _ in range(PASOS):
        obs, rew, term, trunc, infos = env.step(acciones_aleatorias())
        terminadas = term["p1"] | trunc["p1"]
        episodios += terminadas
        victorias_p1 += rew["p1"] > 0
    duracion = time.time() - inicio

    print(f"Episodios por partida: {episodios}")
    print(f"Mínimo: {episodios.min()}   Máximo: {episodios.max()}   Total: {episodios.sum()}")
    print(f"Win rate de p1: {victorias_p1.sum() / episodios.sum():.1%}")
    print(f"Velocidad: {PASOS * NUM_ENVS / duracion:,.0f} pasos/s")


def probar_recoleccion(estado):
    """Recolecta datos con la red, calcula las ventajas y revisa que tengan sentido."""
    print("\n--- Prueba de la recolección y las ventajas ---")
    datos = recolectar(estado)
    calcular_ventajas(estado, datos)

    for nombre, tabla in datos.items():
        print(nombre, tabla.shape)

    ventajas = datos["advantages"]
    recompensas = datos["rewards"]
    print("Recompensas distintas de 0:", (recompensas != 0).sum().item())
    print("Suma de recompensas:", recompensas.sum().item())
    print("Dones en 1:", datos["dones"].sum().item())
    print("Acciones mín/máx:", datos["acciones"].min().item(), datos["acciones"].max().item())
    print("Ventajas válidas:", torch.isfinite(ventajas).all().item())
    print("Ventajas en victorias:", ventajas[recompensas > 0])
    print("Ventajas en derrotas:", ventajas[recompensas < 0])


def main():
    env = crear_entorno()
    try:
        print("Acciones:", env.action_space("p1"))

        estado = crear_estado(env)
        probar_red(estado)
        probar_recoleccion(estado)
        probar_entorno(env)
    except Exception:
        print("Falló con este error:")
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
