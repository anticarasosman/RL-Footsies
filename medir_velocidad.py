"""Mide la velocidad de FootsiesGym en modo vectorizado (varias partidas en paralelo).

Uso:
    python medir_velocidad.py              # prueba con 1, 8, 32 y 64 entornos
    python medir_velocidad.py 16 128       # prueba con los números que indiques

Primero muestra cómo vienen las observaciones con varios entornos, porque el
formato puede cambiar respecto al modo de un solo entorno.
"""
import sys
import time
import traceback

import numpy as np

import footsiesgym

PASOS_POR_PRUEBA = 1000
N_ACCIONES = 6


def describir(nombre, x, sangria="  "):
    """Imprime la estructura de un objeto: diccionario, arreglo o valor."""
    if isinstance(x, dict):
        print(f"{sangria}{nombre}: dict con claves {list(x.keys())}")
        for k, v in list(x.items())[:3]:
            describir(str(k), v, sangria + "  ")
    elif isinstance(x, (list, tuple)):
        print(f"{sangria}{nombre}: {type(x).__name__} de largo {len(x)}")
    else:
        print(f"{sangria}{nombre}: {type(x).__name__} con forma {np.shape(x)}")


def acciones_aleatorias(obs, n_envs):
    """Una acción aleatoria por agente y por partida, con la forma que pide el entorno."""
    acciones = {}
    for agente, o in obs.items():
        if n_envs > 1:
            acciones[agente] = np.random.randint(0, N_ACCIONES, size=n_envs)
        else:
            acciones[agente] = int(np.random.randint(0, N_ACCIONES))
    return acciones


def probar(n_envs, mostrar_estructura):
    config = {"num_envs": n_envs} if n_envs > 1 else {}
    env = footsiesgym.make(config=config, platform="linux")
    try:
        obs, infos = env.reset()
        if mostrar_estructura:
            print(f"\nEstructura con num_envs={n_envs}:")
            describir("obs", obs)

        # Calentamiento: el primer paso suele ser más lento
        resultado = env.step(acciones_aleatorias(obs, n_envs))
        if mostrar_estructura:
            _, rew, term, trunc, _ = resultado
            describir("recompensas", rew)
            describir("terminados", term)

        inicio = time.time()
        for _ in range(PASOS_POR_PRUEBA):
            obs, rew, term, trunc, infos = env.step(acciones_aleatorias(obs, n_envs))
        duracion = time.time() - inicio

        pasos_totales = PASOS_POR_PRUEBA * n_envs
        print(f"num_envs={n_envs:>4}: {pasos_totales / duracion:>10,.0f} pasos/s")
    finally:
        env.close()


def main():
    tamanos = [int(a) for a in sys.argv[1:]] or [1, 8, 32, 64]
    for i, n in enumerate(tamanos):
        try:
            probar(n, mostrar_estructura=(n > 1 and i == next(j for j, m in enumerate(tamanos) if m > 1)))
        except Exception:
            print(f"\nnum_envs={n}: falló con este error:")
            traceback.print_exc()
            print("Pégale este error a Claude.\n")


if __name__ == "__main__":
    main()
