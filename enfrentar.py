"""Enfrenta a dos jugadores y muestra quién gana.

Cada jugador puede ser 'aleatorio' o la ruta a un modelo .pt. Se juega en muchas
partidas a la vez, la mitad con A como p1 y la otra mitad con A como p2.

Ejemplos:
    python enfrentar.py agentes_entrenados/ppo_selfplay_s1.pt aleatorio
    python enfrentar.py agentes_entrenados/ppo_selfplay_s1.pt agentes_entrenados/PPOambosLadosVSRandom.pt
    python enfrentar.py A.pt B.pt --pasos 1024 --determinista
"""
import argparse
import traceback

import numpy as np
import torch

from comun.entorno import NUM_ENVS, crear_entorno
from comun.evaluacion import evaluar
from comun.jugadores import cargar_jugador


def leer_argumentos():
    p = argparse.ArgumentParser(description="Enfrentar dos jugadores de Footsies.")
    p.add_argument("jugador_a", help="'aleatorio' o ruta a un .pt")
    p.add_argument("jugador_b", help="'aleatorio' o ruta a un .pt")
    p.add_argument("--pasos", type=int, default=512,
                   help="pasos a jugar (más pasos = resultado más confiable)")
    p.add_argument("--num-envs", type=int, default=NUM_ENVS)
    p.add_argument("--determinista", action="store_true",
                   help="las redes eligen siempre su acción más probable")
    p.add_argument("--semilla", type=int, default=0)
    return p.parse_args()


def avisos_de_lados(jugador):
    if jugador.lados == "p1":
        print(f"Aviso: {jugador.nombre} solo sabe jugar como p1; "
              f"sus resultados como p2 no son representativos.")
    elif jugador.lados == "desconocido":
        print(f"Nota: {jugador.nombre} está en formato antiguo y no se sabe de qué lados "
              f"aprendió a jugar. Si se entrenó solo como p1, mira solo su columna 'como p1'.")


def imprimir_tabla(a, b, resumen):
    print()
    print(f"{'':24s} | {'como p1':>9s} | {'como p2':>9s} | {'total':>9s}")
    print("-" * 61)
    for etiqueta, jugador, de_a in ((a.nombre, a, True), (b.nombre, b, False)):
        fila = []
        # Cuando A es p1, B es p2 (y viceversa): la columna de B se cruza.
        for lado_a in (("p1", "p2") if de_a else ("p2", "p1")):
            r = resumen[lado_a]
            ganadas = r["victorias"] if de_a else r["derrotas"]
            fila.append(ganadas / max(r["victorias"] + r["derrotas"], 1))
        t = resumen["total"]
        total = (t["victorias"] if de_a else t["derrotas"]) / max(t["victorias"] + t["derrotas"], 1)
        print(f"{etiqueta[:24]:24s} | {fila[0]:9.1%} | {fila[1]:9.1%} | {total:9.1%}")

    t = resumen["total"]
    partidas = t["victorias"] + t["derrotas"] + t["empates"]
    print("-" * 61)
    print(f"{partidas} partidas: {a.nombre} ganó {t['victorias']}, "
          f"{b.nombre} ganó {t['derrotas']}, {t['empates']} empates por tiempo.")


def main():
    args = leer_argumentos()
    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)

    env = crear_entorno(args.num_envs)
    try:
        n_acciones = env.action_space("p1").n
        a = cargar_jugador(args.jugador_a, n_acciones, args.determinista)
        b = cargar_jugador(args.jugador_b, n_acciones, args.determinista)
        if a.nombre == b.nombre:
            b.nombre += " (B)"
        for jugador in (a, b):
            avisos_de_lados(jugador)

        print(f"\n{a.nombre} vs {b.nombre}: {args.pasos} pasos en {args.num_envs} partidas...")
        resumen = evaluar(a, b, env, args.pasos)
        imprimir_tabla(a, b, resumen)
    except Exception:
        print("Falló con este error:")
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
