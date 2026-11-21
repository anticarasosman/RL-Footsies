"""Entrena un agente de Footsies.

Ejemplos:
    python entrenar.py                                      # PPO con self-play, semilla 1
    python entrenar.py --modo vs_aleatorio --iteraciones 300
    python entrenar.py --semilla 3 --nombre ppo_selfplay_s3
    python entrenar.py --modo liga                          # self-play + versiones antiguas
    python entrenar.py --algoritmo dreamer --modo liga --iteraciones 300
    python entrenar.py --modo liga --liga-cada 20 --liga-prob-actual 0.3
    python entrenar.py --rivales-eval aleatorio agentes_entrenados/PPOambosLadosVSRandom.pt

Ver todas las opciones:  python entrenar.py --help
"""
import argparse
import os
import time
import traceback

import numpy as np
import torch

from algoritmos import ALGORITMOS
from comun.entorno import NUM_ENVS, crear_entorno
from comun.estado import Estado
from comun.evaluacion import evaluar, linea_resumen
from comun.jugadores import cargar_jugador
from comun.modelos import guardar_modelo
from comun.modos import MODOS
from comun.registro import Registro

CARPETA_MODELOS = "agentes_entrenados"
RIVALES_EVAL = ["aleatorio", os.path.join(CARPETA_MODELOS, "PPOambosLadosVSRandom.pt")]


# ── Configuración ─────────────────────────────────────────────────────


def leer_argumentos():
    p = argparse.ArgumentParser(description="Entrenar un agente de Footsies.")
    p.add_argument("--algoritmo", choices=ALGORITMOS, default="ppo")
    p.add_argument("--modo", choices=MODOS, default="selfplay",
                   help="contra quién entrena el aprendiz")
    p.add_argument("--iteraciones", type=int, default=500)
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--nombre", default=None,
                   help="nombre del modelo (por defecto: algoritmo_modo_sSEMILLA)")
    p.add_argument("--carpeta", default=CARPETA_MODELOS, help="dónde guardar el modelo")
    p.add_argument("--num-envs", type=int, default=NUM_ENVS, help="partidas en paralelo")
    p.add_argument("--imprimir-cada", type=int, default=10)
    p.add_argument("--eval-cada", type=int, default=25,
                   help="cada cuántas iteraciones evaluar (0 = nunca)")
    p.add_argument("--eval-pasos", type=int, default=256)
    p.add_argument("--rivales-eval", nargs="*", default=RIVALES_EVAL,
                   help="'aleatorio' y/o rutas a .pt contra los que evaluar")
    p.add_argument("--hilos", type=int, default=None,
                   help="hilos de CPU para PyTorch (por defecto: unos pocos, para no "
                        "competir con el juego)")
    p.add_argument("--liga-cada", type=int, default=10,
                   help="modo liga: cada cuántas iteraciones guardar una foto del aprendiz")
    p.add_argument("--liga-prob-actual", type=float, default=0.5,
                   help="modo liga: probabilidad de jugar contra la copia actual "
                        "(el resto, contra una foto antigua)")
    args = p.parse_args()

    if args.nombre is None:
        args.nombre = f"{args.algoritmo}_{args.modo}_s{args.semilla}"
    return args


def hilos_por_defecto():
    """Si PyTorch usa todos los núcleos compite con el juego y todo se vuelve muy lento."""
    return min(4, max(1, (os.cpu_count() or 2) // 2 - 1))


def fijar_semilla(semilla):
    np.random.seed(semilla)
    torch.manual_seed(semilla)


# ── Evaluación durante el entrenamiento ───────────────────────────────


def cargar_rivales_eval(especificaciones, n_acciones):
    """Carga los rivales fijos de evaluación. Omite (con aviso) los que no existen."""
    rivales = []
    for especificacion in especificaciones:
        if especificacion != "aleatorio" and not os.path.exists(especificacion):
            print(f"Aviso: no existe {especificacion}; no se usará para evaluar.")
            continue
        rival = cargar_jugador(especificacion, n_acciones)
        if rival.lados == "p1":
            print(f"Aviso: {rival.nombre} solo sabe jugar como p1; "
                  f"sus resultados como p2 no son representativos.")
        rivales.append(rival)
    return rivales


def evaluar_contra_rivales(algoritmo, rivales, env_eval, pasos, iteracion):
    aprendiz = algoritmo.como_jugador()
    for rival in rivales:
        resumen = evaluar(aprendiz, rival, env_eval, pasos)
        print(f"   [eval it {iteracion:4d}] vs {rival.nombre:24s} | {linea_resumen(resumen)}")


# ── Progreso del entrenamiento ────────────────────────────────────────


def imprimir_progreso(iteracion, registro, metricas, inicio, modo, algoritmo):
    minutos = (time.time() - inicio) / 60
    # Cada algoritmo dice qué métricas mostrar (Dreamer no tiene las mismas que PPO)
    texto_metricas = "".join(f"{clave} {metricas[clave]:.4f} | "
                             for clave in algoritmo.imprimir if clave in metricas)
    print(f"Iteración {iteracion:4d} | vs rival de entrenamiento: "
          f"{linea_resumen(registro.resumen())} | {texto_metricas}"
          f"{algoritmo.pasos_entorno / 1e6:.2f}M pasos | {minutos:.1f} min")
    if hasattr(modo, "resumen_texto"):
        print(f"               {modo.resumen_texto()}")


# ── Entrenamiento ─────────────────────────────────────────────────────


def entrenar(args, env, env_eval, algoritmo, estado):
    """El loop principal: en cada iteración elige rival, entrena y registra."""
    modo = MODOS[args.modo](algoritmo, estado.n_acciones, args)
    registro = Registro(estado.lados)
    rivales = cargar_rivales_eval(args.rivales_eval, estado.n_acciones) if env_eval else []
    evaluar_ahora = bool(rivales) and args.eval_cada > 0

    if evaluar_ahora:
        evaluar_contra_rivales(algoritmo, rivales, env_eval, args.eval_pasos, 0)

    inicio = time.time()
    for iteracion in range(1, args.iteraciones + 1):
        rival = modo.rival_para_iteracion(algoritmo, iteracion)
        datos, metricas = algoritmo.iteracion(estado, rival)
        registro.agregar(datos)

        if iteracion % args.imprimir_cada == 0:
            imprimir_progreso(iteracion, registro, metricas, inicio, modo, algoritmo)
            registro.reiniciar()

        if evaluar_ahora and iteracion % args.eval_cada == 0:
            evaluar_contra_rivales(algoritmo, rivales, env_eval, args.eval_pasos, iteracion)


def guardar(args, algoritmo, iteraciones, sufijo=""):
    ruta = os.path.join(args.carpeta, f"{args.nombre}{sufijo}.pt")
    guardar_modelo(ruta, algoritmo.red, **algoritmo.metadatos(), modo=args.modo,
                   lados="ambos", iteraciones=iteraciones, semilla=args.semilla)
    print(f"Modelo guardado en {ruta}")


# ── Programa principal ────────────────────────────────────────────────


def main():
    args = leer_argumentos()
    fijar_semilla(args.semilla)
    torch.set_num_threads(args.hilos or hilos_por_defecto())
    print(f"Entrenando {args.algoritmo} en modo {args.modo}, semilla {args.semilla}, "
          f"{args.iteraciones} iteraciones → {args.nombre}")

    env = crear_entorno(args.num_envs)
    env_eval = crear_entorno(args.num_envs) if args.eval_cada > 0 else None
    algoritmo = None
    try:
        estado = Estado(env)
        algoritmo = ALGORITMOS[args.algoritmo](estado.n_obs, estado.n_acciones, estado.num_envs)
        entrenar(args, env, env_eval, algoritmo, estado)
        guardar(args, algoritmo, args.iteraciones)
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido.")
        if algoritmo is not None:
            guardar(args, algoritmo, iteraciones=None, sufijo="_interrumpido")
    except Exception:
        print("Falló con este error:")
        traceback.print_exc()
    finally:
        env.close()
        if env_eval is not None:
            env_eval.close()


if __name__ == "__main__":
    main()
