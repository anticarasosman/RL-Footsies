"""Muestra al agente entrenado jugando contra un rival aleatorio, en la ventana del juego.

Funciona en macOS (donde la versión con ventana de FootsiesGym sí tiene gráficos)
y en Linux (donde por ahora el juego corre, pero la ventana no se ve).

Uso:
    python ver_agente.py                      # carga agentes_entrenados/ppo_selfplay_s1.pt, muestrea acciones
    python ver_agente.py --determinista       # elige siempre la acción más probable
    python ver_agente.py --velocidad 2        # el doble de rápido
    python ver_agente.py --partidas 5         # se detiene después de 5 partidas
    python ver_agente.py --modelo otro.pt     # carga otro archivo de pesos

Para salir antes, presiona Ctrl + C en la terminal.
"""
import argparse
import os
import platform
import subprocess
import time

import numpy as np
import portpicker
import torch

import footsiesgym
from footsiesgym.binary_manager import get_binary_manager

from comun.jugadores import cargar_jugador

# Cada paso son 4 cuadros del juego, que corre a 60 cuadros por segundo.
SEGUNDOS_POR_PASO = 4 / 60


def leer_argumentos():
    parser = argparse.ArgumentParser(description="Ver al agente entrenado jugar.")
    parser.add_argument("--modelo", default="agentes_entrenados/ppo_selfplay_s1.pt", help="Archivo con los pesos de la red.")
    parser.add_argument("--determinista", action="store_true",
                        help="Elegir siempre la acción más probable en vez de muestrear.")
    parser.add_argument("--velocidad", type=float, default=1.0,
                        help="Multiplicador de velocidad (1 = tiempo real).")
    parser.add_argument("--partidas", type=int, default=0,
                        help="Cuántas partidas jugar (0 = sin límite).")
    return parser.parse_args()


def detectar_plataforma():
    sistema = platform.system()
    if sistema == "Darwin":
        return "mac"
    if sistema == "Linux":
        return "linux"
    raise RuntimeError("FootsiesGym solo funciona en macOS o Linux (en Windows, usa WSL).")


def ruta_juego_con_ventana(plataforma):
    """Descarga (si hace falta) la versión con ventana y devuelve la ruta del ejecutable."""
    gestor = get_binary_manager()
    carpeta_binarios = os.path.join(os.path.dirname(footsiesgym.__file__), "binaries")
    carpeta_juego = os.path.join(carpeta_binarios, "footsies_binaries_windowed")
    ruta = os.path.join(carpeta_juego, gestor.executable_relpath(plataforma, headless=False))

    if not os.path.exists(ruta):
        print("Descargando la versión del juego con ventana (solo la primera vez)...")
        if not gestor.ensure_binaries_extracted(plataforma, target_dir=carpeta_binarios,
                                                headless=False):
            raise FileNotFoundError("No se pudo descargar el juego. Revisa la conexión a internet.")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el juego en {ruta}.")
    os.chmod(ruta, 0o755)
    return ruta, carpeta_juego


def lanzar_juego_con_ventana(plataforma):
    """Lanza el juego con ventana y devuelve (proceso, puerto).

    Se lanza a mano porque FootsiesGym agrega la opción -batchmode, que en Unity
    impide mostrar la ventana.
    """
    ruta, carpeta_juego = ruta_juego_con_ventana(plataforma)
    puerto = portpicker.pick_unused_port()

    if plataforma == "mac":
        # Firma local para que macOS permita abrir el juego descargado.
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", carpeta_juego],
                       check=False, capture_output=True)
        # El juego usa una librería que solo existe para Intel: corre con Rosetta.
        comando = ["arch", "-x86_64", ruta, "--grpc", "--port", str(puerto)]
    else:
        comando = [ruta, "--grpc", "--port", str(puerto),
                   "-screen-fullscreen", "0", "-screen-width", "1280", "-screen-height", "720"]

    print("Lanzando el juego:", " ".join(comando))
    proceso = subprocess.Popen(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)  # darle tiempo al juego para abrir la ventana y el servidor
    if proceso.poll() is not None:
        raise RuntimeError(
            "El juego se cerró apenas se abrió. En un Mac con chip Apple, revisa que "
            "Rosetta esté instalado (ver LEEME.md)."
        )
    return proceso, puerto


def crear_entorno(plataforma, puerto):
    """Una sola partida, con los especiales activados, conectada al juego ya lanzado."""
    config = {"headless": False, "use_special_charge_action": True, "port": puerto}
    return footsiesgym.make(config=config, platform=plataforma, launch_binaries=False)


def elegir_accion(jugador, obs_p1, nueva):
    """Devuelve la acción de p1 como un entero entre 0 y 8.

    `nueva` es True en la primera observación de cada partida (para jugadores con memoria).
    """
    obs_tensor = torch.tensor(obs_p1, dtype=torch.float32).unsqueeze(0)  # (1, 88)
    return int(jugador.actuar(obs_tensor, np.array([nueva]))[0])


def main():
    args = leer_argumentos()
    plataforma = detectar_plataforma()
    proceso, puerto = lanzar_juego_con_ventana(plataforma)
    env = None
    try:
        env = crear_entorno(plataforma, puerto)
        obs, infos = env.reset()
        n_obs = obs["p1"].shape[0]
        n_acciones = env.action_space("p1").n
        jugador = cargar_jugador(args.modelo, n_acciones, args.determinista)
        nueva = True

        modo = "determinista" if args.determinista else "muestreando"
        print(f"Modelo {args.modelo} cargado ({modo}). Ctrl + C para salir.\n")

        victorias = 0
        partidas = 0
        pausa = SEGUNDOS_POR_PASO / args.velocidad

        while args.partidas == 0 or partidas < args.partidas:
            acciones = {
                "p1": elegir_accion(jugador, obs["p1"], nueva),
                "p2": int(np.random.randint(0, n_acciones)),
            }
            obs, rew, term, trunc, infos = env.step(acciones)
            time.sleep(pausa)
            nueva = False

            if term["p1"] or trunc["p1"]:
                partidas += 1
                if rew["p1"] > 0:
                    victorias += 1
                    resultado = "gana p1 (agente)"
                elif rew["p1"] < 0:
                    resultado = "gana p2 (aleatorio)"
                else:
                    resultado = "empate por tiempo"
                print(f"Partida {partidas}: {resultado}   "
                      f"[agente {victorias}/{partidas} = {victorias / partidas:.0%}]")
                obs, infos = env.reset()
                nueva = True
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    finally:
        if env is not None:
            env.close()
        proceso.terminate()
        try:
            proceso.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proceso.kill()
        print("Juego cerrado.")


if __name__ == "__main__":
    main()
