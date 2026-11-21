"""Crear el entorno y repartir observaciones, acciones y recompensas entre los dos jugadores.

Convención: en cada partida hay un jugador "A" (el aprendiz, o el primero en enfrentar.py)
y un jugador "B" (el rival). La máscara `lados` dice en qué partidas A es p1.
"""
import platform as _platform

import numpy as np
import torch

import footsiesgym

NUM_ENVS = 64


def plataforma():
    """'mac' en macOS y 'linux' en todo lo demás (incluido WSL)."""
    return "mac" if _platform.system() == "Darwin" else "linux"


def crear_entorno(num_envs=NUM_ENVS):
    """Entorno vectorizado con los especiales activados (9 acciones)."""
    config = {"num_envs": num_envs, "use_special_charge_action": True}
    return footsiesgym.make(config=config, platform=plataforma())


def mascara_lados(num_envs):
    """True en las partidas donde A es p1: la primera mitad."""
    return np.arange(num_envs) < num_envs // 2


def obs_de_cada_jugador(obs, lados):
    """Devuelve (obs de A, obs de B) como tensores."""
    lado = lados[:, None]
    obs_a = np.where(lado, obs["p1"], obs["p2"])
    obs_b = np.where(lado, obs["p2"], obs["p1"])
    return (torch.tensor(obs_a, dtype=torch.float32),
            torch.tensor(obs_b, dtype=torch.float32))


def acciones_para_entorno(lados, acciones_a, acciones_b):
    """Traduce las acciones de A y B al diccionario {'p1', 'p2'} que espera el juego."""
    return {
        "p1": np.where(lados, acciones_a, acciones_b),
        "p2": np.where(lados, acciones_b, acciones_a),
    }


def recompensa_de_a(lados, rew):
    """La recompensa desde el punto de vista de A."""
    return np.where(lados, rew["p1"], rew["p2"])
