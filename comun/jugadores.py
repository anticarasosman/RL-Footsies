"""Jugadores: cualquier cosa con un método actuar(obs, nuevas) -> acciones (array de enteros).

`nuevas` marca las partidas cuya observación es la primera de una partida nueva. Los
jugadores sin memoria (aleatorio, PPO) la ignoran; Dreamer la usa para borrar su memoria.

Todos tienen además `nombre` y `lados` ("ambos", "p1" o "desconocido": de qué lados
aprendió a jugar).
"""
import os

import numpy as np
import torch

from comun.modelos import cargar_modelo


class JugadorAleatorio:
    def __init__(self, n_acciones):
        self.n_acciones = n_acciones
        self.nombre = "aleatorio"
        self.lados = "ambos"

    def actuar(self, obs, nuevas=None):
        return np.random.randint(0, self.n_acciones, size=obs.shape[0])


class JugadorRed:
    """Juega con una RedActorCritico, sin aprender."""

    def __init__(self, red, nombre="red", lados="ambos", determinista=False):
        self.red = red
        self.nombre = nombre
        self.lados = lados
        self.determinista = determinista

    def reiniciar(self):
        pass  # no tiene memoria

    def actuar(self, obs, nuevas=None):
        with torch.no_grad():
            if self.determinista:
                return self.red.actor(obs).argmax(dim=1).numpy()
            return self.red.obtener_accion_y_valor(obs)[0].numpy()


def cargar_jugador(especificacion, n_acciones, determinista=False):
    """'aleatorio' o la ruta a un .pt."""
    if especificacion == "aleatorio":
        return JugadorAleatorio(n_acciones)
    red, metadatos = cargar_modelo(especificacion)
    nombre = os.path.splitext(os.path.basename(especificacion))[0]
    lados = metadatos.get("lados", "desconocido")
    if metadatos.get("algoritmo") == "dreamer":
        from algoritmos.dreamer import JugadorDreamer
        return JugadorDreamer(red, nombre=nombre, lados=lados, determinista=determinista)
    return JugadorRed(red, nombre=nombre, lados=lados, determinista=determinista)
