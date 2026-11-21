"""Guardar y cargar modelos (.pt) junto con sus metadatos.

Formato nuevo: {"formato": 2, "algoritmo": ..., "modo": ..., "lados": ..., ..., "pesos": state_dict}
Formato antiguo: el state_dict solo. Se sigue pudiendo cargar.

Los modelos de Dreamer guardan además "arquitectura" (los tamaños de sus redes).
"""
import os

import torch

from agente.red import RedActorCritico


def tamanos_de_la_red(pesos):
    """Deduce (n_obs, n_acciones) a partir de los pesos guardados."""
    return pesos["actor.0.weight"].shape[1], pesos["actor.4.weight"].shape[0]


def guardar_modelo(ruta, red, **metadatos):
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    torch.save({"formato": 2, **metadatos, "pesos": red.state_dict()}, ruta)


def cargar_modelo(ruta):
    """Devuelve (red, metadatos). La red queda en modo evaluación."""
    contenido = torch.load(ruta, map_location="cpu")
    if "pesos" in contenido:
        pesos = contenido["pesos"]
        metadatos = {k: v for k, v in contenido.items() if k != "pesos"}
    else:
        pesos = contenido
        metadatos = {"formato": 1, "algoritmo": "ppo", "lados": "desconocido"}

    if metadatos.get("algoritmo") == "dreamer":
        from algoritmos.dreamer import RedDreamer
        red = RedDreamer(**metadatos["arquitectura"])
    else:
        red = RedActorCritico(*tamanos_de_la_red(pesos))
    red.load_state_dict(pesos)
    red.eval()
    return red, metadatos
