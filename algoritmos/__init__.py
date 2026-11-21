"""Algoritmos de entrenamiento disponibles.

Para agregar uno nuevo: crear su archivo en esta carpeta con una clase que tenga
    nombre, imprimir, red, pasos_entorno,
    iteracion(estado, rival) -> (datos, metricas),
    metadatos(), crear_jugador(red, nombre) y como_jugador(),
y sumarla a ALGORITMOS.
"""
from algoritmos.dreamer import Dreamer
from algoritmos.ppo import PPO

ALGORITMOS = {
    "ppo": PPO,
    "dreamer": Dreamer,
}
