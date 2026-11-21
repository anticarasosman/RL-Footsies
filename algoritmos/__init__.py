"""Algoritmos de entrenamiento disponibles.

Para agregar uno nuevo: crear su archivo en esta carpeta (con una clase que tenga
`nombre`, `red`, `iteracion(estado, rival)` y `como_jugador()`) y sumarlo a ALGORITMOS.
"""
from algoritmos.ppo import PPO

ALGORITMOS = {
    "ppo": PPO,
}
