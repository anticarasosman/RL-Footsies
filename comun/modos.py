"""Modos de entrenamiento: deciden contra quién juega el aprendiz en cada iteración.

Todos tienen el mismo método:  rival_para_iteracion(algoritmo, iteracion) -> jugador.
Para agregar un modo (por ejemplo, la liga de checkpoints), basta con crear una clase
nueva con ese método y sumarla a MODOS.
"""
import copy

from comun.jugadores import JugadorAleatorio, JugadorRed


class ModoVsAleatorio:
    """El rival siempre juega al azar."""

    def __init__(self, algoritmo, n_acciones):
        self.rival = JugadorAleatorio(n_acciones)

    def rival_para_iteracion(self, algoritmo, iteracion):
        return self.rival


class ModoSelfPlay:
    """El rival es una copia congelada del aprendiz, renovada al inicio de cada iteración."""

    def __init__(self, algoritmo, n_acciones):
        self.rival = JugadorRed(copy.deepcopy(algoritmo.red), nombre="copia del aprendiz")

    def rival_para_iteracion(self, algoritmo, iteracion):
        self.rival.red.load_state_dict(algoritmo.red.state_dict())
        return self.rival


MODOS = {
    "vs_aleatorio": ModoVsAleatorio,
    "selfplay": ModoSelfPlay,
}
