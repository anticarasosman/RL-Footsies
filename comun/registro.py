"""Acumula los resultados del aprendiz contra su rival de entrenamiento entre impresiones."""
import numpy as np

from comun.evaluacion import resumir


class Registro:
    def __init__(self, lados):
        self.lados = lados
        self.reiniciar()

    def reiniciar(self):
        n = len(self.lados)
        self.victorias = np.zeros(n)
        self.derrotas = np.zeros(n)
        self.empates = np.zeros(n)

    def agregar(self, datos):
        """datos: las tablas de una iteración, con 'rewards' y 'terminadas' (pasos, partidas)."""
        r = datos["rewards"].numpy()
        terminadas = datos["terminadas"].numpy()
        self.victorias += (r > 0).sum(axis=0)
        self.derrotas += (r < 0).sum(axis=0)
        self.empates += (terminadas & (r == 0)).sum(axis=0)

    def resumen(self):
        return resumir(self.victorias, self.derrotas, self.empates, self.lados)
