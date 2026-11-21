"""Modos de entrenamiento: deciden contra quién juega el aprendiz en cada iteración.

Todos se crean con  Modo(algoritmo, n_acciones, args)  y tienen el método
    rival_para_iteracion(algoritmo, iteracion) -> jugador
y, opcionalmente,
    resumen_texto() -> str   (algo corto para imprimir junto al progreso)

Para agregar un modo basta con crear una clase nueva con esos métodos y sumarla a MODOS.
"""
import copy
import os

import numpy as np

from comun.jugadores import JugadorAleatorio
from comun.modelos import guardar_modelo


class ModoVsAleatorio:
    """El rival siempre juega al azar."""

    def __init__(self, algoritmo, n_acciones, args=None):
        self.rival = JugadorAleatorio(n_acciones)

    def rival_para_iteracion(self, algoritmo, iteracion):
        return self.rival


class ModoSelfPlay:
    """El rival es una copia congelada del aprendiz, renovada al inicio de cada iteración."""

    def __init__(self, algoritmo, n_acciones, args=None):
        self.rival = algoritmo.crear_jugador(copy.deepcopy(algoritmo.red), "copia del aprendiz")

    def rival_para_iteracion(self, algoritmo, iteracion):
        self.rival.red.load_state_dict(algoritmo.red.state_dict())
        self.rival.reiniciar()  # pesos nuevos: la memoria vieja (si tiene) ya no sirve
        return self.rival


class ModoLiga:
    """Self-play contra el aprendiz actual y contra versiones antiguas de sí mismo.

    - Cada `cada` iteraciones se guarda una foto (checkpoint) del aprendiz, en memoria
      y en disco (carpeta `<carpeta>/<nombre>_liga/`).
    - Al inicio de cada iteración se elige el rival:
        con probabilidad `prob_actual`  → la copia actual (como en self-play),
        si no                           → una foto antigua elegida al azar.

    Jugar contra versiones antiguas evita que el aprendiz "olvide" cómo ganarles
    (que se ponga a dar vueltas en círculo entre estrategias).
    """

    def __init__(self, algoritmo, n_acciones, args):
        self.cada = args.liga_cada
        self.prob_actual = args.liga_prob_actual
        self.carpeta = os.path.join(args.carpeta, f"{args.nombre}_liga")
        self.semilla = args.semilla

        # Un solo jugador rival; en cada iteración se le cargan los pesos que toquen.
        self.rival = algoritmo.crear_jugador(copy.deepcopy(algoritmo.red), "copia del aprendiz")
        self.fotos = []          # lista de (iteración, pesos)

        # Para el resumen: cuántas veces tocó cada tipo de rival desde la última impresión.
        self.veces_actual = 0
        self.veces_antigua = 0

    # ── Fotos ─────────────────────────────────────────────────────────

    def guardar_foto(self, algoritmo, iteracion):
        """Guarda los pesos actuales del aprendiz como un rival más de la liga."""
        pesos = copy.deepcopy(algoritmo.red.state_dict())
        self.fotos.append((iteracion, pesos))
        ruta = os.path.join(self.carpeta, f"it_{iteracion:04d}.pt")
        guardar_modelo(ruta, algoritmo.red, **algoritmo.metadatos(), modo="liga",
                       lados="ambos", iteraciones=iteracion, semilla=self.semilla)

    # ── Elegir rival ──────────────────────────────────────────────────

    def rival_para_iteracion(self, algoritmo, iteracion):
        # Los pesos al inicio de la iteración `iteracion` son el resultado de
        # `iteracion - 1` actualizaciones. La primera foto (it_0000) es la red inicial.
        if (iteracion - 1) % self.cada == 0:
            self.guardar_foto(algoritmo, iteracion - 1)

        if np.random.rand() < self.prob_actual:
            self.rival.red.load_state_dict(algoritmo.red.state_dict())
            self.rival.nombre = "copia del aprendiz"
            self.veces_actual += 1
        else:
            it_foto, pesos = self.fotos[np.random.randint(len(self.fotos))]
            self.rival.red.load_state_dict(pesos)
            self.rival.nombre = f"foto it_{it_foto:04d}"
            self.veces_antigua += 1
        self.rival.reiniciar()
        return self.rival

    def resumen_texto(self):
        texto = (f"liga: {len(self.fotos)} fotos, "
                 f"rival actual {self.veces_actual}x / antiguo {self.veces_antigua}x")
        self.veces_actual = self.veces_antigua = 0
        return texto


MODOS = {
    "vs_aleatorio": ModoVsAleatorio,
    "selfplay": ModoSelfPlay,
    "liga": ModoLiga,
}
