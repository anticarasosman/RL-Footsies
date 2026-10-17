import torch
import numpy as np


class Estado:
    """Agrupa lo que comparten las funciones de entrenamiento."""

    def __init__(self, env, red, oponente, optimizador, aprendiz_es_p1,
                 next_obs, next_obs_oponente, next_done):
        self.env = env
        self.red = red
        self.oponente = oponente
        self.optimizador = optimizador
        self.aprendiz_es_p1 = aprendiz_es_p1
        self.next_obs = next_obs
        self.next_obs_oponente = next_obs_oponente
        self.next_done = next_done

    def actualizar(self, obs_env, term, trunc):
        """Guarda las observaciones de ambos jugadores y el fin de partida para el próximo paso."""
        lado = self.aprendiz_es_p1[:, None]
        obs_aprendiz = np.where(lado, obs_env["p1"], obs_env["p2"])
        obs_oponente = np.where(lado, obs_env["p2"], obs_env["p1"])

        self.next_obs = torch.tensor(obs_aprendiz, dtype=torch.float32)
        self.next_obs_oponente = torch.tensor(obs_oponente, dtype=torch.float32)

        terminadas = term["p1"] | trunc["p1"]
        self.next_done = torch.tensor(terminadas, dtype=torch.float32)