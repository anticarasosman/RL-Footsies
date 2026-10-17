import torch
import numpy as np

class Estado:
    """Agrupa lo que comparten las funciones de entrenamiento."""

    def __init__(self, env, red, optimizador, aprendiz_es_p1, next_obs, next_done):
        self.env = env
        self.red = red
        self.optimizador = optimizador
        self.aprendiz_es_p1 = aprendiz_es_p1
        self.next_obs = next_obs
        self.next_done = next_done

    def actualizar(self, obs_env, term, trunc):
        """Guarda la observación y el fin de partida de p1 para el próximo paso."""
        obs_aprendiz = np.where(self.aprendiz_es_p1[:, None], obs_env["p1"], obs_env["p2"])
        self.next_obs = torch.tensor(obs_aprendiz, dtype=torch.float32)
        terminadas = term["p1"] | trunc["p1"]
        self.next_done = torch.tensor(terminadas, dtype=torch.float32)