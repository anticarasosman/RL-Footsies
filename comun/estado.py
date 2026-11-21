"""Estado del entorno compartido entre la recolección, la evaluación y enfrentar.py."""
import torch

from comun.entorno import (acciones_para_entorno, mascara_lados, obs_de_cada_jugador,
                           recompensa_de_a)


class Estado:
    """Guarda la última observación de cada jugador y si la partida recién empezó.

    Al crearlo se reinicia el entorno.
    """

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.lados = mascara_lados(self.num_envs)

        obs, _ = env.reset()
        self.next_obs, self.next_obs_rival = obs_de_cada_jugador(obs, self.lados)
        self.next_done = torch.zeros(self.num_envs)

        self.n_obs = self.next_obs.shape[1]
        self.n_acciones = env.action_space("p1").n

    def paso(self, acciones_a, acciones_b):
        """Juega un paso en todas las partidas.

        Devuelve (recompensa de A, partidas que terminaron en este paso).
        """
        acciones = acciones_para_entorno(self.lados, acciones_a, acciones_b)
        obs, rew, term, trunc, _ = self.env.step(acciones)

        self.next_obs, self.next_obs_rival = obs_de_cada_jugador(obs, self.lados)
        terminadas = term["p1"] | trunc["p1"]
        self.next_done = torch.tensor(terminadas, dtype=torch.float32)
        return recompensa_de_a(self.lados, rew), terminadas
