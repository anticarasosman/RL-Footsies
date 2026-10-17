import torch


class Estado:
    """Agrupa lo que comparten las funciones de entrenamiento."""

    def __init__(self, env, red, optimizador, next_obs, next_done):
        self.env = env
        self.red = red
        self.optimizador = optimizador
        self.next_obs = next_obs
        self.next_done = next_done

    def actualizar(self, obs_env, term, trunc):
        """Guarda la observación y el fin de partida de p1 para el próximo paso."""
        self.next_obs = torch.tensor(obs_env["p1"], dtype=torch.float32)
        terminadas = term["p1"] | trunc["p1"]
        self.next_done = torch.tensor(terminadas, dtype=torch.float32)