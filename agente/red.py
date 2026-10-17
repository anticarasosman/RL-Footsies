import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RedActorCritico(nn.Module):
    def __init__(self, n_obs, n_acciones):
        super().__init__()
        self.critico = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64)),   # capa 1: 88 números → 64
            nn.Tanh(),           # activación
            layer_init(nn.Linear(64, 64)),   # capa 2: 64 → 64
            nn.Tanh(),           # activación
            layer_init(nn.Linear(64, 1), std=1.0),    # capa 3: 64 → 1 (el valor)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64)),   # capa 1: 88 números → 64
            nn.Tanh(),           # activación
            layer_init(nn.Linear(64, 64)),   # capa 2: 64 → 64
            nn.Tanh(),           # activación
            layer_init(nn.Linear(64, n_acciones), std=0.01),    # capa 3: 64 → 9 (acciones posibles)
        )

    def obtener_valor(self, obs):
        return self.critico(obs)

    def obtener_accion_y_valor(self, obs, accion = None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if accion is None:
            accion = probs.sample()
        return accion, probs.log_prob(accion), probs.entropy(), self.critico(obs)
