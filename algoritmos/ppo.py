"""PPO (estilo CleanRL), con el aprendiz jugando de ambos lados contra un rival cualquiera."""
from dataclasses import dataclass

import numpy as np
import torch

from agente.red import RedActorCritico
from comun.jugadores import JugadorRed


@dataclass
class ConfigPPO:
    num_steps: int = 128          # pasos por partida en cada recolección
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4  # qué tan grande es cada ajuste de pesos
    update_epochs: int = 4         # cuántas pasadas sobre los mismos datos
    num_minibatches: int = 4       # en cuántos grupos se divide cada pasada
    clip_coef: float = 0.2         # el clip de PPO: ±20%
    ent_coef: float = 0.01         # peso del bono de entropía
    vf_coef: float = 0.5           # peso de la pérdida del crítico
    max_grad_norm: float = 0.5     # límite al tamaño del gradiente


class PPO:
    nombre = "ppo"
    imprimir = ("entropia", "approx_kl", "clipfrac")   # métricas que muestra entrenar.py

    def __init__(self, n_obs, n_acciones, num_envs, config=None):
        self.config = config or ConfigPPO()
        self.num_envs = num_envs
        self.red = RedActorCritico(n_obs, n_acciones)
        self.optimizador = torch.optim.Adam(self.red.parameters(),
                                            lr=self.config.learning_rate, eps=1e-5)
        self.batch_size = self.config.num_steps * num_envs            # 8192 con 64 partidas
        self.minibatch_size = self.batch_size // self.config.num_minibatches
        self.pasos_entorno = 0

    def metadatos(self):
        """Lo que se guarda junto a los pesos para saber cómo reconstruir la red."""
        return {"algoritmo": self.nombre}

    def crear_jugador(self, red, nombre):
        return JugadorRed(red, nombre=nombre)

    def como_jugador(self):
        """El aprendiz como jugador (para evaluarlo); comparte la red, no la copia."""
        return JugadorRed(self.red, nombre="aprendiz")

    # ── Una iteración completa ────────────────────────────────────────

    def iteracion(self, estado, rival):
        """Recolecta datos contra `rival`, calcula ventajas y ajusta la red."""
        datos = self.recolectar(estado, rival)
        self.calcular_ventajas(estado, datos)
        metricas = self.actualizar(datos)
        return datos, metricas

    # ── A. Recolección ────────────────────────────────────────────────

    def crear_tablas(self, n_obs):
        pasos, n = self.config.num_steps, self.num_envs
        return {
            "obs": torch.zeros((pasos, n, n_obs)),
            "acciones": torch.zeros((pasos, n), dtype=torch.long),
            "logprobs": torch.zeros((pasos, n)),
            "rewards": torch.zeros((pasos, n)),
            "dones": torch.zeros((pasos, n)),
            "values": torch.zeros((pasos, n)),
            "terminadas": torch.zeros((pasos, n), dtype=torch.bool),
        }

    def recolectar(self, estado, rival):
        """Juega num_steps pasos y devuelve las tablas con los datos del aprendiz."""
        datos = self.crear_tablas(estado.n_obs)
        for step in range(self.config.num_steps):
            datos["obs"][step] = estado.next_obs
            datos["dones"][step] = estado.next_done

            with torch.no_grad():
                accion, logprob, _, valor = self.red.obtener_accion_y_valor(estado.next_obs)
            datos["acciones"][step] = accion
            datos["logprobs"][step] = logprob
            datos["values"][step] = valor.flatten()

            accion_rival = rival.actuar(estado.next_obs_rival, estado.next_done)
            recompensa, terminadas = estado.paso(accion.numpy(), accion_rival)

            datos["rewards"][step] = torch.tensor(recompensa, dtype=torch.float32)
            datos["terminadas"][step] = torch.tensor(terminadas)
            self.pasos_entorno += self.num_envs
        return datos

    # ── B. Ventajas (GAE) ─────────────────────────────────────────────

    def calcular_ventajas(self, estado, datos):
        """Calcula las ventajas con GAE y los retornos, y los guarda en datos."""
        c = self.config
        with torch.no_grad():
            next_value = self.red.obtener_valor(estado.next_obs).flatten()
            advantages = torch.zeros_like(datos["rewards"])
            lastgaelam = 0
            for t in reversed(range(c.num_steps)):
                if t == c.num_steps - 1:
                    nextnonterminal = 1.0 - estado.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - datos["dones"][t + 1]
                    nextvalues = datos["values"][t + 1]
                delta = datos["rewards"][t] + c.gamma * nextvalues * nextnonterminal - datos["values"][t]
                advantages[t] = lastgaelam = delta + c.gamma * c.gae_lambda * nextnonterminal * lastgaelam
            datos["advantages"] = advantages
            datos["returns"] = advantages + datos["values"]

    # ── C. Actualización de la red ────────────────────────────────────

    @staticmethod
    def aplanar_datos(datos):
        """Convierte las tablas (pasos, partidas) en listas de decisiones sueltas."""
        n_obs = datos["obs"].shape[2]
        return {
            "obs": datos["obs"].reshape(-1, n_obs),
            "acciones": datos["acciones"].reshape(-1),
            "logprobs": datos["logprobs"].reshape(-1),
            "advantages": datos["advantages"].reshape(-1),
            "returns": datos["returns"].reshape(-1),
        }

    def actualizar(self, datos):
        """Ajusta los pesos de la red con PPO usando los datos recolectados."""
        lote = self.aplanar_datos(datos)
        indices = np.arange(self.batch_size)
        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for inicio in range(0, self.batch_size, self.minibatch_size):
                mb = indices[inicio:inicio + self.minibatch_size]
                metricas = self.paso_de_optimizacion(lote, mb)
        return metricas

    def paso_de_optimizacion(self, lote, mb):
        c = self.config
        # C1. Volver a evaluar las decisiones con la red actual
        _, nuevo_logprob, entropia, nuevo_valor = self.red.obtener_accion_y_valor(
            lote["obs"][mb], lote["acciones"][mb])

        # C2. Razón entre la política nueva y la antigua
        logratio = nuevo_logprob - lote["logprobs"][mb]
        ratio = logratio.exp()

        # C3. Métricas de diagnóstico (no afectan el entrenamiento)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > c.clip_coef).float().mean()

        # C4. Normalizar las ventajas del minibatch
        ventajas = lote["advantages"][mb]
        ventajas = (ventajas - ventajas.mean()) / (ventajas.std() + 1e-8)

        # C5. Pérdida de la política (el clip de PPO)
        perdida_sin_clip = -ventajas * ratio
        perdida_con_clip = -ventajas * torch.clamp(ratio, 1 - c.clip_coef, 1 + c.clip_coef)
        perdida_politica = torch.max(perdida_sin_clip, perdida_con_clip).mean()

        # C6. Pérdida del crítico
        perdida_valor = 0.5 * ((nuevo_valor.view(-1) - lote["returns"][mb]) ** 2).mean()

        # C7. Pérdida total
        entropia_media = entropia.mean()
        perdida = perdida_politica - c.ent_coef * entropia_media + c.vf_coef * perdida_valor

        # C8. Ajustar los pesos
        self.optimizador.zero_grad()
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(self.red.parameters(), c.max_grad_norm)
        self.optimizador.step()

        return {
            "perdida_politica": perdida_politica.item(),
            "perdida_valor": perdida_valor.item(),
            "entropia": entropia_media.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": clipfrac.item(),
        }
