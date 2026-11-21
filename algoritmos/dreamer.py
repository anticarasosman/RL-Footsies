"""DreamerV3 (versión simplificada para observaciones vectoriales y acciones discretas).

Idea general:
  1. Jugar en el juego real y guardar todo en un buffer de repetición.
  2. Entrenar un MODELO DEL MUNDO con secuencias del buffer: aprende a resumir lo que
     ha visto en un estado latente (h, z) y a predecir, a partir de ese estado y de una
     acción, el siguiente estado, la recompensa y si la partida sigue.
  3. Entrenar al ACTOR y al CRÍTICO "soñando": desde estados del buffer, el modelo del
     mundo imagina partidas de H pasos y el actor aprende a sacar más recompensa en ellas.

Referencia: Hafner et al. (2023), "Mastering Diverse Domains through World Models".
Diferencias con el paper (para que sea corto y rápido en CPU): redes más pequeñas,
Adam en vez de LaProp, sin "replay context" y sin normalización adaptativa de gradientes.

Piezas de este archivo:
  - utilidades: symlog / symexp, two-hot, MLP
  - RedDreamer: todas las redes (modelo del mundo + actor + crítico)
  - JugadorDreamer: juega con la red, llevando la memoria (h, z) de cada partida
  - Repeticion: el buffer de experiencia
  - Dreamer: el algoritmo (recolectar, entrenar el modelo, entrenar actor-crítico)
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfigDreamer:
    # Recolección y entrenamiento
    pasos_por_iteracion: int = 16          # pasos jugados en cada partida por iteración
    actualizaciones_por_iteracion: int = 8   # pasos de gradiente por iteración
    batch: int = 32                        # secuencias por actualización
    largo_secuencia: int = 32              # pasos por secuencia
    capacidad: int = 4000                  # pasos guardados por partida en el buffer
    fraccion_con_final: float = 0.5        # secuencias que incluyen el final de una partida
    # Imaginación
    horizonte: int = 15                    # pasos que se "sueñan" desde cada estado
    inicios_imaginacion: int = 256         # desde cuántos estados del lote se sueña
    gamma: float = 0.99                    # mismo descuento que PPO, para comparar
    lambda_: float = 0.95
    # Tamaños de las redes
    deter: int = 128                       # tamaño de h (memoria del GRU)
    stoch: int = 16                        # z: cantidad de variables categóricas...
    clases: int = 16                       # ...con estas clases cada una
    oculto: int = 128                      # neuronas de las capas ocultas
    # Optimización
    lr_modelo: float = 1e-4
    lr_actor: float = 1e-4                 # el paper usa 3e-5, pero con muchas más
    lr_critico: float = 1e-4               # actualizaciones de las que hacemos aquí
    ent_coef: float = 3e-4                 # bono de entropía del actor
    unimix: float = 0.01                   # 1% de probabilidad uniforme en las categóricas
    free_bits: float = 1.0                 # el KL no se castiga por debajo de 1 nat
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    tau_critico_lento: float = 0.02        # qué tan rápido sigue el crítico lento al real


# ── Utilidades ────────────────────────────────────────────────────────


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


def twohot_media(logits, bins):
    """Valor esperado de una distribución two-hot (bins en espacio symlog)."""
    return symexp((logits.softmax(-1) * bins).sum(-1))


def twohot_perdida(logits, objetivo, bins):
    """−log p(objetivo), repartiendo el objetivo entre los dos bins más cercanos."""
    y = symlog(objetivo).clamp(bins[0], bins[-1])
    arriba = torch.bucketize(y, bins).clamp(1, len(bins) - 1)
    abajo = arriba - 1
    peso_arriba = (y - bins[abajo]) / (bins[arriba] - bins[abajo])
    objetivo_dist = torch.zeros_like(logits)
    objetivo_dist.scatter_(-1, abajo.unsqueeze(-1), (1 - peso_arriba).unsqueeze(-1))
    objetivo_dist.scatter_add_(-1, arriba.unsqueeze(-1), peso_arriba.unsqueeze(-1))
    return -(objetivo_dist * logits.log_softmax(-1)).sum(-1)


def mlp(entrada, salida, oculto, capas=2, salida_cero=False):
    """Capas Linear + LayerNorm + SiLU, y una capa final lineal."""
    partes, d = [], entrada
    for _ in range(capas):
        partes += [nn.Linear(d, oculto, bias=False), nn.LayerNorm(oculto), nn.SiLU()]
        d = oculto
    final = nn.Linear(d, salida)
    if salida_cero:  # DreamerV3 parte prediciendo 0 en recompensa y valor
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    partes.append(final)
    return nn.Sequential(*partes)


def kl_categoricas(p, q):
    """KL(p || q) sumado sobre todas las variables categóricas. p, q: (..., stoch, clases)."""
    return (p * (p.log() - q.log())).sum((-2, -1))


# ── Las redes ─────────────────────────────────────────────────────────


class RedDreamer(nn.Module):
    """Modelo del mundo (encoder, RSSM, decoder, recompensa, continuación) + actor + crítico.

    El "estado" del modelo es (h, z):
      h: memoria determinista (salida de un GRU), resume la historia.
      z: variables categóricas muestreadas, el detalle del momento actual.
    feat = [h, z] es lo que leen el decoder, las cabezas, el actor y el crítico.
    """

    def __init__(self, n_obs, n_acciones, deter=256, stoch=16, clases=16, oculto=256,
                 unimix=0.01):
        super().__init__()
        # int()/float(): para que se guarde como números normales de Python en el .pt
        self.arquitectura = dict(n_obs=int(n_obs), n_acciones=int(n_acciones), deter=int(deter),
                                 stoch=int(stoch), clases=int(clases), oculto=int(oculto),
                                 unimix=float(unimix))
        self.n_acciones, self.deter, self.stoch, self.clases = n_acciones, deter, stoch, clases
        self.unimix = unimix
        tam_z = stoch * clases
        self.n_feat = deter + tam_z

        # Modelo del mundo
        self.encoder = mlp(n_obs, oculto, oculto)
        self.entrada_gru = nn.Sequential(nn.Linear(tam_z + n_acciones, oculto, bias=False),
                                         nn.LayerNorm(oculto), nn.SiLU())
        self.gru = nn.GRUCell(oculto, deter)
        self.prior = mlp(deter, tam_z, oculto, capas=1)              # z sin mirar la obs
        self.posterior = mlp(deter + oculto, tam_z, oculto, capas=1)  # z mirando la obs
        self.decoder = mlp(self.n_feat, n_obs, oculto)
        self.recompensa = mlp(self.n_feat, 255, oculto, salida_cero=True)
        self.continua = mlp(self.n_feat, 1, oculto)
        # Actor y crítico
        self.actor = mlp(self.n_feat, n_acciones, oculto)
        self.critico = mlp(self.n_feat, 255, oculto, salida_cero=True)
        # 255 bins en espacio symlog, para la recompensa y el valor
        self.register_buffer("bins", torch.linspace(-20, 20, 255))

    # Partes del modelo del mundo que entrena el optimizador del modelo
    PARTES_MODELO = ("encoder", "entrada_gru", "gru", "prior", "posterior",
                     "decoder", "recompensa", "continua")

    def estado_inicial(self, n, dispositivo):
        return (torch.zeros(n, self.deter, device=dispositivo),
                torch.zeros(n, self.stoch * self.clases, device=dispositivo))

    def probs(self, logits):
        """Logits planos → probabilidades (…, stoch, clases), con 1% uniforme."""
        p = logits.view(*logits.shape[:-1], self.stoch, self.clases).softmax(-1)
        return (1 - self.unimix) * p + self.unimix / self.clases

    def muestrear(self, probs):
        """Muestra z en one-hot, dejando pasar el gradiente (straight-through)."""
        indices = torch.distributions.Categorical(probs=probs).sample()
        uno = F.one_hot(indices, self.clases).float()
        return (uno + probs - probs.detach()).flatten(-2)

    def codificar(self, obs):
        return self.encoder(symlog(obs))

    def paso_imaginado(self, h, z, accion_uno):
        """(h, z) + acción → siguiente h y las probabilidades del prior de z."""
        h = self.gru(self.entrada_gru(torch.cat([z, accion_uno], -1)), h)
        return h, self.probs(self.prior(h))

    def paso_observado(self, h, embed):
        """Probabilidades del posterior de z, mirando la observación codificada."""
        return self.probs(self.posterior(torch.cat([h, embed], -1)))

    def dist_actor(self, feat):
        p = self.actor(feat).softmax(-1)
        p = (1 - self.unimix) * p + self.unimix / self.n_acciones
        return torch.distributions.Categorical(probs=p)


# ── El jugador ────────────────────────────────────────────────────────


class JugadorDreamer:
    """Juega con una RedDreamer. Recuerda (h, z, acción anterior) de cada partida.

    `nuevas` marca las partidas cuya observación es la primera de una partida nueva:
    ahí se borra la memoria.
    """

    def __init__(self, red, nombre="dreamer", lados="ambos", determinista=False):
        self.red = red
        self.nombre = nombre
        self.lados = lados
        self.determinista = determinista
        self.reiniciar()

    def reiniciar(self):
        self.h = None

    def actuar(self, obs, nuevas=None):
        red = self.red
        dispositivo = red.bins.device
        obs = obs.to(dispositivo)
        n = obs.shape[0]
        with torch.no_grad():
            if self.h is None or self.h.shape[0] != n:
                h, _ = red.estado_inicial(n, dispositivo)
            else:
                h, _ = red.paso_imaginado(self.h, self.z, self.accion)
                if nuevas is not None:
                    nuevas = torch.as_tensor(nuevas, device=dispositivo).bool().view(-1, 1)
                    h = torch.where(nuevas, torch.zeros_like(h), h)
            z = red.muestrear(red.paso_observado(h, red.codificar(obs)))
            dist = red.dist_actor(torch.cat([h, z], -1))
            accion = dist.probs.argmax(-1) if self.determinista else dist.sample()
        self.h, self.z = h, z
        self.accion = F.one_hot(accion, red.n_acciones).float()
        return accion.cpu().numpy()


# ── El buffer de repetición ───────────────────────────────────────────


class Repeticion:
    """Guarda los últimos `capacidad` pasos de cada partida en paralelo.

    En la posición t: obs_t, la acción tomada en obs_t, la recompensa y si la partida
    terminó por esa acción, y si obs_t es la primera de una partida.
    """

    def __init__(self, capacidad, num_envs, n_obs):
        self.capacidad, self.num_envs = capacidad, num_envs
        self.obs = np.zeros((capacidad, num_envs, n_obs), np.float32)
        self.acciones = np.zeros((capacidad, num_envs), np.int64)
        self.recompensas = np.zeros((capacidad, num_envs), np.float32)
        self.terminales = np.zeros((capacidad, num_envs), np.float32)
        self.primeros = np.zeros((capacidad, num_envs), np.float32)
        self.n = 0  # pasos escritos en total

    def tamano(self):
        return min(self.n, self.capacidad)

    def agregar(self, obs, acciones, recompensas, terminales, primeros):
        i = self.n % self.capacidad
        self.obs[i] = obs
        self.acciones[i] = acciones
        self.recompensas[i] = recompensas
        self.terminales[i] = terminales
        self.primeros[i] = primeros
        self.n += 1

    def muestrear(self, batch, largo, fraccion_con_final=0.0):
        """`batch` secuencias contiguas de `largo` pasos, cada una de una partida al azar.

        Las partidas terminan en ~0.5% de los pasos, así que casi ninguna secuencia
        tendría un final y el modelo no aprendería a predecir recompensas. Por eso una
        fracción de las secuencias se elige de modo que contenga el final de una partida.
        """
        mas_antiguo = self.n % self.capacidad if self.n > self.capacidad else 0
        tam = self.tamano()
        inicios = np.random.randint(0, tam - largo + 1, size=batch)
        envs = np.random.randint(0, self.num_envs, size=batch)

        n_final = int(batch * fraccion_con_final)
        if n_final > 0:
            orden = (mas_antiguo + np.arange(tam)) % self.capacidad   # del más viejo al más nuevo
            tiempos, envs_final = np.nonzero(self.terminales[orden])
            if len(tiempos) > 0:
                elegidos = np.random.randint(len(tiempos), size=n_final)
                # El final queda en una posición al azar dentro de la secuencia
                posicion = np.random.randint(0, largo, size=n_final)
                inicios[:n_final] = np.clip(tiempos[elegidos] - posicion, 0, tam - largo)
                envs[:n_final] = envs_final[elegidos]
        idx = (mas_antiguo + inicios[:, None] + np.arange(largo)[None]) % self.capacidad

        def tomar(tabla):
            return torch.as_tensor(tabla[idx, envs[:, None]])   # (batch, largo, ...)

        primeros = tomar(self.primeros).clone()
        primeros[:, 0] = 1  # cada secuencia parte con la memoria vacía
        return {"obs": tomar(self.obs), "acciones": tomar(self.acciones),
                "recompensas": tomar(self.recompensas), "terminales": tomar(self.terminales),
                "primeros": primeros}


# ── El algoritmo ──────────────────────────────────────────────────────


class Dreamer:
    nombre = "dreamer"
    # Qué mirar: acierto_recompensa y recompensa_imaginada_max deben subir de 0;
    # después la entropía debería bajar de 2.19 (el actor empieza a decidir).
    imprimir = ("perdida_modelo", "acierto_recompensa", "recompensa_imaginada_max",
                "entropia", "retorno")

    def __init__(self, n_obs, n_acciones, num_envs, config=None):
        self.config = c = config or ConfigDreamer()
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.red = RedDreamer(n_obs, n_acciones, c.deter, c.stoch, c.clases, c.oculto,
                              c.unimix).to(self.dispositivo)
        self.critico_lento = copy.deepcopy(self.red.critico).requires_grad_(False)

        partes = [getattr(self.red, p).parameters() for p in RedDreamer.PARTES_MODELO]
        self.opt_modelo = torch.optim.Adam([p for g in partes for p in g],
                                           lr=c.lr_modelo, eps=1e-8)
        self.opt_actor = torch.optim.Adam(self.red.actor.parameters(), lr=c.lr_actor, eps=1e-5)
        self.opt_critico = torch.optim.Adam(self.red.critico.parameters(),
                                            lr=c.lr_critico, eps=1e-5)

        self.repeticion = Repeticion(c.capacidad, num_envs, n_obs)
        self.jugador = JugadorDreamer(self.red, nombre="aprendiz")
        self.escala_retornos = None
        self.pasos_entorno = 0
        print(f"Dreamer en {self.dispositivo}, "
              f"{sum(p.numel() for p in self.red.parameters()):,} parámetros")

    # ── Lo que usan entrenar.py y los modos ───────────────────────────

    def metadatos(self):
        return {"algoritmo": self.nombre, "arquitectura": self.red.arquitectura}

    def crear_jugador(self, red, nombre):
        return JugadorDreamer(red, nombre=nombre)

    def como_jugador(self):
        return JugadorDreamer(self.red, nombre="aprendiz")

    def iteracion(self, estado, rival):
        datos = self.recolectar(estado, rival)
        metricas = {}
        if self.repeticion.tamano() >= self.config.largo_secuencia:
            for _ in range(self.config.actualizaciones_por_iteracion):
                metricas = self.actualizar()
        return datos, metricas

    # ── A. Jugar y guardar en el buffer ───────────────────────────────

    def recolectar(self, estado, rival):
        pasos, n = self.config.pasos_por_iteracion, estado.num_envs
        datos = {"rewards": torch.zeros(pasos, n),
                 "terminadas": torch.zeros(pasos, n, dtype=torch.bool)}
        for paso in range(pasos):
            obs, primeros = estado.next_obs, estado.next_done
            acciones = self.jugador.actuar(obs, primeros)
            acciones_rival = rival.actuar(estado.next_obs_rival, primeros)
            recompensa, terminadas = estado.paso(acciones, acciones_rival)

            self.repeticion.agregar(obs.numpy(), acciones, recompensa, terminadas,
                                    primeros.numpy())
            datos["rewards"][paso] = torch.as_tensor(recompensa, dtype=torch.float32)
            datos["terminadas"][paso] = torch.as_tensor(terminadas)
            self.pasos_entorno += n
        return datos

    # ── B. Una actualización completa ─────────────────────────────────

    def actualizar(self):
        c = self.config
        lote = self.repeticion.muestrear(c.batch, c.largo_secuencia, c.fraccion_con_final)
        lote = {k: v.to(self.dispositivo) for k, v in lote.items()}
        feats, metricas_modelo = self.entrenar_modelo(lote)
        metricas_ac = self.entrenar_actor_critico(feats.detach())
        return {**metricas_modelo, **metricas_ac}

    # ── C. Modelo del mundo ───────────────────────────────────────────

    def entrenar_modelo(self, lote):
        c, red = self.config, self.red
        obs = lote["obs"]
        acciones = F.one_hot(lote["acciones"], red.n_acciones).float()
        batch, largo = obs.shape[:2]

        embed = red.codificar(obs)
        h, _ = red.estado_inicial(batch, self.dispositivo)
        hs, zs, posts, priors, feats_transicion = [], [], [], [], []
        for t in range(largo):
            if t > 0:  # en el primer paso de cada partida la memoria parte vacía
                h = torch.where(lote["primeros"][:, t, None] > 0, torch.zeros_like(h), h_sig)
            prior = red.probs(red.prior(h))
            post = red.paso_observado(h, embed[:, t])
            z = red.muestrear(post)
            # Lo que el modelo imagina al tomar la acción a_t: de ahí salen la
            # recompensa y si la partida sigue (igual que cuando sueña).
            h_sig, prior_sig = red.paso_imaginado(h, z, acciones[:, t])
            feats_transicion.append(torch.cat([h_sig, red.muestrear(prior_sig)], -1))
            hs.append(h); zs.append(z); posts.append(post); priors.append(prior)

        feat = torch.cat([torch.stack(hs, 1), torch.stack(zs, 1)], -1)   # (B, T, feat)
        feat_trans = torch.stack(feats_transicion, 1)
        post, prior = torch.stack(posts, 1), torch.stack(priors, 1)

        # C1. Predicción: reconstruir la observación, la recompensa y si sigue la partida
        perdida_obs = ((red.decoder(feat) - symlog(obs)) ** 2).sum(-1)
        logits_rec = red.recompensa(feat_trans)
        perdida_rec = twohot_perdida(logits_rec, lote["recompensas"], red.bins)
        perdida_cont = F.binary_cross_entropy_with_logits(
            red.continua(feat_trans).squeeze(-1), 1 - lote["terminales"], reduction="none")
        # C2. Que el prior (sin mirar) se parezca al posterior (mirando), y viceversa
        kl_dyn = kl_categoricas(post.detach(), prior)
        kl_rep = kl_categoricas(post, prior.detach())
        perdida = (perdida_obs + perdida_rec + perdida_cont
                   + c.beta_dyn * kl_dyn.clamp(min=c.free_bits)
                   + c.beta_rep * kl_rep.clamp(min=c.free_bits)).mean()

        self.opt_modelo.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.opt_modelo.param_groups[0]["params"], 1000)
        self.opt_modelo.step()

        # Diagnóstico: en los pasos donde hubo recompensa (±1), ¿la predice el modelo?
        # 1 = la predice perfecto, 0 = predice 0, −1 = predice el signo contrario.
        con_recompensa = lote["recompensas"] != 0
        with torch.no_grad():
            pred = twohot_media(logits_rec, red.bins)
            acierto = (pred * lote["recompensas"])[con_recompensa].mean().item() \
                if con_recompensa.any() else float("nan")

        return feat, {"perdida_modelo": perdida.item(), "acierto_recompensa": acierto, "perdida_obs": perdida_obs.mean().item(),
                      "perdida_recompensa": perdida_rec.mean().item(),
                      "perdida_continua": perdida_cont.mean().item(),
                      "kl": kl_dyn.mean().item()}

    # ── D. Actor y crítico en la imaginación ──────────────────────────

    def imaginar(self, inicio):
        """Sueña `horizonte` pasos desde cada estado de `inicio` (N, feat)."""
        c, red = self.config, self.red
        h, z = inicio[:, :red.deter], inicio[:, red.deter:]
        feats, acciones, recompensas, continuas = [inicio], [], [], []
        with torch.no_grad():
            for _ in range(c.horizonte):
                accion = red.dist_actor(feats[-1]).sample()
                h, prior = red.paso_imaginado(h, z, F.one_hot(accion, red.n_acciones).float())
                z = red.muestrear(prior)
                feat = torch.cat([h, z], -1)
                feats.append(feat)
                acciones.append(accion)
                recompensas.append(twohot_media(red.recompensa(feat), red.bins))
                continuas.append(torch.sigmoid(red.continua(feat).squeeze(-1)))
        return (torch.stack(feats), torch.stack(acciones),
                torch.stack(recompensas), torch.stack(continuas))

    def retornos_lambda(self, recompensas, continuas, valores):
        c = self.config
        retornos, siguiente = [], valores[-1]
        for k in reversed(range(c.horizonte)):
            siguiente = recompensas[k] + c.gamma * continuas[k] * (
                (1 - c.lambda_) * valores[k + 1] + c.lambda_ * siguiente)
            retornos.append(siguiente)
        return torch.stack(retornos[::-1])

    def entrenar_actor_critico(self, feats_inicio):
        c, red = self.config, self.red
        inicios = feats_inicio.reshape(-1, red.n_feat)
        elegidos = torch.randperm(len(inicios), device=inicios.device)[:c.inicios_imaginacion]
        feats, acciones, recompensas, continuas = self.imaginar(inicios[elegidos])

        with torch.no_grad():
            valores = twohot_media(red.critico(feats), red.bins)            # (H+1, N)
            retornos = self.retornos_lambda(recompensas, continuas, valores)  # (H, N)
            # Los pasos después de que el modelo cree que la partida terminó pesan ~0
            peso = torch.cumprod(torch.cat([torch.ones_like(continuas[:1]),
                                            c.gamma * continuas[:-1]]), 0)
            # Escala de los retornos: rango entre el percentil 5 y el 95 (promedio móvil)
            p5, p95 = torch.quantile(retornos.flatten(),
                                     torch.tensor([0.05, 0.95], device=retornos.device))
            rango = (p95 - p5).item()
            self.escala_retornos = rango if self.escala_retornos is None else (
                0.99 * self.escala_retornos + 0.01 * rango)
            ventajas = (retornos - valores[:-1]) / max(1.0, self.escala_retornos)

        # D1. Actor: más probabilidad a las acciones con ventaja positiva (REINFORCE)
        dist = red.dist_actor(feats[:-1])
        entropia = dist.entropy()
        perdida_actor = -(peso * (dist.log_prob(acciones) * ventajas
                                  + c.ent_coef * entropia)).mean()
        self.opt_actor.zero_grad()
        perdida_actor.backward()
        nn.utils.clip_grad_norm_(red.actor.parameters(), 100)
        self.opt_actor.step()

        # D2. Crítico: predecir los retornos, sin alejarse mucho del crítico lento
        logits_valor = red.critico(feats[:-1])
        with torch.no_grad():
            valor_lento = twohot_media(self.critico_lento(feats[:-1]), red.bins)
        perdida_critico = (peso * (twohot_perdida(logits_valor, retornos, red.bins)
                                   + twohot_perdida(logits_valor, valor_lento, red.bins))).mean()
        self.opt_critico.zero_grad()
        perdida_critico.backward()
        nn.utils.clip_grad_norm_(red.critico.parameters(), 100)
        self.opt_critico.step()

        with torch.no_grad():
            for lento, actual in zip(self.critico_lento.parameters(), red.critico.parameters()):
                lento.lerp_(actual, c.tau_critico_lento)

        return {"perdida_actor": perdida_actor.item(), "perdida_critico": perdida_critico.item(),
                "entropia": entropia.mean().item(), "retorno": retornos.mean().item(),
                "escala": self.escala_retornos, "ventaja_std": ventajas.std().item(),
                "recompensa_imaginada_max": recompensas.abs().max().item()}
