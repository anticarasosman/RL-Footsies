import traceback

import numpy as np
import torch

import footsiesgym

from agente.red import RedActorCritico
from clases.estado import Estado

SEMILLA = 1
# 9 acciones permite usar especiales
N_ACCIONES = 9
NUM_ENVS = 64
NUM_STEPS = 128
AGENTES = ["p1", "p2"]
GAMMA = 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE = 2.5e-4   # qué tan grande es cada ajuste de pesos
UPDATE_EPOCHS = 4        # cuántas pasadas sobre los mismos datos
NUM_MINIBATCHES = 4      # en cuántos grupos se divide cada pasada
CLIP_COEF = 0.2          # el clip de PPO: ±20%
ENT_COEF = 0.01          # peso del bono de entropía
VF_COEF = 0.5            # peso de la pérdida del crítico
MAX_GRAD_NORM = 0.5      # límite al tamaño del gradiente
NUM_ITERACIONES = 500    # cuántas veces se repite el ciclo completo
BATCH_SIZE = NUM_STEPS * NUM_ENVS             # 8192 decisiones por recolección
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES  # 2048 por minibatch



def crear_entorno():
    """Crea el entorno vectorizado con los especiales activados y lo devuelve."""
    config = {"num_envs": NUM_ENVS, "use_special_charge_action": True}
    return footsiesgym.make(config=config, platform="linux")


def crear_estado(env):
    obs, infos = env.reset()
    aprendiz_es_p1 = np.arange(NUM_ENVS) < NUM_ENVS // 2
    obs_aprendiz = np.where(aprendiz_es_p1[:, None], obs["p1"], obs["p2"])
    obs_oponente = np.where(aprendiz_es_p1[:, None], obs["p2"], obs["p1"])

    n_obs = obs_aprendiz.shape[1]
    n_acciones = env.action_space("p1").n
    red = RedActorCritico(n_obs, n_acciones)
    rival = RedActorCritico(n_obs, n_acciones)
    optimizador = torch.optim.Adam(red.parameters(), lr=LEARNING_RATE, eps=1e-5)
    next_obs = torch.tensor(obs_aprendiz, dtype=torch.float32)
    next_obs_oponente = torch.tensor(obs_oponente, dtype=torch.float32)
    next_done = torch.zeros(NUM_ENVS)
    return Estado(env=env, red=red, oponente=rival, optimizador=optimizador,
                  aprendiz_es_p1=aprendiz_es_p1, next_obs=next_obs,
                  next_obs_oponente=next_obs_oponente, next_done=next_done)


def acciones_aleatorias():
    """Una acción aleatoria por partida para cada jugador."""
    acciones = {}
    for agente in AGENTES:
        acciones[agente] = np.random.randint(0, N_ACCIONES, size=NUM_ENVS)
    return acciones


def recolectar(estado):
    """Juega NUM_STEPS pasos y devuelve las tablas con los datos de p1."""
    datos = crear_tablas(estado)
    poblar_tablas(estado, datos)
    return datos


def crear_tablas(estado):
    """Crea las tablas vacías donde se guardan los datos de p1."""
    n_obs = estado.next_obs.shape[1]
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, n_obs))
    acciones = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.long)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS))
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS))
    dones = torch.zeros((NUM_STEPS, NUM_ENVS))
    values = torch.zeros((NUM_STEPS, NUM_ENVS))

    datos = {
        "obs": obs,
        "acciones": acciones,
        "logprobs": logprobs,
        "rewards": rewards,
        "dones": dones,
        "values": values,
    }
    return datos


def poblar_tablas(estado, datos):
    """Llena las tablas jugando con la red como p1 y p2 al azar."""
    for step in range(NUM_STEPS):
        datos["obs"][step] = estado.next_obs
        datos["dones"][step] = estado.next_done
        with torch.no_grad():
            accion, logprob, _, valor = estado.red.obtener_accion_y_valor(estado.next_obs)
        datos["acciones"][step] = accion
        datos["logprobs"][step] = logprob
        datos["values"][step] = valor.flatten()

        with torch.no_grad():
            accion_op, _, _, _ = estado.oponente.obtener_accion_y_valor(estado.next_obs_oponente)

        accion_red = accion.numpy()
        accion_oponente = accion_op.numpy()

        acciones_juego = {
            "p1": np.where(estado.aprendiz_es_p1, accion_red, accion_oponente),
            "p2": np.where(estado.aprendiz_es_p1, accion_oponente, accion_red),
        }

        obs_env, rew, term, trunc, infos = estado.env.step(acciones_juego)

        recompensa = np.where(estado.aprendiz_es_p1, rew["p1"], rew["p2"])
        datos["rewards"][step] = torch.tensor(recompensa, dtype=torch.float32)

        estado.actualizar(obs_env, term, trunc)


def calcular_ventajas(estado, datos):
    """Calcula las ventajas con GAE y los retornos, y los guarda en datos."""
    with torch.no_grad():
        next_value = estado.red.obtener_valor(estado.next_obs).flatten()
        advantages = torch.zeros_like(datos["rewards"])
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - estado.next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - datos["dones"][t + 1]
                nextvalues = datos["values"][t + 1]
            delta = datos["rewards"][t] + GAMMA * nextvalues * nextnonterminal - datos["values"][t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + datos["values"]
        datos["advantages"] = advantages
        datos["returns"] = returns

def aplanar_datos(datos):
    n_obs = datos["obs"].shape[2]
    lote = {
        "obs": datos["obs"].reshape(-1, n_obs),
        "acciones": datos["acciones"].reshape(-1),
        "logprobs": datos["logprobs"].reshape(-1),
        "advantages": datos["advantages"].reshape(-1),
        "returns": datos["returns"].reshape(-1),
    }
    return lote

def actualizar_red(estado, datos):
    lote = aplanar_datos(datos)
    indices = np.arange(BATCH_SIZE)

    for epoca in range(UPDATE_EPOCHS):
        np.random.shuffle(indices)
        for inicio in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            fin = inicio + MINIBATCH_SIZE
            mb = indices[inicio:fin]
            # C1. Volver a evaluar las decisiones con la red actual
            _, nuevo_logprob, entropia, nuevo_valor = estado.red.obtener_accion_y_valor(
                lote["obs"][mb], lote["acciones"][mb]
            )

            # C2. Razón entre la política nueva y la antigua
            logratio = nuevo_logprob - lote["logprobs"][mb]
            ratio = logratio.exp()

            # C3. Métricas de diagnóstico (no afectan el entrenamiento)
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > CLIP_COEF).float().mean()

            # C4. Normalizar las ventajas del minibatch
            ventajas_mb = lote["advantages"][mb]
            ventajas_mb = (ventajas_mb - ventajas_mb.mean()) / (ventajas_mb.std() + 1e-8)

            # C5. Pérdida de la política (el clip de PPO)
            perdida_sin_clip = -ventajas_mb * ratio
            perdida_con_clip = -ventajas_mb * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            perdida_politica = torch.max(perdida_sin_clip, perdida_con_clip).mean()

            # C6. Pérdida del crítico
            nuevo_valor = nuevo_valor.view(-1)
            perdida_valor = 0.5 * ((nuevo_valor - lote["returns"][mb]) ** 2).mean()

            # C7. Pérdida total
            entropia_media = entropia.mean()
            perdida = perdida_politica - ENT_COEF * entropia_media + VF_COEF * perdida_valor

            # C8. Ajustar los pesos
            estado.optimizador.zero_grad()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(estado.red.parameters(), MAX_GRAD_NORM)
            estado.optimizador.step()
    return {
        "perdida_politica": perdida_politica.item(),
        "perdida_valor": perdida_valor.item(),
        "entropia": entropia_media.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": clipfrac.item(),
    }

def contar_resultados(datos):
    """Victorias y partidas terminadas del aprendiz, separadas por lado."""
    mitad = NUM_ENVS // 2
    r = datos["rewards"]
    lado_p1, lado_p2 = r[:, :mitad], r[:, mitad:]
    return {
        "victorias_p1": (lado_p1 > 0).sum().item(),
        "partidas_p1": (lado_p1 != 0).sum().item(),
        "victorias_p2": (lado_p2 > 0).sum().item(),
        "partidas_p2": (lado_p2 != 0).sum().item(),
    }

def main():
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    env = crear_entorno()
    try:
        estado = crear_estado(env)
        acum = {"victorias_p1": 0, "partidas_p1": 0, "victorias_p2": 0, "partidas_p2": 0}


        for iteracion in range(1, NUM_ITERACIONES + 1):
            estado.oponente.load_state_dict(estado.red.state_dict())
            datos = recolectar(estado)
            calcular_ventajas(estado, datos)
            metricas = actualizar_red(estado, datos)

            resultados = contar_resultados(datos)
            for clave in acum:
                acum[clave] += resultados[clave]

            if iteracion % 10 == 0:
                wr_p1 = acum["victorias_p1"] / max(acum["partidas_p1"], 1)
                wr_p2 = acum["victorias_p2"] / max(acum["partidas_p2"], 1)
                print(
                    f"Iteración {iteracion:4d} | "
                    f"como p1 {wr_p1:6.1%} | como p2 {wr_p2:6.1%} | "
                    f"entropía {metricas['entropia']:.3f} | "
                    f"kl {metricas['approx_kl']:.4f} | "
                    f"clipfrac {metricas['clipfrac']:.3f}"
                )
                acum = {clave: 0 for clave in acum}

        torch.save(estado.red.state_dict(), "modelo.pt")
        print("Modelo guardado en modelo.pt")
    except Exception:
        print("Falló con este error:")
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
