"""Hacer jugar a dos jugadores (sin aprender) y resumir quién ganó."""
import numpy as np

from comun.estado import Estado


def resumir(victorias, derrotas, empates, lados):
    """Cuenta por partida → resultados de A como p1, como p2 y en total."""
    def bloque(mascara):
        v, d, e = victorias[mascara].sum(), derrotas[mascara].sum(), empates[mascara].sum()
        return {"victorias": int(v), "derrotas": int(d), "empates": int(e),
                "win_rate": v / max(v + d, 1)}

    return {"p1": bloque(lados), "p2": bloque(~lados), "total": bloque(np.ones_like(lados))}


def evaluar(jugador_a, jugador_b, env, pasos=256):
    """Juega `pasos` pasos en todas las partidas del entorno (que se reinicia)."""
    estado = Estado(env)
    victorias = np.zeros(estado.num_envs)
    derrotas = np.zeros(estado.num_envs)
    empates = np.zeros(estado.num_envs)

    for _ in range(pasos):
        acciones_a = jugador_a.actuar(estado.next_obs, estado.next_done)
        acciones_b = jugador_b.actuar(estado.next_obs_rival, estado.next_done)
        recompensa, terminadas = estado.paso(acciones_a, acciones_b)
        victorias += recompensa > 0
        derrotas += recompensa < 0
        empates += terminadas & (recompensa == 0)

    return resumir(victorias, derrotas, empates, estado.lados)


def linea_resumen(resumen):
    t = resumen["total"]
    partidas = t["victorias"] + t["derrotas"] + t["empates"]
    return (f"como p1 {resumen['p1']['win_rate']:6.1%} | "
            f"como p2 {resumen['p2']['win_rate']:6.1%} | "
            f"total {t['win_rate']:6.1%} | "
            f"{partidas} partidas ({t['empates']} empates)")
