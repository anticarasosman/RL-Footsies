"""Dibuja al agente entrenado peleando contra un rival aleatorio.

La versión con ventana de FootsiesGym para Linux no tiene gráficos, así que este
script juega en modo headless, guarda el estado de cada paso y dibuja la pelea
con matplotlib: dos personajes, sus acciones, su guardia y el ataque especial.

Uso:
    python visualizar.py                       # 3 partidas, ventana + visualizacion.gif
    python visualizar.py --partidas 5
    python visualizar.py --determinista        # acción más probable, sin muestrear
    python visualizar.py --sin-ventana         # solo guarda el GIF
    python visualizar.py --salida pelea.gif
    python visualizar.py --modelo otro.pt

Requiere matplotlib y pillow:  pip install matplotlib pillow
"""
import argparse

import matplotlib

import numpy as np
import torch

# Cada paso son 4 cuadros del juego a 60 cuadros por segundo → 15 pasos por segundo.
PASOS_POR_SEGUNDO = 15
LIMITE_ESCENARIO = 4.0   # el escenario va de -4 a 4

# Nombres de las acciones que elige el agente (índices 0 a 8 del entorno).
ACCIONES_AGENTE = [
    "nada", "atrás", "adelante", "ataque", "atrás + ataque", "adelante + ataque",
    "carga especial", "adelante + carga", "atrás + carga",
]

# Estado del personaje en el juego (current_action_id).
ESTADOS_JUEGO = {
    0: "quieto", 1: "avanza", 2: "retrocede", 10: "dash adelante", 11: "dash atrás",
    100: "ataque", 105: "ataque bajo", 110: "ESPECIAL", 115: "ESPECIAL",
    200: "recibe golpe", 301: "bloquea", 305: "bloquea", 306: "bloquea",
    310: "guardia rota", 350: "bloquea", 500: "derrotado", 510: "gana",
}
ATAQUES_NORMALES = {100, 105}
ATAQUES_ESPECIALES = {110, 115}
BLOQUEOS = {301, 305, 306, 350}


def leer_argumentos():
    parser = argparse.ArgumentParser(description="Dibujar al agente peleando.")
    parser.add_argument("--modelo", default="modelo.pt")
    parser.add_argument("--partidas", type=int, default=3)
    parser.add_argument("--determinista", action="store_true",
                        help="Elegir siempre la acción más probable en vez de muestrear.")
    parser.add_argument("--salida", default="visualizacion.gif",
                        help="Archivo donde guardar la animación ('' para no guardar).")
    parser.add_argument("--sin-ventana", action="store_true",
                        help="No mostrar la animación en pantalla, solo guardarla.")
    return parser.parse_args()


# ── Jugar y grabar ────────────────────────────────────────────────────


def estado_jugador(p):
    """Copia lo que interesa dibujar del estado de un jugador."""
    return {
        "x": float(p.player_position_x),
        "accion_id": int(p.current_action_id),
        "guardia": int(p.guard_health),
        "muerto": bool(p.is_dead),
        "mira_derecha": bool(p.is_face_right),
        "especial": float(p.special_attack_progress),
    }


def elegir_accion(red, obs_p1, determinista):
    obs_tensor = torch.tensor(obs_p1, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if determinista:
            accion = red.actor(obs_tensor).argmax(dim=1)
        else:
            accion, _, _, _ = red.obtener_accion_y_valor(obs_tensor)
    return int(accion.item())


def jugar_y_grabar(ruta_modelo, n_partidas, determinista):
    """Juega n_partidas y devuelve una lista de cuadros para dibujar."""
    import footsiesgym
    from agente.red import RedActorCritico

    config = {"use_special_charge_action": True}
    env = footsiesgym.make(config=config, platform="linux")
    cuadros = []
    victorias = 0
    try:
        obs, infos = env.reset()
        red = RedActorCritico(obs["p1"].shape[0], env.action_space("p1").n)
        red.load_state_dict(torch.load(ruta_modelo))
        red.eval()
        n_acciones = env.action_space("p1").n

        for partida in range(1, n_partidas + 1):
            paso = 0
            terminada = False
            while not terminada:
                accion_p1 = elegir_accion(red, obs["p1"], determinista)
                accion_p2 = int(np.random.randint(0, n_acciones))
                obs, rew, term, trunc, infos = env.step({"p1": accion_p1, "p2": accion_p2})
                paso += 1
                terminada = bool(term["p1"] or trunc["p1"])

                resultado = None
                if terminada:
                    if rew["p1"] > 0:
                        victorias += 1
                        resultado = "GANA EL AGENTE"
                    elif rew["p1"] < 0:
                        resultado = "GANA EL ALEATORIO"
                    else:
                        resultado = "EMPATE POR TIEMPO"
                    print(f"Partida {partida}: {resultado.lower()} en {paso} pasos")

                estado = env.last_game_state
                cuadros.append({
                    "p1": estado_jugador(estado.player1),
                    "p2": estado_jugador(estado.player2),
                    "accion_p1": accion_p1,
                    "partida": partida,
                    "paso": paso,
                    "marcador": victorias,
                    "resultado": resultado,
                })

            # Repetir el último cuadro un segundo para que se lea el resultado.
            cuadros.extend([cuadros[-1]] * PASOS_POR_SEGUNDO)
            obs, infos = env.reset()
    finally:
        env.close()

    print(f"\nEl agente ganó {victorias} de {n_partidas} partidas.")
    return cuadros


# ── Dibujar ───────────────────────────────────────────────────────────


def dibujar_jugador(ax, j, color):
    """El cuerpo del personaje, su ataque o su bloqueo."""
    from matplotlib.patches import Rectangle

    ancho, alto = 0.5, 1.6
    direccion = 1 if j["mira_derecha"] else -1
    aid = j["accion_id"]

    if j["muerto"] or aid == 500:
        ax.add_patch(Rectangle((j["x"] - alto / 2, 0), alto, ancho * 0.6,
                               color=color, alpha=0.5))
    else:
        relleno = "#f2a541" if aid in (200, 310) else color
        ax.add_patch(Rectangle((j["x"] - ancho / 2, 0), ancho, alto, color=relleno))

        frente = j["x"] + direccion * ancho / 2
        if aid in ATAQUES_NORMALES:
            largo = 0.7
            x0 = frente if direccion > 0 else frente - largo
            ax.add_patch(Rectangle((x0, 1.05), largo, 0.22, color=color))
        elif aid in ATAQUES_ESPECIALES:
            largo = 1.1
            x0 = frente if direccion > 0 else frente - largo
            ax.add_patch(Rectangle((x0, 0.85), largo, 0.5, color=color, alpha=0.85))
        elif aid in BLOQUEOS:
            xs = frente + direccion * 0.08
            ax.plot([xs, xs], [0.15, 1.5], color="#444444", linewidth=4)


def dibujar_marcador(ax, j, color, etiqueta, x0, alinear):
    """Nombre, estado, guardia y carga del especial, fijos en una esquina."""
    from matplotlib.patches import Rectangle

    nombre_estado = ESTADOS_JUEGO.get(j["accion_id"], str(j["accion_id"]))
    signo = 1 if alinear == "left" else -1
    ax.text(x0, 3.25, etiqueta, ha=alinear, fontsize=10, fontweight="bold", color=color)
    ax.text(x0, 2.9, nombre_estado, ha=alinear, fontsize=9, color="#333333")

    # Guardia: 3 cuadritos.
    for i in range(3):
        lleno = i < j["guardia"]
        x = x0 + signo * i * 0.24 - (0.18 if signo < 0 else 0)
        ax.add_patch(Rectangle((x, 2.55), 0.18, 0.15, color=color if lleno else "#dddddd"))

    # Carga del especial: barra que se llena.
    largo = 0.7
    x_barra = x0 if signo > 0 else x0 - largo
    ax.add_patch(Rectangle((x_barra, 2.3), largo, 0.1, color="#eeeeee"))
    if j["especial"] > 0:
        relleno = largo * min(j["especial"], 1.0)
        x_relleno = x0 if signo > 0 else x0 - relleno
        ax.add_patch(Rectangle((x_relleno, 2.3), relleno, 0.1, color="#9b59b6"))


def dibujar_cuadro(ax, c):
    ax.clear()
    ax.set_xlim(-LIMITE_ESCENARIO - 0.8, LIMITE_ESCENARIO + 0.8)
    ax.set_ylim(-0.6, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-LIMITE_ESCENARIO - 0.5, LIMITE_ESCENARIO + 0.5], [0, 0], color="#888888", linewidth=2)

    azul, rojo = "#2e6fd8", "#d64545"
    borde = LIMITE_ESCENARIO + 0.7
    dibujar_jugador(ax, c["p1"], azul)
    dibujar_jugador(ax, c["p2"], rojo)
    dibujar_marcador(ax, c["p1"], azul, "Agente (p1)", -borde, "left")
    dibujar_marcador(ax, c["p2"], rojo, "Aleatorio (p2)", borde, "right")

    ax.text(0, 3.25, f"Partida {c['partida']}  ·  paso {c['paso']}",
            ha="center", fontsize=9, color="#333333")
    ax.text(0, 2.9, f"victorias del agente: {c['marcador']}",
            ha="center", fontsize=9, color="#333333")
    ax.text(-borde, -0.5, f"El agente eligió: {ACCIONES_AGENTE[c['accion_p1']]}",
            fontsize=9, color=azul)
    if c["resultado"]:
        ax.text(0, 2.2, c["resultado"], ha="center", fontsize=14, fontweight="bold",
                color=azul if "AGENTE" in c["resultado"] else rojo)


def animar(cuadros, salida, mostrar):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    animacion = FuncAnimation(fig, lambda i: dibujar_cuadro(ax, cuadros[i]),
                              frames=len(cuadros), interval=1000 / PASOS_POR_SEGUNDO,
                              repeat=False)

    if salida:
        print(f"Guardando {len(cuadros)} cuadros en {salida} (puede tardar un poco)...")
        animacion.save(salida, writer=PillowWriter(fps=PASOS_POR_SEGUNDO), dpi=80)
        print(f"Animación guardada en {salida}")
    if mostrar:
        plt.show()
    plt.close(fig)


def main():
    args = leer_argumentos()
    if args.sin_ventana:
        matplotlib.use("Agg")
    cuadros = jugar_y_grabar(args.modelo, args.partidas, args.determinista)
    animar(cuadros, args.salida, mostrar=not args.sin_ventana)


if __name__ == "__main__":
    main()
