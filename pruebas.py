"""Pruebas rápidas del entorno, la red, PPO y los modelos guardados.

Uso:
    python pruebas.py

Si el entrenamiento se comporta raro, córrelo para descartar que el problema
esté en el entorno, en la red, en la recolección o al guardar/cargar modelos.
"""
import os
import tempfile
import time
import traceback

import torch

from agente.red import RedActorCritico
from algoritmos.dreamer import ConfigDreamer, Dreamer
from algoritmos.ppo import PPO
from comun.entorno import crear_entorno
from comun.estado import Estado
from comun.evaluacion import evaluar, linea_resumen
from comun.jugadores import JugadorAleatorio, cargar_jugador
from comun.modelos import cargar_modelo, guardar_modelo

PASOS = 1000


def probar_red(estado):
    """Formas de salida de la red y entropía inicial (debería ser ≈ ln 9 = 2.197)."""
    print("\n--- Red ---")
    red = RedActorCritico(estado.n_obs, estado.n_acciones)
    with torch.no_grad():
        accion, log_prob, entropia, valor = red.obtener_accion_y_valor(estado.next_obs)
    print("Forma acción:  ", tuple(accion.shape))
    print("Forma log_prob:", tuple(log_prob.shape))
    print("Forma entropía:", tuple(entropia.shape))
    print("Forma valor:   ", tuple(valor.shape))
    print(f"Acciones entre {accion.min().item()} y {accion.max().item()}")
    print(f"Entropía promedio: {entropia.mean().item():.4f}")


def probar_entorno(env):
    """Aleatorio contra aleatorio: el win rate debería rondar el 50% en ambos lados."""
    print("\n--- Entorno (aleatorio vs aleatorio) ---")
    n_acciones = env.action_space("p1").n
    aleatorio = JugadorAleatorio(n_acciones)
    inicio = time.time()
    resumen = evaluar(aleatorio, aleatorio, env, PASOS)
    duracion = time.time() - inicio
    print(linea_resumen(resumen))
    print(f"Velocidad: {PASOS * env.num_envs / duracion:,.0f} pasos/s")


def probar_recoleccion(env):
    """Una recolección de PPO contra el aleatorio y el cálculo de ventajas."""
    print("\n--- Recolección y ventajas (PPO vs aleatorio) ---")
    estado = Estado(env)
    ppo = PPO(estado.n_obs, estado.n_acciones, estado.num_envs)
    datos = ppo.recolectar(estado, JugadorAleatorio(estado.n_acciones))
    ppo.calcular_ventajas(estado, datos)

    for nombre, tabla in datos.items():
        print(f"{nombre:11s} {tuple(tabla.shape)}")

    ventajas, recompensas = datos["advantages"], datos["rewards"]
    print("Ventajas válidas:", torch.isfinite(ventajas).all().item())
    # Con la red sin entrenar, el crítico da valores al azar (por ejemplo, +1 en todos lados),
    # así que el signo de cada ventaja no dice mucho. Lo que sí debe cumplirse es que la
    # ventaja al ganar sea mayor que al perder, por una diferencia de aproximadamente 2.
    v_gana = ventajas[recompensas > 0].mean().item()
    v_pierde = ventajas[recompensas < 0].mean().item()
    print(f"Victorias: {(recompensas > 0).sum().item()}  (ventaja media {v_gana:+.2f})")
    print(f"Derrotas:  {(recompensas < 0).sum().item()}  (ventaja media {v_pierde:+.2f})")
    print(f"Diferencia: {v_gana - v_pierde:+.2f}  (debería rondar +2)")


def probar_guardar_y_cargar(estado):
    """Un modelo guardado y vuelto a cargar debe dar exactamente las mismas salidas."""
    print("\n--- Guardar y cargar modelos ---")
    red = RedActorCritico(estado.n_obs, estado.n_acciones)
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "prueba.pt")
        guardar_modelo(ruta, red, algoritmo="ppo", lados="ambos")
        cargada, metadatos = cargar_modelo(ruta)
    with torch.no_grad():
        iguales = torch.allclose(red.actor(estado.next_obs), cargada.actor(estado.next_obs))
    print("Metadatos:", metadatos)
    print("Mismas salidas después de cargar:", iguales)


def probar_dreamer(env):
    """Una iteración corta de Dreamer, y que el modelo guardado juegue igual al cargarlo."""
    print("\n--- Dreamer (iteración corta vs aleatorio) ---")
    estado = Estado(env)
    config = ConfigDreamer(pasos_por_iteracion=40, largo_secuencia=32, batch=8,
                           actualizaciones_por_iteracion=3)
    dreamer = Dreamer(estado.n_obs, estado.n_acciones, estado.num_envs, config)
    inicio = time.time()
    datos, metricas = dreamer.iteracion(estado, JugadorAleatorio(estado.n_acciones))
    print(f"Buffer: {dreamer.repeticion.tamano()} pasos por partida, "
          f"recompensas {tuple(datos['rewards'].shape)}")
    for clave, valor in metricas.items():
        print(f"   {clave:20s} {valor:+.4f}")
    print("Métricas válidas:", all(torch.isfinite(torch.tensor(v)) for v in metricas.values()))
    print(f"Tiempo: {time.time() - inicio:.1f} s")

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "dreamer.pt")
        guardar_modelo(ruta, dreamer.red, **dreamer.metadatos(), lados="ambos")
        cargado = cargar_jugador(ruta, estado.n_acciones, determinista=True)
    original = dreamer.como_jugador()
    original.determinista = True
    obs, nuevas = estado.next_obs, torch.ones(estado.num_envs)
    torch.manual_seed(0)
    a = original.actuar(obs, nuevas)
    torch.manual_seed(0)
    b = cargado.actuar(obs, nuevas)
    print("Mismas acciones después de cargar:", bool((a == b).all()))


def main():
    torch.set_num_threads(min(4, max(1, (os.cpu_count() or 2) // 2 - 1)))
    env = crear_entorno()
    try:
        print("Acciones:", env.action_space("p1"))
        estado = Estado(env)
        probar_red(estado)
        probar_guardar_y_cargar(estado)
        probar_recoleccion(env)
        probar_entorno(env)
        probar_dreamer(env)
    except Exception:
        print("Falló con este error:")
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
