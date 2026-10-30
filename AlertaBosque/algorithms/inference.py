"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize


def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)

    ocultas = []
    for variable in network.variables:
      if variable != query and variable not in evidence:
        ocultas.append(variable)
        pesos = {0: 0.0, 1: 0.0}
        cantidad = 2 ** len(ocultas)
        for valor in (0, 1):
          posible = True
          if query in evidence and evidence[query] != valor:
            posible = False
            if posible:
              for numero in range(cantidad):
                caso = {}
                for nombre in evidence:
                  caso[nombre] = evidence[nombre]
                  caso[query] = valor
                  for i in range(len(ocultas)):
                    caso[ocultas[i]] = (numero // 2 ** i) % 2
                    producto = 1.0
                    for variable in network.variables:
                      producto = producto * network.probability(variable, caso[variable], caso)
                      pesos[valor] += producto
        if pesos[0] + pesos[1] == 0:
            raise ValueError("Evidencia imposible")
        return normalize(pesos)
