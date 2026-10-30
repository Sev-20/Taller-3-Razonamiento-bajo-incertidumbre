"""Punto 1. Distribuciones discretas y decisiones de respuesta."""
import math


def normalize(weights: dict) -> dict:
    """Retorna un diccionario nuevo cuyas probabilidades suman uno.

    Entrada: weights = {resultado: peso}.
    Salida: {resultado: probabilidad}, con las mismas claves.

    Tips:
    - Conserve las claves y su orden; no modifique weights.
    - Rechace pesos negativos, no finitos, diccionarios vacíos y suma cero.
    """    
    if not weights: # un diccionario vacío cuenta como falso en Python, así que esto atrapa el caso de entrada vacía.
        raise ValueError("El diccionario de pesos no puede estar vacío")
    
    for clave, peso in weights.items():
        if not math.isfinite(peso): # Valida que el peso sea finito, sino error 
            raise ValueError(f"Peso no finito para {clave}: {peso}")
        if peso < 0: # Valida que el peso no sea negativo, sino error
            raise ValueError(f"Peso negativo para {clave}: {peso}")
        
    total = sum(weights.values()) # Calcula la suma de los pesos
    
    if total == 0: # Si la suma es cero da error 
        raise ValueError("La suma de los pesos es cero, no se puede normalizar")
    
    return {clave: peso / total for clave, peso in weights.items()} # retorna un nuevo diccionario con las probabilidades normalizadas

    


def expected_utility(distribution: dict, utilities: dict) -> float:
    """Calcula sum_s P(s) U(s) para una acción.

    Entradas: distribution = {resultado: probabilidad} y
    utilities = {resultado: utilidad} para una sola acción.
    Salida: un número con la utilidad esperada.

    Tips:
    - distribution ya debe estar normalizada; no normalice utilidades.
    - Ambas tablas deben contener los mismos resultados posibles.
    """
    # Validación de las entradas; no modificar.
    if set(distribution) != set(utilities) or not distribution:
        raise ValueError("Probabilidades y utilidades deben tener las mismas claves")
    if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
        raise ValueError("Distribución inválida")
    if not math.isclose(sum(distribution.values()), 1, abs_tol=1e-9):
        raise ValueError("La distribución debe sumar uno")
    if any(not math.isfinite(u) for u in utilities.values()):
        raise ValueError("Utilidades no finitas")
    
    total = 0.0
    for resultado, prob in distribution.items():
        total += prob * utilities[resultado] # Calcula la utilidad esperada sumando P(s) * U(s) para cada resultado
    return total # Retorna la utilidad esperada


def best_action(distribution: dict, actions: dict) -> tuple[str, dict]:
    """Retorna (nombre de acción, utilidades esperadas por acción).

    Entradas: distribution = {resultado: probabilidad} y
    actions = {acción: {resultado: utilidad}}.
    Salida: (acción elegida, {acción: utilidad esperada}).

    Tips:
    - actions asocia cada acción con su tabla de utilidades.
    - Reutilice expected_utility y conserve la primera acción en los empates.
    """
    # Validación de las acciones; no modificar.
    if not actions:
        raise ValueError("Se requiere al menos una acción")
    
    utilidades_esperadas = {}
    for nombre,tabla in actions.items():
        utilidades_esperadas[nombre] = expected_utility(distribution, tabla) # Calcula la utilidad esperada para cada acción usando expected_utility
        
    mejor = None
    mejor_valor = float('-inf')
    for nombre, eu in utilidades_esperadas.items():
        if mejor is None or eu > mejor_valor: # Encuentra la acción con la mayor utilidad esperada
            mejor = nombre
            mejor_valor = eu
    
    return mejor, utilidades_esperadas # Retorna la mejor acción y el diccionario de utilidades esperadas