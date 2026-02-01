"""
Exercice 3.1: Pattern Décorateur
Implémentation de décorateurs pour le logging et le timing.
Uniquement les fonctionnalités demandées.
"""

import time
import functools
import logging

# Configuration basique du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def timing_decorator(func):
    """Décorateur mesurant le temps d'exécution d'une fonction."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"[TIMING] {func.__name__} exécuté en {duration:.4f} secondes")
        return result
    return wrapper


def logging_decorator(func):
    """Décorateur ajoutant des logs avant et après exécution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Début de l'exécution de: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Fin de l'exécution de: {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Erreur dans {func.__name__}: {str(e)}")
            raise e
    return wrapper


