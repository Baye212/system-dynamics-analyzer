"""
Module stability.py : Analyse et classification de la stabilité des points fixes
à partir de leurs valeurs propres.
"""
import numpy as np
from typing import Dict

def classify_eigenvalues(eigvals: np.ndarray) -> Dict:
    """
    Classifie la stabilité et détermine la nature d'un point fixe à partir de ses valeurs propres (2D).
    
    Args:
        eigvals: Tableau numpy contenant les valeurs propres.
        
    Returns:
        Un dictionnaire contenant :
            - 'valeurs_propres': Le tableau d'origine
            - 'classification': Une liste des classifications individuelles pour chaque valeur propre
            - 'nature': La nature globale du point fixe (ex: Selle, Nœud stable, Foyer stable, etc.)
    """
    classification = []
    for val in eigvals:
        if np.isclose(val.imag, 0):
            # Valeur propre réelle
            real_val = val.real
            if real_val < 0:
                cls = 'Stable'
            elif real_val > 0:
                cls = 'Instable'
            else:
                cls = 'Centre'
        else:
            # Valeur propre complexe
            if val.real < 0:
                cls = 'Stable (spirale)'
            elif val.real > 0:
                cls = 'Instable (spirale)'
            else:
                cls = 'Centre (cycle limite)'
        classification.append(cls)
        
    # Nature du point fixe
    if np.all(np.isreal(eigvals)):
        if np.all(eigvals < 0):
            nat = 'Nœud stable'
        elif np.all(eigvals > 0):
            nat = 'Nœud instable'
        elif np.any(eigvals < 0) and np.any(eigvals > 0):
            nat = 'Selle'
        else:
            nat = 'Cas dégénéré'
    else:
        if np.all(np.real(eigvals) < 0):
            nat = 'Foyer stable'
        elif np.all(np.real(eigvals) > 0):
            nat = 'Foyer instable'
        elif np.all(np.real(eigvals) == 0):
            nat = 'Centre'
        else:
            nat = 'Foyer-selle'
            
    return {
        'valeurs_propres': eigvals,
        'classification': classification,
        'nature': nat
    }
