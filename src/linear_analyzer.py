"""
Module linear_analyzer.py : Analyse complète des systèmes dynamiques linéaires.
"""
import numpy as np
from typing import Dict
from src.stability import classify_eigenvalues

def analyze_linear_system(A: np.ndarray) -> Dict:
    """
    Analyse complète d'un système linéaire : valeurs propres, vecteurs propres,
    classification et sous-espaces propres.
    
    Args:
        A: Matrice 2x2 du système linéaire.
        
    Returns:
        Un dictionnaire contenant :
            - 'valeurs_propres': Les valeurs propres
            - 'vecteurs_propres': Les vecteurs propres (matrice)
            - 'classification': Liste des classifications des valeurs propres
            - 'nature': Nature du point fixe
            - 'eigenspaces': Sous-espaces stables, instables, centre avec dimensions et bases
    """
    eigvals, eigvecs = np.linalg.eig(A)
    analysis = classify_eigenvalues(eigvals)
    analysis['vecteurs_propres'] = eigvecs
    
    # Séparation et calcul des bases des sous-espaces E_s, E_u, E_c
    E_s, E_u, E_c = [], [], []
    for i, val in enumerate(eigvals):
        v = eigvecs[:, i].real
        # Normalisation du vecteur pour l'affichage
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
            
        if np.isclose(val.imag, 0):
            if val.real < 0:
                E_s.append(v)
            elif val.real > 0:
                E_u.append(v)
            else:
                E_c.append(v)
        else:
            if np.isclose(val.real, 0):
                E_c.append(v)
            elif val.real < 0:
                E_s.append(v)
            else:
                E_u.append(v)
                
    analysis['eigenspaces'] = {
        'E_s_dim': len(E_s), 'E_s_basis': E_s,
        'E_u_dim': len(E_u), 'E_u_basis': E_u,
        'E_c_dim': len(E_c), 'E_c_basis': E_c
    }
    return analysis
