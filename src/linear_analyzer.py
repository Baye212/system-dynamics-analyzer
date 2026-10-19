"""
Module linear_analyzer.py : Analyse complète des systèmes dynamiques linéaires.
"""
import numpy as np
from typing import Dict
from src.stability import classify_eigenvalues, clean_eigenvalues, _sign, TOL


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def compute_eigenspaces(A: np.ndarray) -> Dict:
    """
    Calcule les sous-espaces stable E_s, instable E_u et centre E_c d'une matrice 2x2,
    avec une base réelle pour chacun :
        - valeurs propres réelles distinctes : les vecteurs propres ;
        - valeur propre double non diagonalisable : vecteur propre v + vecteur propre
          généralisé w tel que (A - λI) w = v ;
        - valeurs propres complexes λ = α ± iβ de vecteur propre v : {Re(v), Im(v)}.
    """
    A = np.asarray(A, dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = clean_eigenvalues(eigvals)
    spaces = {-1: [], 1: [], 0: []}

    if np.iscomplexobj(eigvals):
        # Paire conjuguée : un seul vecteur propre suffit, v2 = conj(v1)
        v = eigvecs[:, 0]
        spaces[_sign(eigvals[0].real)] += [_normalize(v.real), _normalize(v.imag)]
    else:
        eigvecs = eigvecs.real
        l1, l2 = eigvals
        double = abs(l1 - l2) <= TOL * max(1.0, abs(l1))
        if double and not np.allclose(A, l1 * np.eye(2), atol=TOL):
            # Bloc de Jordan : un seul vecteur propre, on complète par un vecteur généralisé
            N = A - l1 * np.eye(2)
            v = _normalize(eigvecs[:, 0])
            w = np.linalg.lstsq(N, v, rcond=None)[0]
            spaces[_sign(l1)] += [v, _normalize(w)]
        else:
            for i, val in enumerate(eigvals):
                spaces[_sign(val)].append(_normalize(eigvecs[:, i]))

    E_s, E_u, E_c = spaces[-1], spaces[1], spaces[0]
    return {
        'E_s_dim': len(E_s), 'E_s_basis': E_s,
        'E_u_dim': len(E_u), 'E_u_basis': E_u,
        'E_c_dim': len(E_c), 'E_c_basis': E_c
    }


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
    analysis = classify_eigenvalues(eigvals, A)
    analysis['vecteurs_propres'] = eigvecs
    analysis['eigenspaces'] = compute_eigenspaces(A)
    return analysis
