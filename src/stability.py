"""
Module stability.py : Analyse et classification de la stabilité des points fixes
à partir de leurs valeurs propres.
"""
import numpy as np
from typing import Dict, Optional

# Tolérance numérique : en dessous, une quantité est considérée comme nulle
TOL = 1e-9


def _sign(x: float, tol: float = TOL) -> int:
    """Signe de x avec tolérance numérique (0 si |x| <= tol)."""
    if abs(x) <= tol:
        return 0
    return 1 if x > 0 else -1


def clean_eigenvalues(eigvals: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Supprime les parties imaginaires résiduelles (bruit numérique) si toutes sont négligeables."""
    eigvals = np.asarray(eigvals)
    if np.iscomplexobj(eigvals) and np.all(np.abs(eigvals.imag) <= tol):
        return eigvals.real
    return eigvals


def classify_eigenvalues(eigvals: np.ndarray, A: Optional[np.ndarray] = None) -> Dict:
    """
    Classifie la stabilité et détermine la nature d'un point fixe à partir de ses valeurs propres (2D).

    Args:
        eigvals: Tableau numpy contenant les valeurs propres.
        A: Matrice du système (optionnelle). Permet de distinguer, pour une valeur propre
           double, le nœud propre (étoile, A = λI) du nœud impropre (dégénéré, non diagonalisable).

    Returns:
        Un dictionnaire contenant :
            - 'valeurs_propres': Les valeurs propres (réelles si les parties imaginaires sont négligeables)
            - 'classification': Une liste des classifications individuelles pour chaque valeur propre
            - 'nature': La nature globale du point fixe (ex: Selle, Nœud stable, Foyer stable, etc.)
    """
    eigvals = clean_eigenvalues(eigvals)
    is_real = not np.iscomplexobj(eigvals)

    classification = []
    for val in eigvals:
        s = _sign(np.real(val))
        if is_real:
            cls = {-1: 'Stable', 1: 'Instable', 0: 'Neutre (λ = 0)'}[s]
        else:
            cls = {-1: 'Stable (spirale)', 1: 'Instable (spirale)', 0: 'Centre (imaginaire pur)'}[s]
        classification.append(cls)

    # Nature du point fixe
    signs = [_sign(np.real(v)) for v in eigvals]
    if is_real:
        if 0 in signs:
            nat = 'Cas dégénéré (valeur propre nulle)'
        elif signs[0] != signs[1]:
            nat = 'Selle'
        else:
            stab = 'stable' if signs[0] < 0 else 'instable'
            l1, l2 = eigvals
            if abs(l1 - l2) <= TOL * max(1.0, abs(l1)):
                # Valeur propre double : étoile si A = λI, sinon nœud dégénéré
                if A is not None and not np.allclose(A, l1 * np.eye(2), atol=TOL):
                    nat = f'Nœud impropre (dégénéré) {stab}'
                else:
                    nat = f'Nœud propre (étoile) {stab}'
            else:
                nat = f'Nœud {stab}'
    else:
        if signs[0] < 0:
            nat = 'Foyer stable'
        elif signs[0] > 0:
            nat = 'Foyer instable'
        else:
            nat = 'Centre'

    return {
        'valeurs_propres': eigvals,
        'classification': classification,
        'nature': nat
    }
