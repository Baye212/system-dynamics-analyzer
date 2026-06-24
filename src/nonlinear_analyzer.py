"""
Module nonlinear_analyzer.py : Analyse des systèmes dynamiques non-linéaires 2D
avec recherche de points fixes multiples et linéarisation locale.
"""
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Callable
from src.stability import classify_eigenvalues

def analyze_nonlinear_system(f1_str: str, f2_str: str, domain: Tuple[float, float, float, float]) -> Dict:
    """
    Analyse complète d'un système non-linéaire :
    1. Parsing des équations sous forme symbolique SymPy
    2. Recherche de tous les points fixes réels dans le domaine
    3. Calcul de la Jacobienne symbolique
    4. Linéarisation locale autour de chaque point fixe (valeurs propres, classification)
    
    Args:
        f1_str: Expression de dx1/dt sous forme de chaîne (e.g. 'x1 - x2 + x1**2')
        f2_str: Expression de dx2/dt sous forme de chaîne (e.g. '-x1 + 2*x2')
        domain: Domaine d'étude (x1_min, x1_max, x2_min, x2_max)
        
    Returns:
        Un dictionnaire d'analyse contenant les points fixes et leurs stabilités locales.
    """
    x1, x2 = sp.symbols('x1 x2')
    
    # 1. Parsing sécurisé
    expr1 = sp.sympify(f1_str.replace('x', 'x1').replace('y', 'x2').replace('x11', 'x1').replace('x22', 'x2'))
    # Wait, some formulas might already have x1 or x2, let's normalize:
    # First convert 'x1' -> 'x1', 'x2' -> 'x2', but if they are 'x' and 'y':
    # Let's write a robust replacement helper
    expr1 = normalize_variables(f1_str, x1, x2)
    expr2 = normalize_variables(f2_str, x1, x2)
    
    # Création des fonctions numériques associées pour intégration ou autre
    f1_num = sp.lambdify((x1, x2), expr1, 'numpy')
    f2_num = sp.lambdify((x1, x2), expr2, 'numpy')
    
    # 2. Recherche des points fixes
    solutions = []
    try:
        solutions = sp.solve([expr1, expr2], (x1, x2), dict=True)
    except Exception:
        # En cas d'échec de solve analytique, on essaie des approximations ou on continue vide
        pass
        
    # Filtrage des points réels dans le domaine
    x1_min, x1_max, x2_min, x2_max = domain
    fixed_points = []
    
    for sol in solutions:
        try:
            val1 = sol.get(x1, None)
            val2 = sol.get(x2, None)
            if val1 is None or val2 is None:
                continue
            
            # Évaluation numérique complexe
            c1 = complex(val1.evalf())
            c2 = complex(val2.evalf())
            
            # Vérification de la partie imaginaire
            if abs(c1.imag) < 1e-7 and abs(c2.imag) < 1e-7:
                px, py = c1.real, c2.real
                # Vérification des limites du domaine
                if x1_min <= px <= x1_max and x2_min <= py <= x2_max:
                    # Éliminer les doublons numériques
                    if not any(np.allclose([px, py], [opx, opy], atol=1e-5) for opx, opy in fixed_points):
                        fixed_points.append((px, py))
        except Exception:
            continue
            
    # 3. Calcul de la Jacobienne symbolique
    f_mat = sp.Matrix([expr1, expr2])
    J_sym = f_mat.jacobian([x1, x2])
    
    # 4. Linéarisation locale pour chaque point fixe
    points_analysis = []
    for (px, py) in fixed_points:
        try:
            # Évaluation de la Jacobienne au point fixe
            J_eval = J_sym.subs({x1: px, x2: py})
            J_num = np.array(J_eval.tolist(), dtype=float)
            
            # Calcul valeurs propres
            eigvals, eigvecs = np.linalg.eig(J_num)
            
            local_stability = classify_eigenvalues(eigvals)
            
            # Calcul des sous-espaces locaux
            E_s, E_u, E_c = [], [], []
            for i, val in enumerate(eigvals):
                v = eigvecs[:, i].real
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
            
            points_analysis.append({
                'point': (px, py),
                'jacobian': J_num,
                'valeurs_propres': local_stability['valeurs_propres'],
                'classification': local_stability['classification'],
                'nature': local_stability['nature'],
                'eigenspaces': {
                    'E_s_dim': len(E_s), 'E_s_basis': E_s,
                    'E_u_dim': len(E_u), 'E_u_basis': E_u,
                    'E_c_dim': len(E_c), 'E_c_basis': E_c
                }
            })
        except Exception:
            continue
            
    return {
        'expressions': (expr1, expr2),
        'fonctions_numeriques': (f1_num, f2_num),
        'points_fixes': points_analysis
    }

def normalize_variables(expr_str: str, x1: sp.Symbol, x2: sp.Symbol) -> sp.Expr:
    """Normalise l'expression mathématique pour utiliser x1 et x2."""
    s = expr_str.strip()
    # Si l'expression utilise x/y au lieu de x1/x2
    # Pour éviter de remplacer x1 par x11, on remplace de manière intelligente :
    # Si on a x1 et x2, on n'a rien à faire. Si on a x et y, on les transforme.
    if 'x1' not in s and 'x2' not in s:
        # Remplacement de 'x' par 'x1' et 'y' par 'x2'
        # Attention aux fonctions comme exp, sin, etc. qui contiennent 'x' ou 'y'
        # On utilise sympy pour parser directement avec un dictionnaire de symboles
        x, y = sp.symbols('x y')
        expr = sp.sympify(s, locals={'x': x, 'y': y})
        expr = expr.subs({x: x1, y: x2})
    else:
        expr = sp.sympify(s, locals={'x1': x1, 'x2': x2})
    return expr
