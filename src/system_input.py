"""
Module system_input.py
Permet la saisie et la validation de systèmes dynamiques linéaires et non-linéaires.
"""
import numpy as np
import re
import sympy as sp
from typing import Callable, Tuple, Union

def safe_lambdify(expr_str: str) -> Callable:
    """
    Parse de manière sécurisée une expression mathématique et la rend exécutable (numérique).
    Supporte les variables x/y et x1/x2.
    """
    x1, x2 = sp.symbols('x1 x2')
    norm_expr = expr_str.strip()
    if 'x1' not in norm_expr and 'x2' not in norm_expr:
        x, y = sp.symbols('x y')
        expr = sp.sympify(norm_expr, locals={'x': x, 'y': y})
        expr = expr.subs({x: x1, y: x2})
    else:
        expr = sp.sympify(norm_expr, locals={'x1': x1, 'x2': x2})
    return sp.lambdify((x1, x2), expr, 'numpy')

def input_linear_system() -> np.ndarray:
    print("Entrer la matrice A (2x2) ligne par ligne, séparée par des espaces (ex: 1 2):")
    A = []
    for i in range(2):
        while True:
            try:
                row = input(f"Ligne {i+1}: ").strip()
                vals = [float(x) for x in row.split()]
                if len(vals) != 2:
                    raise ValueError
                A.append(vals)
                break
            except ValueError:
                print("Entrée invalide. Veuillez entrer exactement 2 nombres réels.")
    return np.array(A)

def input_nonlinear_system() -> Tuple[str, str]:
    print("Entrer f₁(x₁, x₂) (ex: 'x1 - x2 + x1**2'):")
    f1_str = input("f₁(x₁, x₂): ")
    print("Entrer f₂(x₁, x₂) (ex: '-x1 + 2*x2'):")
    f2_str = input("f₂(x₁, x₂): ")
    return f1_str, f2_str

def parse_equations(eq1: str, eq2: str) -> np.ndarray:
    # Ex: eq1 = "x' = 2*x + 3*y", eq2 = "y' = -x + 4*y"
    pattern = r"([+-]?\d*\.?\d*)\*?x"  # matches ax
    pattern2 = r"([+-]?\d*\.?\d*)\*?y"  # matches by
    def get_coeff(eq, var):
        m = re.search(pattern if var == 'x' else pattern2, eq.replace(' ', ''))
        if m:
            c = m.group(1)
            return float(c) if c not in ('', '+', '-') else (1.0 if c in ('', '+') else -1.0)
        return 0.0
    a = get_coeff(eq1, 'x')
    b = get_coeff(eq1, 'y')
    c = get_coeff(eq2, 'x')
    d = get_coeff(eq2, 'y')
    return np.array([[a, b], [c, d]])

def input_domain() -> Tuple[float, float, float, float]:
    print("Définir le domaine d'étude [x_min, x_max] × [y_min, y_max] (ex: -5 5 -5 5):")
    while True:
        try:
            vals = [float(x) for x in input("Domaine: ").split()]
            if len(vals) != 4:
                raise ValueError
            return tuple(vals)
        except ValueError:
            print("Entrée invalide. Veuillez entrer 4 nombres réels.")

def example_linear():
    return np.array([[1, 2], [-3, 4]])

def example_nonlinear():
    f1 = safe_lambdify("x1 - x2 + x1**2")
    f2 = safe_lambdify("-x1 + 2*x2")
    return f1, f2
