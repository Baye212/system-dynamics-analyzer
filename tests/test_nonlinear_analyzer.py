"""
Tests unitaires pour l'analyseur non-linéaire (nonlinear_analyzer.py).
"""
import numpy as np
import pytest
from src.nonlinear_analyzer import analyze_nonlinear_system

def test_nonlinear_fixed_points():
    # Système avec deux points fixes réels : (0,0) et (1,1)
    # dx1/dt = x1 - x2**2
    # dx2/dt = -x1 + x2
    domain = (-2.0, 2.0, -2.0, 2.0)
    res = analyze_nonlinear_system("x1 - x2**2", "-x1 + x2", domain)
    
    points = res['points_fixes']
    assert len(points) == 2
    
    coords = [pt['point'] for pt in points]
    # Tri par coordonnées pour faciliter la comparaison
    coords = sorted(coords, key=lambda p: p[0])
    
    assert np.allclose(coords[0], [0.0, 0.0], atol=1e-4)
    assert np.allclose(coords[1], [1.0, 1.0], atol=1e-4)
    
    # Vérification des stabilités locales
    # Jacobienne J = [[1, -2*x2], [-1, 1]]
    # Au point (0,0) : J = [[1, 0], [-1, 1]] => val_propres = 1.0, 1.0 (Nœud instable)
    # Au point (1,1) : J = [[1, -2], [-1, 1]] => trace = 2, det = 1 - 2 = -1 => Selle
    for pt in points:
        px, py = pt['point']
        if np.allclose([px, py], [0.0, 0.0]):
            assert pt['nature'] == 'Nœud instable'
        elif np.allclose([px, py], [1.0, 1.0]):
            assert pt['nature'] == 'Selle'
