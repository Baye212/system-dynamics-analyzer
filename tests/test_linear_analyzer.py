"""
Tests unitaires pour l'analyseur linéaire (linear_analyzer.py).
"""
import numpy as np
import pytest
from src.linear_analyzer import analyze_linear_system

def test_linear_saddle():
    # Matrice avec valeurs propres 3.0 (instable) et -1.0 (stable)
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    res = analyze_linear_system(A)
    
    assert res['nature'] == 'Selle'
    assert len(res['valeurs_propres']) == 2
    assert np.allclose(sorted(res['valeurs_propres']), [-1.0, 3.0])
    
    esp = res['eigenspaces']
    assert esp['E_s_dim'] == 1
    assert esp['E_u_dim'] == 1
    assert esp['E_c_dim'] == 0
    
    # Vérifier que les vecteurs de base sont unitaires
    for v in esp['E_s_basis'] + esp['E_u_basis']:
        assert np.isclose(np.linalg.norm(v), 1.0)

def test_linear_stable_node():
    A = np.array([[-2.0, 0.0], [0.0, -3.0]])
    res = analyze_linear_system(A)
    
    assert res['nature'] == 'Nœud stable'
    esp = res['eigenspaces']
    assert esp['E_s_dim'] == 2
    assert esp['E_u_dim'] == 0
    assert esp['E_c_dim'] == 0
