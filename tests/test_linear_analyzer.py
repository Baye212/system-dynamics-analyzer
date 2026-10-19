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

def _is_basis(vectors):
    return len(vectors) == 2 and abs(np.linalg.det(np.column_stack(vectors))) > 1e-6

def test_focus_basis_is_re_im():
    A = np.array([[-1.0, -2.0], [2.0, -1.0]])
    esp = analyze_linear_system(A)['eigenspaces']
    assert esp['E_s_dim'] == 2
    assert _is_basis(esp['E_s_basis'])

def test_center_basis():
    A = np.array([[0.0, -1.0], [1.0, 0.0]])
    res = analyze_linear_system(A)
    assert res['nature'] == 'Centre'
    assert res['eigenspaces']['E_c_dim'] == 2
    assert _is_basis(res['eigenspaces']['E_c_basis'])

def test_jordan_generalized_eigenvector():
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    res = analyze_linear_system(A)
    assert res['nature'] == 'Nœud impropre (dégénéré) instable'
    v, w = res['eigenspaces']['E_u_basis']
    assert _is_basis([v, w])
    assert np.allclose((A - np.eye(2)) @ v, 0)

def test_real_eigenvalues_have_no_imaginary_part():
    res = analyze_linear_system(np.eye(2))
    assert not np.iscomplexobj(res['valeurs_propres'])
    assert res['nature'] == 'Nœud propre (étoile) instable'
