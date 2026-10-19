"""
Tests unitaires pour la classification de stabilité (stability.py).
"""
import numpy as np
import pytest
from src.stability import classify_eigenvalues

def test_saddle():
    eigvals = np.array([-2.0, 3.0])
    res = classify_eigenvalues(eigvals)
    assert res['nature'] == 'Selle'
    assert 'Stable' in res['classification']
    assert 'Instable' in res['classification']

def test_stable_node():
    eigvals = np.array([-1.0, -4.0])
    res = classify_eigenvalues(eigvals)
    assert res['nature'] == 'Nœud stable'
    assert all(c == 'Stable' for c in res['classification'])

def test_unstable_node():
    eigvals = np.array([2.5, 1.2])
    res = classify_eigenvalues(eigvals)
    assert res['nature'] == 'Nœud instable'
    assert all(c == 'Instable' for c in res['classification'])

def test_stable_focus():
    eigvals = np.array([-1.0 + 2.0j, -1.0 - 2.0j])
    res = classify_eigenvalues(eigvals)
    assert res['nature'] == 'Foyer stable'
    assert all(c == 'Stable (spirale)' for c in res['classification'])

def test_unstable_focus():
    eigvals = np.array([0.5 + 3.0j, 0.5 - 3.0j])
    res = classify_eigenvalues(eigvals)
    assert res['nature'] == 'Foyer instable'
    assert all(c == 'Instable (spirale)' for c in res['classification'])

def test_center():
    eigvals = np.array([2.0j, -2.0j])
    res = classify_eigenvalues(eigvals)
    assert res['nature'] == 'Centre'
    assert all(c == 'Centre (imaginaire pur)' for c in res['classification'])

def test_numerical_noise_center():
    # Partie réelle résiduelle due aux arrondis : doit rester un centre
    eigvals = np.array([1e-16 + 1.0j, 1e-16 - 1.0j])
    assert classify_eigenvalues(eigvals)['nature'] == 'Centre'

def test_zero_eigenvalue():
    res = classify_eigenvalues(np.array([0.0, -1.0]))
    assert res['nature'] == 'Cas dégénéré (valeur propre nulle)'
    assert res['classification'][0] == 'Neutre (λ = 0)'

def test_star_and_improper_node():
    eig = np.array([1.0, 1.0])
    assert classify_eigenvalues(eig, np.eye(2))['nature'] == 'Nœud propre (étoile) instable'
    J = np.array([[1.0, 1.0], [0.0, 1.0]])
    assert classify_eigenvalues(eig, J)['nature'] == 'Nœud impropre (dégénéré) instable'
