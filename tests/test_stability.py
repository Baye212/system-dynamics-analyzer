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
    assert all(c == 'Centre (cycle limite)' for c in res['classification'])
