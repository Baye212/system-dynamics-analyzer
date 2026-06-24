"""
Tests pour le module phase_plotter.py.
"""
import numpy as np
import pytest
from src.dynamics_analyzer import DynamicsAnalyzer

def test_phase_plotter_no_crash():
    # Créer un analyseur linéaire stable
    A = np.array([[-2.0, 0.0], [0.0, -2.0]])
    domain = (-3, 3, -3, 3)
    analyzer = DynamicsAnalyzer(A, domain)
    
    # Vérifier que le tracé des isoclines s'exécute sans erreur
    try:
        analyzer.plot_isoclines(plot=False)
        assert True
    except Exception as e:
        pytest.fail(f"plot_isoclines a planté avec l'erreur : {e}")
        
    # Vérifier que le tracé du portrait de phase s'exécute sans erreur
    try:
        analyzer.plot_final_phase_portrait(plot=False)
        assert True
    except Exception as e:
        pytest.fail(f"plot_final_phase_portrait a planté avec l'erreur : {e}")
