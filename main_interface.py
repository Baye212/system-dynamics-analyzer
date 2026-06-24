"""
main_interface.py : Interface interactive étape par étape pour l'analyse qualitative de systèmes dynamiques.
"""
import sys
import numpy as np
from src import system_input
from src.dynamics_analyzer import DynamicsAnalyzer

def step(msg):
    input(f"\n--- {msg} (Entrée pour continuer) ---\n")

def main():
    print("=== Analyse Qualitative de Systèmes Dynamiques ===")
    print("1. Système linéaire\n2. Système non-linéaire\n3. Exemple prédéfini")
    choix = input("Choix: ").strip()
    if choix == '1':
        A = system_input.input_linear_system()
        domain = system_input.input_domain()
        analyzer = DynamicsAnalyzer(A, domain)
    elif choix == '2':
        f1, f2 = system_input.input_nonlinear_system()
        domain = system_input.input_domain()
        analyzer = DynamicsAnalyzer((f1, f2), domain)
    else:
        print("Exemples:\n1. Nœud stable\n2. Selle\n3. Centre\n4. Spirale stable (non-linéaire)\n5. Selle non-linéaire")
        ex = input("Numéro exemple: ").strip()
        if ex == '1':
            A = np.array([[-2, 0], [0, -1]])
            domain = (-5,5,-5,5)
            analyzer = DynamicsAnalyzer(A, domain)
        elif ex == '2':
            A = np.array([[1, 0], [0, -2]])
            domain = (-5,5,-5,5)
            analyzer = DynamicsAnalyzer(A, domain)
        elif ex == '3':
            A = np.array([[0, -1], [1, 0]])
            domain = (-5,5,-5,5)
            analyzer = DynamicsAnalyzer(A, domain)
        elif ex == '4':
            f1 = "-x1 - x2 + 0.1*x1**2"
            f2 = "x1 - x2"
            domain = (-5,5,-5,5)
            analyzer = DynamicsAnalyzer((f1, f2), domain)
        elif ex == '5':
            f1 = "x1 - x2**2"
            f2 = "-x1 + x2"
            domain = (-5,5,-5,5)
            analyzer = DynamicsAnalyzer((f1, f2), domain)
        else:
            print("Exemple inconnu.")
            sys.exit(1)
    # Étape 2: Valeurs propres et Étape 3: Sous-espaces
    if analyzer.is_linear:
        step("Étape 2: Valeurs propres + classification")
        try:
            eig = analyzer.analyze_eigenvalues()
            print("Valeurs propres:", eig['valeurs_propres'])
            print("Classification:", eig['classification'])
            print("Nature:", eig['nature'])
        except Exception as e:
            print("Erreur:", e)
        # Étape 3: Vecteurs propres et sous-espaces
        step("Étape 3: Vecteurs propres + sous-espaces")
        try:
            esp = analyzer.compute_eigenspaces()
            print("Eₛ (stable): dim=", esp['E_s_dim'], "base:", esp['E_s_basis'])
            print("Eᵤ (instable): dim=", esp['E_u_dim'], "base:", esp['E_u_basis'])
            print("E꜀ (centre): dim=", esp['E_c_dim'], "base:", esp['E_c_basis'])
        except Exception as e:
            print("Erreur:", e)
    else:
        step("Étape 2 & 3: Analyse des Points Fixes (Système Non-Linéaire)")
        if analyzer.fixed_points_data:
            print(f"Nombre de points fixes réels détectés dans le domaine : {len(analyzer.fixed_points_data)}")
            for idx, pt in enumerate(analyzer.fixed_points_data):
                px, py = pt['point']
                print(f"\n--- Point Fixe #{idx+1} : ({px:.3f}, {py:.3f}) ---")
                print(f"Nature locale : {pt['nature']}")
                print(f"Valeurs propres : {pt['valeurs_propres']}")
                print(f"Classification : {pt['classification']}")
                esp = pt['eigenspaces']
                print(f"Sous-espace Stable (Eₛ)   : dim={esp['E_s_dim']}, base={esp['E_s_basis']}")
                print(f"Sous-espace Instable (Eᵤ) : dim={esp['E_u_dim']}, base={esp['E_u_basis']}")
                print(f"Sous-espace Centre (E꜀)   : dim={esp['E_c_dim']}, base={esp['E_c_basis']}")
        else:
            print("Aucun point fixe réel détecté dans le domaine.")
    # Étape 4: Isoclines
    step("Étape 4: Graphique isoclines + orientation")
    analyzer.plot_isoclines()
    # Étape 5: Portrait de phase final
    step("Étape 5: Portrait de phase final intégré")
    analyzer.plot_final_phase_portrait()
    # Export PDF (optionnel)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages('rapport_analyse.pdf') as pdf:
            analyzer.plot_isoclines(plot=False)
            pdf.savefig(plt.gcf())
            plt.close()
            analyzer.plot_final_phase_portrait(plot=False)
            pdf.savefig(plt.gcf())
            plt.close()
        print("Rapport PDF exporté sous rapport_analyse.pdf")
    except Exception as e:
        print("Export PDF échoué:", e)

if __name__ == "__main__":
    main()
