"""
Classe DynamicsAnalyzer : Façade et coordinateur principal de l'analyse.
Garantit la compatibilité avec l'interface Streamlit et CLI existante.
"""
import numpy as np
from typing import Union, Tuple, Callable, Dict, List
from src.linear_analyzer import analyze_linear_system
from src.nonlinear_analyzer import analyze_nonlinear_system
from src.system_input import safe_lambdify
from src.phase_plotter import plot_isoclines, plot_final_phase_portrait

class DynamicsAnalyzer:
    def __init__(self, system: Union[np.ndarray, Tuple[Union[str, Callable], Union[str, Callable]]], domain: Tuple[float, float, float, float]=(-5,5,-5,5)):
        self.domain = domain
        self.report = {}
        self.is_linear = isinstance(system, np.ndarray)
        
        if self.is_linear:
            self.system = system
            # Lancer l'analyse linéaire
            self.linear_results = analyze_linear_system(self.system)
            # Remplir self.report pour compatibilité descendante
            self.report['eigen'] = {
                'valeurs_propres': self.linear_results['valeurs_propres'],
                'vecteurs_propres': self.linear_results['vecteurs_propres'],
                'classification': self.linear_results['classification'],
                'nature': self.linear_results['nature']
            }
            self.report['eigenspaces'] = self.linear_results['eigenspaces']
            self.fixed_points_data = [{
                'point': (0.0, 0.0),
                'jacobian': self.system,
                'nature': self.linear_results['nature'],
                'classification': self.linear_results['classification'],
                'valeurs_propres': self.linear_results['valeurs_propres'],
                'eigenspaces': self.linear_results['eigenspaces']
            }]
        else:
            # Système non-linéaire
            f1, f2 = system
            if isinstance(f1, str) and isinstance(f2, str):
                self.f1_str = f1
                self.f2_str = f2
                self.system = (safe_lambdify(f1), safe_lambdify(f2))
            else:
                self.system = (f1, f2)
                # Tenter de reconstruire des chaînes si possible, ou définir à None
                self.f1_str = None
                self.f2_str = None
                
            # Lancer l'analyse non-linéaire avec SymPy si on a les chaînes d'équations
            if self.f1_str and self.f2_str:
                self.nonlinear_results = analyze_nonlinear_system(self.f1_str, self.f2_str, self.domain)
                self.fixed_points_data = self.nonlinear_results['points_fixes']
            else:
                self.fixed_points_data = []
                self.nonlinear_results = {}
                
            # Remplir self.report avec le premier point fixe trouvé pour compatibilité
            if self.fixed_points_data:
                first_pf = self.fixed_points_data[0]
                self.report['eigen'] = {
                    'valeurs_propres': first_pf['valeurs_propres'],
                    'classification': first_pf['classification'],
                    'nature': first_pf['nature']
                }
                self.report['eigenspaces'] = first_pf['eigenspaces']
            else:
                self.report['eigen'] = {
                    'valeurs_propres': np.array([]),
                    'classification': [],
                    'nature': "Aucun point fixe trouvé dans le domaine"
                }
                self.report['eigenspaces'] = {
                    'E_s_dim': 0, 'E_s_basis': [],
                    'E_u_dim': 0, 'E_u_basis': [],
                    'E_c_dim': 0, 'E_c_basis': []
                }

    def analyze_eigenvalues(self) -> Dict:
        """Retourne l'analyse des valeurs propres du système linéaire."""
        if not self.is_linear:
            raise ValueError("L'analyse des valeurs propres s'applique uniquement aux systèmes linéaires (matrice A)")
        return self.report['eigen']

    def compute_eigenspaces(self, plot=True) -> Dict:
        """Calcule et affiche les sous-espaces pour un système linéaire."""
        if not self.is_linear:
            raise ValueError("Eigenspaces: système linéaire requis (matrice A)")
            
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5,5))
            plt.axhline(0, color='k', lw=0.5)
            plt.axvline(0, color='k', lw=0.5)
            colors = {'E_s':'b', 'E_u':'r', 'E_c':'g'}
            esp = self.report['eigenspaces']
            for v in esp['E_s_basis']:
                plt.arrow(0,0,v[0],v[1],color=colors['E_s'],width=0.05,head_width=0.2,length_includes_head=True,label='Stable')
            for v in esp['E_u_basis']:
                plt.arrow(0,0,v[0],v[1],color=colors['E_u'],width=0.05,head_width=0.2,length_includes_head=True,label='Instable')
            for v in esp['E_c_basis']:
                plt.arrow(0,0,v[0],v[1],color=colors['E_c'],width=0.05,head_width=0.2,length_includes_head=True,label='Centre')
            plt.xlim(-3,3)
            plt.ylim(-3,3)
            plt.title('Directions principales (espaces propres)')
            plt.grid(True)
            plt.show()
            
        return self.report['eigenspaces']

    def plot_isoclines(self, plot=True):
        """Trace les isoclines orientées."""
        return plot_isoclines(self, plot=plot)

    def quadrant_analysis(self) -> List[Dict]:
        """Analyse le comportement du flux dans chaque quadrant."""
        if self.is_linear:
            def f(x, y):
                A = self.system
                return A[0,0]*x + A[0,1]*y, A[1,0]*x + A[1,1]*y
            fixed = np.zeros(2)
        else:
            def f(x, y):
                return self.system[0](x, y), self.system[1](x, y)
            # Utiliser le premier point fixe trouvé comme centre de l'analyse, ou l'origine
            if self.fixed_points_data:
                fixed = np.array(self.fixed_points_data[0]['point'])
            else:
                fixed = np.zeros(2)
                
        x0, y0 = fixed[0], fixed[1]
        dx = (self.domain[1] - self.domain[0]) / 4
        dy = (self.domain[3] - self.domain[2]) / 4
        
        quadrants = [
            (x0+dx, y0+dy),
            (x0-dx, y0+dy),
            (x0-dx, y0-dy),
            (x0+dx, y0-dy)
        ]
        summary = []
        for i, (xq, yq) in enumerate(quadrants):
            u, v = f(xq, yq)
            sign = (np.sign(u), np.sign(v))
            if sign == (0,0):
                sens = 'Stationnaire'
            elif sign[0] == 0:
                sens = 'Verticale'
            elif sign[1] == 0:
                sens = 'Horizontale'
            elif sign == (1,1):
                sens = 'Divergent'
            elif sign == (-1,-1):
                sens = 'Convergent'
            else:
                sens = 'Tournant'
            summary.append({'quadrant':i+1,'point':(xq,yq),'sign':sign,'sens':sens})
            
        self.report['quadrants'] = summary
        return summary

    def plot_final_phase_portrait(self, n_trajectories=8, traj_len=10, plot=True):
        """Trace le portrait de phase complet."""
        return plot_final_phase_portrait(self, n_trajectories=n_trajectories, traj_len=traj_len, plot=plot)

    def generate_report(self, filename="rapport_analyse.pdf") -> str:
        """Génère le rapport PDF pédagogique complet."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        def add_text_page(pdf, title, content):
            fig, ax = plt.subplots(figsize=(8.3, 11.7))
            ax.axis('off')
            ax.text(0.5, 0.95, title, fontsize=18, ha='center', va='top', weight='bold')
            ax.text(0.05, 0.9, content, fontsize=12, ha='left', va='top', wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
            
        with PdfPages(filename) as pdf:
            # 1. Système étudié
            sys_txt = "Système étudié :\n"
            if self.is_linear:
                sys_txt += f"Système linéaire :\nA = {self.system}\n\nÉquations :\n"
                sys_txt += f"x' = {self.system[0,0]:.2f} x + {self.system[0,1]:.2f} y\n"
                sys_txt += f"y' = {self.system[1,0]:.2f} x + {self.system[1,1]:.2f} y"
            else:
                sys_txt += "Système non-linéaire :\n"
                if self.f1_str and self.f2_str:
                    sys_txt += f"x' = {self.f1_str}\ny' = {self.f2_str}\n"
                else:
                    sys_txt += "x' = f₁(x, y)\ny' = f₂(x, y)\n"
            sys_txt += f"\nDomaine d'étude : {self.domain}"
            add_text_page(pdf, "1. Système étudié", sys_txt + "\n\nCe système sera analysé qualitativement selon la méthode en 5 étapes.")
            
            # 2. Valeurs propres et classification
            if self.is_linear:
                eig = self.report['eigen']
                eig_txt = f"Valeurs propres : {eig['valeurs_propres']}\nClassification : {eig['classification']}\nNature : {eig['nature']}\n\n"
                eig_txt += "\nLes valeurs propres déterminent la stabilité locale du point fixe.\n"
            else:
                eig_txt = "Système Non-Linéaire - Analyse des Points Fixes :\n\n"
                if self.fixed_points_data:
                    for idx, pt in enumerate(self.fixed_points_data):
                        px, py = pt['point']
                        eig_txt += f"Point Fixe #{idx+1} : ({px:.3f}, {py:.3f})\n"
                        eig_txt += f"  - Valeurs propres : {pt['valeurs_propres']}\n"
                        eig_txt += f"  - Classification : {pt['classification']}\n"
                        eig_txt += f"  - Nature locale : {pt['nature']}\n\n"
                else:
                    eig_txt += "Aucun point fixe trouvé dans le domaine.\n"
            add_text_page(pdf, "2. Valeurs propres et classification", eig_txt)
            
            # 3. Vecteurs propres et sous-espaces
            if self.is_linear:
                esp = self.report['eigenspaces']
                esp_txt = f"Eₛ (stable) : dim={esp['E_s_dim']} base={esp['E_s_basis']}\n"
                esp_txt += f"Eᵤ (instable) : dim={esp['E_u_dim']} base={esp['E_u_basis']}\n"
                esp_txt += f"E꜀ (centre) : dim={esp['E_c_dim']} base={esp['E_c_basis']}\n\n"
                esp_txt += "\nLes sous-espaces propres structurent le flux autour du point fixe.\n"
            else:
                esp_txt = "Système Non-Linéaire - Sous-espaces Propres Locaux :\n\n"
                if self.fixed_points_data:
                    for idx, pt in enumerate(self.fixed_points_data):
                        px, py = pt['point']
                        esp = pt['eigenspaces']
                        esp_txt += f"Autour de ({px:.2f}, {py:.2f}) :\n"
                        esp_txt += f"  - Eₛ (stable) : dim={esp['E_s_dim']} base={esp['E_s_basis']}\n"
                        esp_txt += f"  - Eᵤ (instable) : dim={esp['E_u_dim']} base={esp['E_u_basis']}\n"
                        esp_txt += f"  - E꜀ (centre) : dim={esp['E_c_dim']} base={esp['E_c_basis']}\n\n"
                else:
                    esp_txt += "Aucun point fixe à analyser.\n"
            add_text_page(pdf, "3. Vecteurs propres et sous-espaces", esp_txt)
            
            # 4. Graphique isoclines
            fig_iso = self.plot_isoclines(plot=False)
            pdf.savefig(fig_iso)
            plt.close(fig_iso)
            add_text_page(pdf, "4. Isoclines orientées", "Les isoclines ẋ=0 et ẏ=0 séparent les régions de directions opposées. Les flèches indiquent l'orientation du mouvement sur chaque isocline.")
            
            # 5. Portrait de phase final
            fig_phase = self.plot_final_phase_portrait(plot=False)
            pdf.savefig(fig_phase)
            plt.close(fig_phase)
            add_text_page(pdf, "5. Portrait de phase final", "Ce graphique intègre toutes les informations : isoclines, directions principales, sous-espaces, trajectoires, points fixes et flux.")
            
            # 6. Conclusion
            concl = "Le comportement dynamique global dépend de la nature des valeurs propres et des sous-espaces.\n"
            if self.is_linear:
                nat = self.report['eigen']['nature']
                if 'nœud' in nat.lower() or 'stable' in nat.lower():
                    concl += "Le système converge vers le point fixe (stable)."
                elif 'instable' in nat.lower():
                    concl += "Le système diverge du point fixe (instable)."
                elif 'selle' in nat.lower():
                    concl += "Le point fixe est une selle : certaines directions sont stables, d'autres instables."
                elif 'centre' in nat.lower():
                    concl += "Le système présente des oscillations périodiques (centre)."
                else:
                    concl += f"Nature : {nat}"
            else:
                concl += "Pour ce système non-linéaire, plusieurs points fixes ont été analysés localement. Le comportement dynamique global présente des régions d'attraction, des trajectoires divergentes ou des orbites cycliques autour de ces points fixes."
            add_text_page(pdf, "6. Conclusion", concl)
        return filename
