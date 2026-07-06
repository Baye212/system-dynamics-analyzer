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
        """Génère le rapport PDF pédagogique complet style LaTeX."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib as mpl
        import numpy as np

        # Configurer Matplotlib pour un rendu MathText style LaTeX
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        mpl.rcParams['font.size'] = 12

        def format_complex(c):
            if np.isreal(c) or np.isclose(c.imag, 0):
                return f"{c.real:.3f}"
            sign = "+" if c.imag > 0 else "-"
            return f"{c.real:.3f} {sign} {abs(c.imag):.3f}i"

        def explain_stability(eigenvalues, nature):
            if len(eigenvalues) == 0:
                return "Aucune valeur propre à analyser."
            l1, l2 = eigenvalues[0], eigenvalues[1]
            txt = f"Soient les valeurs propres $\lambda_1 = {format_complex(l1)}$ et $\lambda_2 = {format_complex(l2)}$.\n\n"
            
            if np.isrealobj(eigenvalues) or (np.isclose(l1.imag, 0) and np.isclose(l2.imag, 0)):
                if l1.real < 0 and l2.real < 0:
                    txt += "Les deux valeurs propres sont réelles et strictement négatives.\n"
                    txt += "Par conséquent, le point d'équilibre est asymptotiquement stable.\n"
                elif l1.real > 0 and l2.real > 0:
                    txt += "Les deux valeurs propres sont réelles et strictement positives.\n"
                    txt += "Par conséquent, le point d'équilibre est instable.\n"
                elif l1.real * l2.real < 0:
                    txt += "Les valeurs propres sont réelles et de signes opposés.\n"
                    txt += "Le point d'équilibre est un point col (ou selle), qui est instable.\n"
                else:
                    txt += "L'une des valeurs propres au moins est nulle.\n"
            else:
                alpha = l1.real
                txt += f"Les valeurs propres sont complexes conjuguées, de partie réelle $\\alpha = {alpha:.3f}$.\n"
                if alpha < 0:
                    txt += "Comme $\\alpha < 0$, les trajectoires spiralent vers le point d'équilibre.\n"
                    txt += "Le point est asymptotiquement stable.\n"
                elif alpha > 0:
                    txt += "Comme $\\alpha > 0$, les trajectoires s'éloignent en spiralant.\n"
                    txt += "Le point est instable.\n"
                else:
                    txt += "Comme $\\alpha = 0$, les valeurs propres sont des imaginaires purs.\n"
                    txt += "Le système linéarisé présente un centre (oscillations périodiques).\n"
            txt += f"\nConclusion : {nature}."
            return txt

        def add_text_page(pdf, title, content):
            fig, ax = plt.subplots(figsize=(8.3, 11.7))
            ax.axis('off')
            ax.text(0.5, 0.95, title, fontsize=18, ha='center', va='top', weight='bold')
            ax.text(0.05, 0.88, content, fontsize=12, ha='left', va='top', wrap=True, linespacing=1.6)
            pdf.savefig(fig)
            plt.close(fig)

        with PdfPages(filename) as pdf:
            # 1. Système étudié
            sys_title = "1. Définition du Système Dynamique"
            if self.is_linear:
                A = self.system
                sys_content = "Le système étudié est un système dynamique linéaire plan défini par :\n\n"
                sys_content += "$\dot{\mathbf{X}} = A \mathbf{X}$\n\n"
                sys_content += "Avec la matrice $A$ suivante :\n\n"
                sys_content += f"A = [ {A[0,0]:.2f}  {A[0,1]:.2f} ]\n"
                sys_content += f"    [ {A[1,0]:.2f}  {A[1,1]:.2f} ]\n\n"
                sys_content += "Soit sous forme de système d'équations :\n"
                sys_content += f"$\\dot{{x}} = {A[0,0]:.2f} x + {A[0,1]:.2f} y$\n"
                sys_content += f"$\\dot{{y}} = {A[1,0]:.2f} x + {A[1,1]:.2f} y$\n"
            else:
                sys_content = "Le système étudié est un système dynamique non-linéaire plan défini par :\n\n"
                sys_content += "$\dot{x} = f_1(x, y)$\n"
                sys_content += "$\dot{y} = f_2(x, y)$\n\n"
                if self.f1_str and self.f2_str:
                    sys_content += "Les fonctions du système sont :\n"
                    f1_tex = self.f1_str.replace('**', '^').replace('*', '')
                    f2_tex = self.f2_str.replace('**', '^').replace('*', '')
                    sys_content += f"$\\dot{{x}} = {f1_tex}$\n"
                    sys_content += f"$\\dot{{y}} = {f2_tex}$\n"
            sys_content += f"\nDomaine d'étude spatial : $x \in [{self.domain[0]}, {self.domain[1]}]$, $y \in [{self.domain[2]}, {self.domain[3]}]$.\n"
            add_text_page(pdf, sys_title, sys_content)

            # 2. Points Fixes
            stab_title = "2. Points d'Équilibre et Stabilité Locale"
            if self.is_linear:
                eig = self.report['eigen']
                content = "Le système étant linéaire et homogène, l'origine $(0,0)$ est un point d'équilibre.\n\n"
                content += explain_stability(eig['valeurs_propres'], eig['nature'])
                add_text_page(pdf, stab_title, content)
            else:
                if self.fixed_points_data:
                    for idx, pt in enumerate(self.fixed_points_data):
                        px, py = pt['point']
                        content = f"Point d'équilibre $P_{idx+1} = ({px:.3f}, {py:.3f})$\n\n"
                        content += "Pour étudier la stabilité locale, on linéarise le système autour de ce point\n"
                        content += "en calculant la matrice Jacobienne $J(P)$ :\n\n"
                        J = pt['jacobian']
                        content += f"J(P) = [ {J[0,0]:.2f}  {J[0,1]:.2f} ]\n"
                        content += f"       [ {J[1,0]:.2f}  {J[1,1]:.2f} ]\n\n"
                        content += explain_stability(pt['valeurs_propres'], pt['nature'])
                        add_text_page(pdf, f"2.{idx+1} Étude du point ({px:.2f}, {py:.2f})", content)
                else:
                    add_text_page(pdf, stab_title, "Aucun point fixe réel n'a été trouvé dans le domaine d'étude.")

            # 3. Sous-espaces Propres
            esp_title = "3. Décomposition des Sous-espaces Propres"
            esp_intro = "L'espace se décompose en trois sous-espaces invariants selon le signe\n"
            esp_intro += "de la partie réelle des valeurs propres :\n"
            esp_intro += "- $E_s$ (Stable) : associé aux valeurs propres de partie réelle $< 0$.\n"
            esp_intro += "- $E_u$ (Instable) : associé aux valeurs propres de partie réelle $> 0$.\n"
            esp_intro += "- $E_c$ (Centre) : associé aux valeurs propres de partie réelle $= 0$.\n\n"
            
            def format_vec(v):
                return f"[{v[0]:.3f}, {v[1]:.3f}]^T"

            if self.is_linear:
                esp = self.report['eigenspaces']
                content = esp_intro + "Pour le système linéaire (à l'origine) :\n\n"
                if esp['E_s_basis']:
                    basis_str = ", ".join([f"${format_vec(v)}$" for v in esp['E_s_basis']])
                    content += f"Sous-espace Stable $E_s$ (Attractif) :\nDimension = {esp['E_s_dim']}. Base : {basis_str}\n\n"
                if esp['E_u_basis']:
                    basis_str = ", ".join([f"${format_vec(v)}$" for v in esp['E_u_basis']])
                    content += f"Sous-espace Instable $E_u$ (Répulsif) :\nDimension = {esp['E_u_dim']}. Base : {basis_str}\n\n"
                if esp['E_c_basis']:
                    basis_str = ", ".join([f"${format_vec(v)}$" for v in esp['E_c_basis']])
                    content += f"Sous-espace Centre $E_c$ (Oscillant) :\nDimension = {esp['E_c_dim']}. Base : {basis_str}\n\n"
                add_text_page(pdf, esp_title, content)
            else:
                if self.fixed_points_data:
                    for idx, pt in enumerate(self.fixed_points_data):
                        esp = pt['eigenspaces']
                        px, py = pt['point']
                        content = esp_intro + f"Pour le point $P = ({px:.2f}, {py:.2f})$ (linéarisé) :\n\n"
                        if esp['E_s_basis']:
                            basis_str = ", ".join([f"${format_vec(v)}$" for v in esp['E_s_basis']])
                            content += f"$E_s$ (Stable) : Dim = {esp['E_s_dim']}. Base : {basis_str}\n\n"
                        if esp['E_u_basis']:
                            basis_str = ", ".join([f"${format_vec(v)}$" for v in esp['E_u_basis']])
                            content += f"$E_u$ (Instable) : Dim = {esp['E_u_dim']}. Base : {basis_str}\n\n"
                        if esp['E_c_basis']:
                            basis_str = ", ".join([f"${format_vec(v)}$" for v in esp['E_c_basis']])
                            content += f"$E_c$ (Centre) : Dim = {esp['E_c_dim']}. Base : {basis_str}\n\n"
                        add_text_page(pdf, f"3.{idx+1} Sous-espaces pour ({px:.2f}, {py:.2f})", content)

            # 4. Isoclines
            fig_iso = self.plot_isoclines(plot=False)
            fig_iso.suptitle("4. Isoclines Orientées", fontsize=16, weight='bold', y=0.98)
            pdf.savefig(fig_iso)
            plt.close(fig_iso)

            # 5. Portrait de phase Global
            fig_phase = self.plot_final_phase_portrait(plot=False)
            fig_phase.suptitle("5. Portrait de Phase Global", fontsize=16, weight='bold', y=0.98)
            pdf.savefig(fig_phase)
            plt.close(fig_phase)

            # 6. Portraits de phase locaux (Pour les non-linéaires)
            if not self.is_linear and self.fixed_points_data:
                for idx, pt in enumerate(self.fixed_points_data):
                    px, py = pt['point']
                    local_analyzer = DynamicsAnalyzer(pt['jacobian'], domain=(-2, 2, -2, 2))
                    fig_local = local_analyzer.plot_final_phase_portrait(plot=False)
                    fig_local.suptitle(f"5.{idx+1} Portrait Local Linéarisé autour de ({px:.2f}, {py:.2f})", fontsize=16, weight='bold', y=0.98)
                    pdf.savefig(fig_local)
                    plt.close(fig_local)

            mpl.rcParams.update(mpl.rcParamsDefault)
            
        return filename
