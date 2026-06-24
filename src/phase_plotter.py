"""
Module phase_plotter.py : Gestion des tracés graphiques (portraits de phase, isoclines).
Conserve la logique et le rendu visuel exacts définis par l'utilisateur.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.colors import ListedColormap

def plot_isoclines(analyzer, plot=True):
    """
    Trace les isoclines ẋ=0 et ẏ=0, colorie les régions, ajoute flèches, marque intersections, carte des directions.
    """
    if isinstance(analyzer.system, np.ndarray):
        def f(x, y):
            A = analyzer.system
            return A[0,0]*x + A[0,1]*y, A[1,0]*x + A[1,1]*y
    else:
        def f(x, y):
            return analyzer.system[0](x, y), analyzer.system[1](x, y)
    x_min, x_max, y_min, y_max = analyzer.domain
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    U, V = f(X, Y)
    fig, ax = plt.subplots(figsize=(6,6))
    # Isoclines (labels supprimés car non pris en charge par contour)
    ax.contour(X, Y, U, levels=[0], colors='b', linewidths=2, linestyles='--')
    ax.contour(X, Y, V, levels=[0], colors='r', linewidths=2, linestyles='-.')
    # Color regions
    cmap = ListedColormap(['#d0f0c0','#f0d0c0'])
    ax.contourf(X, Y, np.sign(U), alpha=0.1, cmap='Blues')
    ax.contourf(X, Y, np.sign(V), alpha=0.1, cmap='Reds')
    # Arrows on isoclines
    for c in [0]:
        iso_x = ax.contour(X, Y, U, levels=[c], colors='b', linewidths=2)
        iso_y = ax.contour(X, Y, V, levels=[c], colors='r', linewidths=2)
        # Compatibilité: certaines versions n'exposent pas 'collections'
        collections_x = getattr(iso_x, 'collections', [])
        collections_y = getattr(iso_y, 'collections', [])
        for collection in list(collections_x) + list(collections_y):
            # get_paths peut ne pas exister sur certains artistes; on vérifie
            get_paths = getattr(collection, 'get_paths', None)
            if not callable(get_paths):
                continue
            for path in get_paths():
                v = path.vertices
                if len(v) > 10:
                    idx = len(v)//2
                    dx, dy = f(v[idx,0], v[idx,1])
                    ax.arrow(v[idx,0], v[idx,1], dx/10, dy/10, head_width=0.2, color='k')
    # Points fixes
    if isinstance(analyzer.system, np.ndarray):
        try:
            from numpy.linalg import solve
            fixed = solve(analyzer.system, np.zeros(2))
            ax.plot(fixed[0], fixed[1], 'ko', label='Point fixe')
        except Exception:
            pass
    else:
        # Pour les systèmes non-linéaires, si on a trouvé des points fixes réels dans le rapport, on les affiche
        if hasattr(analyzer, 'fixed_points_data') and analyzer.fixed_points_data:
            for pt_data in analyzer.fixed_points_data:
                px, py = pt_data['point']
                ax.plot(px, py, 'ko')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Isoclines et directions')
    ax.grid(True)
    if plot:
        plt.show()
        plt.close(fig)
    return fig

def plot_final_phase_portrait(analyzer, n_trajectories=8, traj_len=10, plot=True):
    """
    Portrait de phase de haut niveau avec style professionnel :
    - Champ de vecteurs en arrière-plan
    - Trajectoires colorées avec conditions initiales
    - Variétés stables/instables mises en évidence
    - Formatage LaTeX pour les équations
    """
    # Définition du système
    if isinstance(analyzer.system, np.ndarray):
        A = analyzer.system
        def system_ode(t, y):
            return A @ y
        def field_vector(x, y):
            return A[0,0]*x + A[0,1]*y, A[1,0]*x + A[1,1]*y
        # Classification pour le titre
        try:
            eigvals = np.linalg.eigvals(A)
            if 'eigen' in analyzer.report:
                nature = analyzer.report['eigen']['nature']
            else:
                nature = "Point fixe"
            # Équations pour le titre
            title_eq = f"$\\dot{{x}} = {A[0,0]:.2f}x + {A[0,1]:.2f}y, \\quad \\dot{{y}} = {A[1,0]:.2f}x + {A[1,1]:.2f}y$"
        except:
            nature = "Système linéaire"
            title_eq = "$\\dot{\\mathbf{x}} = A\\mathbf{x}$"
    else:
        def system_ode(t, y):
            return [analyzer.system[0](y[0], y[1]), analyzer.system[1](y[0], y[1])]
        def field_vector(x, y):
            return analyzer.system[0](x, y), analyzer.system[1](x, y)
        nature = "Système non-linéaire"
        title_eq = "$\\dot{x} = f_1(x,y), \\quad \\dot{y} = f_2(x,y)$"
    
    # Paramètres du tracé
    x_min, x_max, y_min, y_max = analyzer.domain
    num_points = 15
    t_span = [0, traj_len]
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    
    # Création de la grille pour le champ de vecteurs
    x_grid = np.linspace(x_min, x_max, num_points)
    y_grid = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calcul du champ de vecteurs
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            u_val, v_val = field_vector(X[i, j], Y[i, j])
            U[i, j] = u_val
            V[i, j] = v_val
    
    # Conditions initiales variées
    if isinstance(analyzer.system, np.ndarray):
        # Pour systèmes linéaires : conditions autour de l'origine
        conditions_initiales = [
            [2.5, 0.0], [-2.5, 0.0],  # Sur l'axe x
            [0.0, 2.5], [0.0, -2.5],  # Sur l'axe y
            [1.8, 1.8], [-1.8, 1.8],  # Quadrants I et II
            [-1.8, -1.8], [1.8, -1.8] # Quadrants III et IV
        ]
    else:
        # Pour systèmes non-linéaires : conditions réparties
        conditions_initiales = [
            [2.0, 0.0], [-2.0, 0.0],
            [0.0, 2.0], [0.0, -2.0],
            [1.5, 1.5], [-1.5, 1.5],
            [-1.5, -1.5], [1.5, -1.5]
        ]
        # Si on a des points fixes, on peut ajuster ou rajouter des trajectoires autour d'eux
        if hasattr(analyzer, 'fixed_points_data') and analyzer.fixed_points_data:
            # Ajouter des points autour de chaque point fixe
            for pt_data in analyzer.fixed_points_data:
                px, py = pt_data['point']
                eps = 0.2
                conditions_initiales.extend([
                    [px + eps, py + eps],
                    [px - eps, py - eps],
                    [px + eps, py - eps],
                    [px - eps, py + eps]
                ])
            # Limiter pour ne pas surcharger
            conditions_initiales = conditions_initiales[:12]
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Champ de vecteurs en arrière-plan
    ax.quiver(X, Y, U, V, scale=30, color='lightblue', alpha=0.6, width=0.005)
    
    # 2. Trajectoires avec couleurs distinctes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    for i, y0 in enumerate(conditions_initiales):
        try:
            sol = solve_ivp(system_ode, t_span, y0, t_eval=t_eval, method='RK45')
            ax.plot(sol.y[0], sol.y[1], color=colors[i % len(colors)], linewidth=2, 
                   label=f'$y_0=({y0[0]:.1f}, {y0[1]:.1f})$')
            # Point initial
            ax.plot(sol.y[0][0], sol.y[1][0], 'o', color=colors[i % len(colors)], markersize=6)
        except Exception as e:
            continue
    
    # 3. Variétés stables/instables (si applicable)
    if isinstance(analyzer.system, np.ndarray) and 'eigenspaces' in analyzer.report:
        esp_data = analyzer.report['eigenspaces']
        
        # Variété stable (E_s)
        if esp_data['E_s_basis']:
            for v in esp_data['E_s_basis']:
                scale = max(x_max - x_min, y_max - y_min) / 2
                ax.plot([-scale*v[0], scale*v[0]], [-scale*v[1], scale*v[1]], 
                       'darkblue', linestyle='-', linewidth=4, 
                       label='Variété stable $W^s$')
        
        # Variété instable (E_u)
        if esp_data['E_u_basis']:
            for v in esp_data['E_u_basis']:
                scale = max(x_max - x_min, y_max - y_min) / 2
                ax.plot([-scale*v[0], scale*v[0]], [-scale*v[1], scale*v[1]], 
                       'darkred', linestyle='-', linewidth=4, 
                       label='Variété instable $W^u$')
    else:
        # Pour les systèmes non-linéaires, dessiner les points fixes
        if hasattr(analyzer, 'fixed_points_data') and analyzer.fixed_points_data:
            for pt_data in analyzer.fixed_points_data:
                px, py = pt_data['point']
                nature_pt = pt_data['nature']
                if 'stable' in nature_pt.lower() or 'stable' in pt_data['classification']:
                    color_pt = 'go' # Vert pour stable
                elif 'instable' in nature_pt.lower() or 'instable' in pt_data['classification']:
                    color_pt = 'ro' # Rouge pour instable
                elif 'selle' in nature_pt.lower():
                    color_pt = 'bo' # Bleu pour selle
                else:
                    color_pt = 'ko'
                ax.plot(px, py, color_pt, markersize=10, label=f'PF ({px:.2f}, {py:.2f}) - {nature_pt}')
    
    # 4. Axes et grille
    ax.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 5. Mise en forme
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_title(f'Portrait de phase : {title_eq}\n({nature})', fontsize=16, pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if plot:
        plt.show()
        plt.close(fig)
    return fig
