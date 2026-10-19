import io
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.dynamics_analyzer import DynamicsAnalyzer

# ─── Configuration de la page (DOIT être en premier) ───────────────────────
st.set_page_config(
    page_title="Analyse Qualitative de Systèmes Dynamiques - Outil Interactif",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Baye212/system-dynamics-analyzer',
        'Report a bug': 'https://github.com/Baye212/system-dynamics-analyzer/issues',
        'About': "Mor DIOUF - Outil pédagogique pour l'analyse des systèmes dynamiques en 2D"
    }
)

# ─── SEO ────────────────────────────────────────────────────────────────────
st.markdown("""
<meta name="description" content="Outil interactif gratuit pour analyser qualitativement les systèmes dynamiques linéaires et non-linéaires.">
<meta name="keywords" content="systèmes dynamiques, analyse qualitative, valeurs propres, portrait de phase, isoclines, stabilité">
<meta name="author" content="Mor DIOUF">
<meta name="google-site-verification" content="IrUWbS1NZxO8bpVaINJJLL2vI9_nLcroUq8Ylu_Hv_w">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Fonctions d'affichage
# ═══════════════════════════════════════════════════════════════════════════

def fmt(x):
    """Nombre réel lisible : entier sans décimales, sinon 3 décimales."""
    x = float(np.real(x))
    if abs(x) < 5e-4:
        return "0"
    return f"{x:.0f}" if abs(x - round(x)) < 5e-4 else f"{x:.3f}"


def format_eigenvalue(val):
    """Écrit une valeur propre en LaTeX : réelle (a) ou complexe (a ± bi)."""
    if abs(np.imag(val)) < 1e-9:
        return fmt(val)
    sign = "+" if np.imag(val) > 0 else "-"
    return f"{fmt(np.real(val))} {sign} {fmt(abs(np.imag(val)))}\\,i"


def matrix_latex(M):
    return f"\\begin{{pmatrix}} {fmt(M[0][0])} & {fmt(M[0][1])} \\\\ {fmt(M[1][0])} & {fmt(M[1][1])} \\end{{pmatrix}}"


def vector_latex(v):
    return f"\\begin{{pmatrix}} {fmt(v[0])} \\\\ {fmt(v[1])} \\end{{pmatrix}}"


def show_system(analyzer):
    """Affiche le système étudié et le domaine en tête des résultats."""
    d = analyzer.domain
    col_sys, col_dom = st.columns([3, 2])
    with col_sys:
        st.markdown("#### Système étudié")
        if analyzer.is_linear:
            st.latex(f"\\dot{{X}} = A X, \\qquad A = {matrix_latex(analyzer.system)}")
        else:
            exprs = analyzer.nonlinear_results.get('expressions')
            if exprs:
                e1, e2 = (sp.latex(e) for e in exprs)
                st.latex(f"\\begin{{cases}} \\dot{{x}}_1 = {e1} \\\\ \\dot{{x}}_2 = {e2} \\end{{cases}}")
    with col_dom:
        st.markdown("#### Domaine d'étude")
        st.latex(f"x_1 \\in [{fmt(d[0])},\\ {fmt(d[1])}], \\quad x_2 \\in [{fmt(d[2])},\\ {fmt(d[3])}]")


def format_eigenvalues_display(eig_data, local=False):
    """Formate l'affichage des valeurs propres en style LaTeX.

    local=True : analyse d'un point fixe d'un système non-linéaire (conclusions locales,
    valables seulement si le point est hyperbolique — théorème de Hartman–Grobman).
    """
    st.markdown("### Valeurs propres")
    eigenvalues = eig_data['valeurs_propres']
    if len(eigenvalues) == 2:
        lambda1, lambda2 = eigenvalues
        st.latex(f"\\lambda_1 = {format_eigenvalue(lambda1)}, \\qquad \\lambda_2 = {format_eigenvalue(lambda2)}")
    else:
        st.write(f"λ = {eigenvalues}")

    items = []
    for i, (val, cls) in enumerate(zip(eigenvalues, eig_data['classification'])):
        re = np.real(val)
        color = "🟢" if re < -1e-9 else ("🔴" if re > 1e-9 else "🟡")
        items.append(f"{color} $\\lambda_{{{i+1}}}$ : **{cls}**")
    st.markdown("**Classification** : " + " &nbsp;·&nbsp; ".join(items))

    nature = eig_data['nature']
    nl = nature.lower()
    hyperbolic = all(abs(np.real(v)) > 1e-9 for v in eigenvalues)
    near = "Les trajectoires proches du point fixe" if local else "Les trajectoires"
    st.markdown("**Nature du point fixe** :")
    if "selle" in nl:
        st.warning(f"🔄 **{nature}** — {near} s'approchent selon $E_s$ puis s'éloignent selon $E_u$.")
    elif "instable" in nl:
        st.error(f"⚠️ **{nature}** — {near} s'éloignent du point fixe.")
    elif "stable" in nl:
        st.success(f"🎯 **{nature}** — {near} convergent vers le point fixe.")
    elif "centre" in nl:
        st.info(f"🌀 **{nature}** — {near} sont des orbites périodiques fermées.")
    else:
        st.info(f"📊 **{nature}**")

    if local:
        if hyperbolic:
            st.caption("Le point fixe est hyperbolique (aucune valeur propre de partie réelle nulle) : "
                       "d'après le théorème de Hartman–Grobman, le système non-linéaire a localement "
                       "le même comportement que son linéarisé.")
        else:
            st.warning("⚠️ Point fixe **non hyperbolique** (valeur propre de partie réelle nulle) : "
                       "la linéarisation **ne permet pas de conclure** sur la nature du point fixe "
                       "du système non-linéaire (Hartman–Grobman ne s'applique pas). "
                       "Une étude complémentaire est nécessaire (fonction de Lyapunov, variété centrale…).")


def format_subspaces_display(esp_data):
    """Formate l'affichage des sous-espaces en style LaTeX (trois colonnes)."""
    st.markdown("### Sous-espaces propres")
    spaces = [
        ("Stable", "E_s", esp_data['E_s_dim'], esp_data['E_s_basis']),
        ("Instable", "E_u", esp_data['E_u_dim'], esp_data['E_u_basis']),
        ("Centre", "E_c", esp_data['E_c_dim'], esp_data['E_c_basis']),
    ]
    for col, (label, sym, dim, basis) in zip(st.columns(3), spaces):
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}** ${sym}$ — $\\dim {sym} = {dim}$")
                if basis:
                    vecs = ",\\ ".join(vector_latex(v) for v in basis)
                    st.latex(f"{sym} = \\operatorname{{Vect}}\\left\\{{ {vecs} \\right\\}}")
                else:
                    st.latex(f"{sym} = \\{{0\\}}")


# ═══════════════════════════════════════════════════════════════════════════
# En-tête principal
# ═══════════════════════════════════════════════════════════════════════════

st.title("🔬 Analyse Qualitative de Systèmes Dynamiques")
st.caption("Outil pédagogique interactif et gratuit pour l'analyse des systèmes dynamiques linéaires et non-linéaires en 2D")

with st.expander("ℹ️ Fonctionnalités de l'outil"):
    st.markdown("""
- 📊 Analyse des valeurs propres et classification des points fixes
- 🧮 Calcul des vecteurs propres et sous-espaces (stable, instable, centre)
- 📈 Visualisation des isoclines orientées
- 🎨 Génération de portraits de phase complets
- 📄 Export de rapports PDF détaillés
""")


# ═══════════════════════════════════════════════════════════════════════════
# Barre latérale — Configuration (clés uniques sur tous les widgets)
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## ⚙️ Configuration du système")

system_type = st.sidebar.selectbox(
    "Type de système",
    ["Linéaire", "Non-linéaire", "Exemple prédéfini"],
    key="sb_system_type"
)


def domain_inputs(prefix, default=(-5.0, 5.0, -5.0, 5.0)):
    """Saisie du domaine d'étude par quatre champs numériques."""
    st.sidebar.markdown("**Domaine d'étude**")
    c1, c2 = st.sidebar.columns(2)
    xmin = c1.number_input("x₁ min", value=default[0], key=f"{prefix}_xmin")
    xmax = c2.number_input("x₁ max", value=default[1], key=f"{prefix}_xmax")
    ymin = c1.number_input("x₂ min", value=default[2], key=f"{prefix}_ymin")
    ymax = c2.number_input("x₂ max", value=default[3], key=f"{prefix}_ymax")
    return xmin, xmax, ymin, ymax


btn_linear = btn_nonlin = btn_example = False

if system_type == "Linéaire":
    st.sidebar.markdown("**Matrice A**")
    c1, c2 = st.sidebar.columns(2)
    a11 = c1.number_input("a₁₁", value=1.0, key="ni_a11")
    a12 = c2.number_input("a₁₂", value=0.0, key="ni_a12")
    a21 = c1.number_input("a₂₁", value=0.0, key="ni_a21")
    a22 = c2.number_input("a₂₂", value=1.0, key="ni_a22")
    domain_lin = domain_inputs("lin")
    btn_linear = st.sidebar.button("▶ Analyser", key="btn_linear", type="primary", use_container_width=True)

elif system_type == "Non-linéaire":
    st.sidebar.markdown("**Équations** (variables `x1`, `x2` ou `x`, `y`)")
    f1_input = st.sidebar.text_input("ẋ₁ = f₁(x₁, x₂)", "x1 - x2 + x1**2", key="ti_f1")
    f2_input = st.sidebar.text_input("ẋ₂ = f₂(x₁, x₂)", "-x1 + 2*x2", key="ti_f2")
    domain_nl = domain_inputs("nl")
    btn_nonlin = st.sidebar.button("▶ Analyser", key="btn_nonlin", type="primary", use_container_width=True)

else:
    EXEMPLES = ["Nœud stable", "Selle", "Centre", "Spirale stable", "Système non-linéaire simple"]
    ex_choice = st.sidebar.selectbox("Exemple", EXEMPLES, key="sb_example")
    btn_example = st.sidebar.button("▶ Charger l'exemple", key="btn_example", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Logique d'analyse (construction de l'analyseur)
# ═══════════════════════════════════════════════════════════════════════════

def check_domain(domain):
    xmin, xmax, ymin, ymax = domain
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("le domaine est invalide : il faut x₁ min < x₁ max et x₂ min < x₂ max.")
    return domain

def build_analyzer_linear():
    A = np.array([[a11, a12], [a21, a22]])
    return DynamicsAnalyzer(A, check_domain(domain_lin))

def build_analyzer_nonlinear():
    return DynamicsAnalyzer((f1_input, f2_input), check_domain(domain_nl))

def build_analyzer_example(name):
    configs = {
        "Nœud stable":               (np.array([[-2., 1.], [1., -3.]]),  (-3, 3, -3, 3)),
        "Selle":                     (np.array([[1., 2.], [2., 1.]]),    (-3, 3, -3, 3)),
        "Centre":                    (np.array([[0., -1.], [1., 0.]]),   (-3, 3, -3, 3)),
        "Spirale stable":            (np.array([[-1., -2.], [2., -1.]]), (-3, 3, -3, 3)),
        "Système non-linéaire simple": (("x1*(1 - x1 - x2)", "x2*(x1 - 0.5)"), (0, 1.5, 0, 1.5)),
    }
    system, domain = configs[name]
    return DynamicsAnalyzer(system, domain)

error_msg = None

if btn_linear:
    try:
        st.session_state['analyzer'] = build_analyzer_linear()
        st.session_state['analyzer_type'] = 'linear'
    except Exception as e:
        error_msg = f"Erreur (linéaire) : {e}"

if btn_nonlin:
    try:
        st.session_state['analyzer'] = build_analyzer_nonlinear()
        st.session_state['analyzer_type'] = 'nonlinear'
    except Exception as e:
        error_msg = f"Erreur (non-linéaire) : vérifiez la syntaxe des équations (ex : x1**2, sin(x2)). Détail : {e}"

if btn_example:
    try:
        st.session_state['analyzer'] = build_analyzer_example(ex_choice)
        st.session_state['analyzer_type'] = 'example'
    except Exception as e:
        error_msg = f"Erreur (exemple) : {e}"

if error_msg:
    st.error(error_msg)

analyzer = st.session_state.get('analyzer', None)


# ═══════════════════════════════════════════════════════════════════════════
# Zone principale — Résultats (structure DOM STABLE via st.tabs)
# ═══════════════════════════════════════════════════════════════════════════

if analyzer is None:
    st.info("👈 Configurez un système dans la barre latérale puis cliquez sur **Analyser**.")
else:
    show_system(analyzer)

    # ── Sélection du point fixe (non-linéaire) ──────────────────────────
    selected_point_data = None
    if not analyzer.is_linear:
        st.markdown("#### Points fixes réels détectés")
        if analyzer.fixed_points_data:
            n_pts = len(analyzer.fixed_points_data)
            st.success(f"Le système possède **{n_pts}** point(s) fixe(s) réel(s) dans le domaine.")
            options = [
                f"P{i+1} = ({fmt(pt['point'][0])}, {fmt(pt['point'][1])}) — {pt['nature']}"
                for i, pt in enumerate(analyzer.fixed_points_data)
            ]
            # Clé fixe + index stocké en session pour éviter les re-rendus
            sel_idx = st.radio(
                "Sélectionner un point fixe pour l'analyse locale :",
                range(len(options)),
                format_func=lambda i: options[i],
                key="radio_fixed_point",
                horizontal=True,
            )
            selected_point_data = analyzer.fixed_points_data[sel_idx]
        else:
            st.warning("⚠️ Aucun point fixe réel n'a été détecté dans le domaine d'étude.")

    # ── Onglets d'analyse (structure stable) ────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Valeurs propres",
        "2. Sous-espaces",
        "3. Isoclines",
        "4. Portrait de phase",
        "Rapport PDF",
    ])

    # ── Onglet 1 : Valeurs propres ───────────────────────────────────────
    with tab1:
        if analyzer.is_linear:
            try:
                eig = analyzer.analyze_eigenvalues()
                format_eigenvalues_display(eig)
            except Exception as e:
                st.warning(f"⚠️ **Non applicable** : {e}")
        else:
            if selected_point_data:
                px, py = selected_point_data['point']
                st.markdown("Linéarisation au point fixe sélectionné :")
                st.latex(f"J({fmt(px)},\\ {fmt(py)}) = {matrix_latex(selected_point_data['jacobian'])}")
                eig_temp = {
                    'valeurs_propres': selected_point_data['valeurs_propres'],
                    'classification':  selected_point_data['classification'],
                    'nature':          selected_point_data['nature'],
                }
                format_eigenvalues_display(eig_temp, local=True)
            else:
                st.warning("⚠️ Aucun point fixe disponible.")

    # ── Onglet 2 : Sous-espaces ─────────────────────────────────────────
    with tab2:
        if analyzer.is_linear:
            try:
                esp = analyzer.compute_eigenspaces(plot=False)
                format_subspaces_display(esp)
            except Exception as e:
                st.warning(f"⚠️ **Non applicable** : {e}")
        else:
            if selected_point_data:
                px, py = selected_point_data['point']
                st.markdown(f"Sous-espaces de la linéarisation au point fixe **({fmt(px)}, {fmt(py)})** :")
                format_subspaces_display(selected_point_data['eigenspaces'])
            else:
                st.warning("⚠️ Aucun point fixe disponible.")

    # ── Onglet 3 : Isoclines ────────────────────────────────────────────
    with tab3:
        st.markdown("*Les isoclines séparent les régions de directions opposées du flux.*")
        with st.spinner("Calcul des isoclines..."):
            fig_iso = analyzer.plot_isoclines(plot=False)
        st.pyplot(fig_iso, clear_figure=True)
        plt.close(fig_iso)

    # ── Onglet 4 : Portrait de phase ────────────────────────────────────
    with tab4:
        if not analyzer.is_linear and selected_point_data:
            view_mode = st.radio(
                "Mode d'affichage du portrait de phase :",
                ["🌍 Globale (tout le domaine)", "🔍 Locale (linéarisation autour du point sélectionné)"],
                horizontal=True,
                key="radio_phase_view"
            )
        else:
            view_mode = "🌍 Globale (tout le domaine)"

        if "Locale" in view_mode:
            st.markdown(f"*Portrait de phase local (linéarisé) autour du point **({fmt(selected_point_data['point'][0])}, {fmt(selected_point_data['point'][1])})**.*")
            with st.spinner("Génération du portrait de phase local..."):
                # Créer un analyseur temporaire pour le système linéarisé (matrice Jacobienne)
                # Le domaine est centré sur 0 pour montrer les variations locales
                local_analyzer = DynamicsAnalyzer(selected_point_data['jacobian'], domain=(-2, 2, -2, 2))
                fig_phase = local_analyzer.plot_final_phase_portrait(plot=False)
            st.pyplot(fig_phase, clear_figure=True)
            plt.close(fig_phase)
        else:
            st.markdown("*Synthèse complète : isoclines, trajectoires et analyse qualitative.*")
            with st.spinner("Génération du portrait de phase global..."):
                fig_phase = analyzer.plot_final_phase_portrait(plot=False)
            st.pyplot(fig_phase, clear_figure=True)
            plt.close(fig_phase)

    # ── Onglet 5 : Rapport PDF ───────────────────────────────────────────
    with tab5:
        st.markdown("Cliquez ci-dessous pour générer et télécharger le rapport PDF complet.")
        if st.button("📥 Générer le rapport PDF", type="primary", key="btn_generate_pdf"):
            with st.spinner("Génération du rapport en cours..."):
                try:
                    # Rapport généré en mémoire : pas de fichier partagé entre utilisateurs
                    buffer = io.BytesIO()
                    analyzer.generate_report(buffer)
                    st.download_button(
                        label="📥 Télécharger le rapport PDF",
                        data=buffer.getvalue(),
                        file_name="rapport_analyse.pdf",
                        mime="application/pdf",
                        type="primary",
                        key="btn_download_pdf",
                    )
                    st.success("✅ Rapport généré avec succès !")
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération : {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Pied de page
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🧑‍🏫 À propos")
st.markdown(
    "Cette application a été développée par **Mor DIOUF**, étudiant en **Master 2 Mathématiques Pures** "
    "à l'Université Assane Seck de Ziguinchor."
)
st.markdown(
    "Elle est conçue pour les étudiants de l'Université Assane Seck de Ziguinchor, "
    "ainsi que pour toute personne souhaitant apprendre les systèmes dynamiques."
)
st.markdown(
    "L'application suit la méthodologie du professeur pour tracer les portraits de phase, "
    "analyser les valeurs propres et résoudre les exercices."
)
