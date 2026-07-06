import streamlit as st
import numpy as np
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
        'Get Help': 'https://github.com/AZ473/system-dynamics-analyzer',
        'Report a bug': 'https://github.com/AZ473/system-dynamics-analyzer/issues',
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

def format_eigenvalues_display(eig_data):
    """Formate l'affichage des valeurs propres en style LaTeX."""
    st.markdown("### 📊 **Analyse des Valeurs Propres**")
    eigenvalues = eig_data['valeurs_propres']
    st.markdown("**Valeurs propres** $\\lambda$ :")
    if len(eigenvalues) == 2:
        lambda1, lambda2 = eigenvalues
        st.latex(f"\\lambda_1 = {lambda1:.3f}, \\quad \\lambda_2 = {lambda2:.3f}")
    else:
        st.write(f"λ = {eigenvalues}")

    st.markdown("**Classification** :")
    for i, (val, cls) in enumerate(zip(eigenvalues, eig_data['classification'])):
        if np.iscomplexobj(val):
            color = "🟢" if val.real < 0 else ("🔴" if val.real > 0 else "🟡")
        else:
            color = "🟢" if val < 0 else ("🔴" if val > 0 else "🟡")
        st.markdown(f"- {color} $\\lambda_{{{i+1}}}$ : **{cls}**")

    nature = eig_data['nature']
    st.markdown("**Nature du point fixe** :")
    nl = nature.lower()
    if "stable" in nl and "instable" not in nl:
        st.success(f"🎯 **{nature}** - Le système converge vers le point fixe")
    elif "instable" in nl:
        st.error(f"⚠️ **{nature}** - Le système diverge du point fixe")
    elif "selle" in nl:
        st.warning(f"🔄 **{nature}** - Comportement mixte (stable/instable)")
    elif "centre" in nl:
        st.info(f"🌀 **{nature}** - Oscillations périodiques")
    else:
        st.info(f"📊 **{nature}**")


def format_subspaces_display(esp_data):
    """Formate l'affichage des sous-espaces en style LaTeX."""
    st.markdown("### 🧮 **Analyse des Sous-espaces Propres**")

    def show_space(label, dim, basis):
        st.markdown(f"**{label}** :")
        st.markdown(f"- **Dimension** : $\\dim = {dim}$")
        if basis:
            st.markdown("- **Base** :")
            for i, vec in enumerate(basis):
                st.latex(f"\\mathbf{{v}}_{i+1} = \\begin{{pmatrix}} {vec[0]:.3f} \\\\ {vec[1]:.3f} \\end{{pmatrix}}")
        else:
            st.markdown("- **Base** : $\\emptyset$")

    show_space("Sous-espace Stable $E_s$",  esp_data['E_s_dim'], esp_data['E_s_basis'])
    show_space("Sous-espace Instable $E_u$", esp_data['E_u_dim'], esp_data['E_u_basis'])
    show_space("Sous-espace Centre $E_c$",  esp_data['E_c_dim'], esp_data['E_c_basis'])


# ═══════════════════════════════════════════════════════════════════════════
# En-tête principal
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
# 🔬 **Analyse Qualitative de Systèmes Dynamiques**

*Outil pédagogique interactif et gratuit pour l'analyse complète des systèmes dynamiques linéaires et non-linéaires en 2D*

**Fonctionnalités principales :**
- 📊 Analyse des valeurs propres et classification des points fixes
- 🧮 Calcul des vecteurs propres et sous-espaces (stable, instable, centre)
- 📈 Visualisation des isoclines orientées
- 🎨 Génération de portraits de phase complets
- 📄 Export de rapports PDF détaillés

---
""")


# ═══════════════════════════════════════════════════════════════════════════
# Barre latérale — Configuration (clés uniques sur tous les widgets)
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## ⚙️ **Configuration du Système**")

system_type = st.sidebar.selectbox(
    "Type de système",
    ["Linéaire", "Non-linéaire", "Exemple prédéfini"],
    key="sb_system_type"
)

# --- Formulaire selon le type (on garde les mêmes widgets même si vides) ---
# Linéaire
with st.sidebar.expander("📐 Matrice A (2×2)", expanded=(system_type == "Linéaire")):
    a11 = st.number_input("a₁₁", value=1.0, key="ni_a11")
    a12 = st.number_input("a₁₂", value=0.0, key="ni_a12")
    a21 = st.number_input("a₂₁", value=0.0, key="ni_a21")
    a22 = st.number_input("a₂₂", value=1.0, key="ni_a22")
    domain_lin = st.text_input("Domaine [xmin xmax ymin ymax]", "-5 5 -5 5", key="ti_domain_lin")
    btn_linear = st.button("▶ Analyser (Linéaire)", key="btn_linear", disabled=(system_type != "Linéaire"))

# Non-linéaire
with st.sidebar.expander("📐 Fonctions f₁, f₂", expanded=(system_type == "Non-linéaire")):
    f1_input = st.text_input("f₁(x₁, x₂)", "x1 - x2 + x1**2", key="ti_f1")
    f2_input = st.text_input("f₂(x₁, x₂)", "-x1 + 2*x2", key="ti_f2")
    domain_nl = st.text_input("Domaine [xmin xmax ymin ymax]", "-5 5 -5 5", key="ti_domain_nl")
    btn_nonlin = st.button("▶ Analyser (Non-linéaire)", key="btn_nonlin", disabled=(system_type != "Non-linéaire"))

# Exemples prédéfinis
EXEMPLES = ["Nœud stable", "Selle", "Centre", "Spirale stable", "Système non-linéaire simple"]
with st.sidebar.expander("📚 Exemples prédéfinis", expanded=(system_type == "Exemple prédéfini")):
    ex_choice = st.selectbox("Exemple", EXEMPLES, key="sb_example")
    btn_example = st.button("▶ Charger l'exemple", key="btn_example", disabled=(system_type != "Exemple prédéfini"))


# ═══════════════════════════════════════════════════════════════════════════
# Logique d'analyse (construction de l'analyseur)
# ═══════════════════════════════════════════════════════════════════════════

def build_analyzer_linear():
    A = np.array([[a11, a12], [a21, a22]])
    domain_vals = tuple(map(float, domain_lin.split()))
    return DynamicsAnalyzer(A, domain_vals)

def build_analyzer_nonlinear():
    domain_vals = tuple(map(float, domain_nl.split()))
    return DynamicsAnalyzer((f1_input, f2_input), domain_vals)

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
        error_msg = f"Erreur (non-linéaire) : {e}"

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
    # ── Sélection du point fixe (non-linéaire) ──────────────────────────
    selected_point_data = None
    if not analyzer.is_linear:
        st.markdown("---")
        st.markdown("## 📍 **Points Fixes Réels Détectés**")
        if analyzer.fixed_points_data:
            n_pts = len(analyzer.fixed_points_data)
            st.success(f"Le système non-linéaire possède **{n_pts}** point(s) fixe(s) réel(s) dans le domaine.")
            options = [
                f"Point #{i+1} : ({pt['point'][0]:.3f}, {pt['point'][1]:.3f}) — {pt['nature']}"
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
        "🔍 Étape 1 : Valeurs propres",
        "🧮 Étape 2 : Sous-espaces",
        "📊 Étape 3 : Isoclines",
        "🎨 Étape 4 : Portrait de phase",
        "📄 Rapport PDF",
    ])

    # ── Onglet 1 : Valeurs propres ───────────────────────────────────────
    with tab1:
        st.markdown("---")
        if analyzer.is_linear:
            try:
                eig = analyzer.analyze_eigenvalues()
                format_eigenvalues_display(eig)
            except Exception as e:
                st.warning(f"⚠️ **Non applicable** : {e}")
        else:
            if selected_point_data:
                px, py = selected_point_data['point']
                st.markdown(f"Analyse linéarisée au point fixe **({px:.3f}, {py:.3f})** :")
                eig_temp = {
                    'valeurs_propres': selected_point_data['valeurs_propres'],
                    'classification':  selected_point_data['classification'],
                    'nature':          selected_point_data['nature'],
                }
                format_eigenvalues_display(eig_temp)
            else:
                st.warning("⚠️ Aucun point fixe disponible.")

    # ── Onglet 2 : Sous-espaces ─────────────────────────────────────────
    with tab2:
        st.markdown("---")
        if analyzer.is_linear:
            try:
                esp = analyzer.compute_eigenspaces(plot=False)
                format_subspaces_display(esp)
            except Exception as e:
                st.warning(f"⚠️ **Non applicable** : {e}")
        else:
            if selected_point_data:
                px, py = selected_point_data['point']
                st.markdown(f"Sous-espaces de la linéarisation locale au point fixe **({px:.3f}, {py:.3f})** :")
                format_subspaces_display(selected_point_data['eigenspaces'])
            else:
                st.warning("⚠️ Aucun point fixe disponible.")

    # ── Onglet 3 : Isoclines ────────────────────────────────────────────
    with tab3:
        st.markdown("---")
        st.markdown("*Les isoclines séparent les régions de directions opposées du flux.*")
        with st.spinner("Calcul des isoclines..."):
            fig_iso = analyzer.plot_isoclines(plot=False)
        st.pyplot(fig_iso, clear_figure=True)
        plt.close(fig_iso)

    # ── Onglet 4 : Portrait de phase ────────────────────────────────────
    with tab4:
        st.markdown("---")
        
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
            st.markdown(f"*Portrait de phase local (linéarisé) autour du point **({selected_point_data['point'][0]:.3f}, {selected_point_data['point'][1]:.3f})**.*")
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
        st.markdown("---")
        st.markdown("Cliquez ci-dessous pour générer et télécharger le rapport PDF complet.")
        if st.button("📥 Générer le rapport PDF", type="primary", key="btn_generate_pdf"):
            with st.spinner("Génération du rapport en cours..."):
                try:
                    filename = analyzer.generate_report()
                    with open(filename, "rb") as f_pdf:
                        st.download_button(
                            label="📥 Télécharger le rapport PDF",
                            data=f_pdf,
                            file_name=filename,
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