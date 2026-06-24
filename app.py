import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics_analyzer import DynamicsAnalyzer

def format_eigenvalues_display(eig_data):
    """Formate l'affichage des valeurs propres en style LaTeX"""
    st.markdown("### 📊 **Analyse des Valeurs Propres**")
    
    # Valeurs propres
    eigenvalues = eig_data['valeurs_propres']
    st.markdown("**Valeurs propres** $\\lambda$ :")
    
    if len(eigenvalues) == 2:
        lambda1, lambda2 = eigenvalues
        if np.iscomplexobj(lambda1) or np.iscomplexobj(lambda2):
            # Nombres complexes
            st.latex(f"\\lambda_1 = {lambda1:.3f}, \\quad \\lambda_2 = {lambda2:.3f}")
        else:
            # Nombres réels
            st.latex(f"\\lambda_1 = {lambda1:.3f}, \\quad \\lambda_2 = {lambda2:.3f}")
    else:
        # Cas général
        st.write(f"λ = {eigenvalues}")
    
    # Classification
    st.markdown("**Classification** :")
    for i, (val, cls) in enumerate(zip(eigenvalues, eig_data['classification'])):
        if np.iscomplexobj(val):
            if val.real < 0:
                color = "🟢"
            elif val.real > 0:
                color = "🔴"
            else:
                color = "🟡"
        else:
            if val < 0:
                color = "🟢"
            elif val > 0:
                color = "🔴"
            else:
                color = "🟡"
        
        st.markdown(f"- {color} $\\lambda_{i+1}$ : **{cls}**")
    
    # Nature du point fixe
    nature = eig_data['nature']
    st.markdown("**Nature du point fixe** :")
    if "stable" in nature.lower():
        st.success(f"🎯 **{nature}** - Le système converge vers le point fixe")
    elif "instable" in nature.lower():
        st.error(f"⚠️ **{nature}** - Le système diverge du point fixe")
    elif "selle" in nature.lower():
        st.warning(f"🔄 **{nature}** - Comportement mixte (stable/instable)")
    elif "centre" in nature.lower():
        st.info(f"🌀 **{nature}** - Oscillations périodiques")
    else:
        st.info(f"📊 **{nature}**")

def format_subspaces_display(esp_data):
    """Formate l'affichage des sous-espaces en style LaTeX"""
    st.markdown("### 🧮 **Analyse des Sous-espaces Propres**")
    
    # Sous-espace stable
    E_s_dim = esp_data['E_s_dim']
    E_s_basis = esp_data['E_s_basis']
    st.markdown("**Sous-espace Stable** $E_s$ :")
    st.markdown(f"- **Dimension** : $\\dim(E_s) = {E_s_dim}$")
    if E_s_basis:
        st.markdown("- **Base** :")
        for i, vec in enumerate(E_s_basis):
            st.latex(f"\\mathbf{{v}}_{i+1}^s = \\begin{{pmatrix}} {vec[0]:.3f} \\\\ {vec[1]:.3f} \\end{{pmatrix}}")
    else:
        st.markdown("- **Base** : $\\emptyset$")
    
    # Sous-espace instable
    E_u_dim = esp_data['E_u_dim']
    E_u_basis = esp_data['E_u_basis']
    st.markdown("**Sous-espace Instable** $E_u$ :")
    st.markdown(f"- **Dimension** : $\\dim(E_u) = {E_u_dim}$")
    if E_u_basis:
        st.markdown("- **Base** :")
        for i, vec in enumerate(E_u_basis):
            st.latex(f"\\mathbf{{v}}_{i+1}^u = \\begin{{pmatrix}} {vec[0]:.3f} \\\\ {vec[1]:.3f} \\end{{pmatrix}}")
    else:
        st.markdown("- **Base** : $\\emptyset$")
    
    # Sous-espace centre
    E_c_dim = esp_data['E_c_dim']
    E_c_basis = esp_data['E_c_basis']
    st.markdown("**Sous-espace Centre** $E_c$ :")
    st.markdown(f"- **Dimension** : $\\dim(E_c) = {E_c_dim}$")
    if E_c_basis:
        st.markdown("- **Base** :")
        for i, vec in enumerate(E_c_basis):
            st.latex(f"\\mathbf{{v}}_{i+1}^c = \\begin{{pmatrix}} {vec[0]:.3f} \\\\ {vec[1]:.3f} \\end{{pmatrix}}")
    else:
        st.markdown("- **Base** : $\\emptyset$")

def format_quadrant_analysis(quad_data):
    """Formate l'analyse par quadrant"""
    st.markdown("### 🧭 **Analyse par Quadrant**")
    
    # Tableau des quadrants
    st.markdown("**Tableau des quadrants** :")
    
    # En-tête du tableau
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("**Quadrant**")
    with col2:
        st.markdown("**Point**")
    with col3:
        st.markdown("**Signe (ẋ, ẏ)**")
    with col4:
        st.markdown("**Sens**")
    with col5:
        st.markdown("**Symbole**")
    
    for q in quad_data:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write(f"Q{q['quadrant']}")
        with col2:
            st.write(f"({q['point'][0]:.2f}, {q['point'][1]:.2f})")
        with col3:
            st.write(f"({q['sign'][0]:.0f}, {q['sign'][1]:.0f})")
        with col4:
            st.write(q['sens'])
        with col5:
            if q['sens'] == 'Convergent':
                st.write("🔄")
            elif q['sens'] == 'Divergent':
                st.write("📈")
            elif q['sens'] == 'Tournant':
                st.write("🌪️")
            else:
                st.write("📏")

st.set_page_config(
    page_title="Analyse Qualitative de Systèmes Dynamiques - Outil Interactif", 
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/AZ473/system-dynamics-analyzer',
        'Report a bug': 'https://github.com/AZ473/system-dynamics-analyzer/issues',
        'About': "Mory Diouf - Outil pédagogique pour l'analyse des systèmes dynamiques en 2D"
    }
)

# Métadonnées SEO
st.markdown("""
<meta name="description" content="Outil interactif gratuit pour analyser qualitativement les systèmes dynamiques linéaires et non-linéaires. Analyse des valeurs propres, vecteurs propres, isoclines, portraits de phase.">
<meta name="keywords" content="systèmes dynamiques, analyse qualitative, valeurs propres, vecteurs propres, portrait de phase, isoclines, stabilité, mathématiques appliquées">
<meta name="author" content="Mory Diouf">
<meta name="google-site-verification" content="IrUWbS1NZxO8bpVaINJJLL2vI9_nLcroUq8Ylu_Hv_w" />
""", unsafe_allow_html=True)

st.markdown("""
# 🔬 **Analyse Qualitative de Systèmes Dynamiques**

*Outil pédagogique interactif et gratuit pour l'analyse complète des systèmes dynamiques linéaires et non-linéaires en 2D*

**Fonctionnalités principales :**
- 📊 Analyse des valeurs propres et classification des points fixes
- 🧮 Calcul des vecteurs propres et sous-espaces (stable, instable, centre)
- 📈 Visualisation des isoclines orientées
- 🧭 Analyse par quadrant du comportement du système
- 🎨 Génération de portraits de phase complets
- 📄 Export de rapports PDF détaillés

---

## 🧑‍🏫 À propos

Cette application a été développée par **Mory Diouf**, étudiant en master de mathématiques pures.
Elle suit la méthodologie du professeur pour tracer les portraits de phase, analyser les valeurs propres et résoudre les exercices.
L'outil reprend la démarche du cours pour offrir une analyse pas à pas des systèmes dynamiques en 2D.

---

""")

st.sidebar.markdown("## ⚙️ **Configuration du Système**")
st.sidebar.markdown("""
---
## 📘 À propos
Créée par **Mory Diouf**, étudiant en master de mathématiques pures.
Basée sur la méthode du professeur, cette application suit une démarche pédagogique pour l'analyse qualitative des systèmes dynamiques en 2D.
""")
system_type = st.sidebar.selectbox("Type de système", ["Linéaire", "Non-linéaire", "Exemple prédéfini"])

if system_type == "Linéaire":
    st.sidebar.subheader("Matrice A (2x2)")
    a11 = st.sidebar.number_input("a11", value=1.0)
    a12 = st.sidebar.number_input("a12", value=0.0)
    a21 = st.sidebar.number_input("a21", value=0.0)
    a22 = st.sidebar.number_input("a22", value=1.0)
    domain = st.sidebar.text_input("Domaine [xmin xmax ymin ymax]", "-5 5 -5 5")
    if st.sidebar.button("Analyser"):
        A = np.array([[a11, a12], [a21, a22]])
        domain_vals = list(map(float, domain.split()))
        analyzer = DynamicsAnalyzer(A, tuple(domain_vals))
        st.session_state['analyzer'] = analyzer
elif system_type == "Non-linéaire":
    st.sidebar.subheader("Fonctions f₁(x₁,x₂), f₂(x₁,x₂)")
    f1 = st.sidebar.text_input("f₁(x₁,x₂)", "x1 - x2 + x1**2")
    f2 = st.sidebar.text_input("f₂(x₁,x₂)", "-x1 + 2*x2")
    domain = st.sidebar.text_input("Domaine [xmin xmax ymin ymax]", "-5 5 -5 5")
    if st.sidebar.button("Analyser"):
        def fun1(x1, x2):
            return eval(f1, {"x1": x1, "x2": x2, "np": np})
        def fun2(x1, x2):
            return eval(f2, {"x1": x1, "x2": x2, "np": np})
        domain_vals = list(map(float, domain.split()))
        analyzer = DynamicsAnalyzer((fun1, fun2), tuple(domain_vals))
        st.session_state['analyzer'] = analyzer
else:
    st.sidebar.subheader("Exemples prédéfinis")
    ex = st.sidebar.selectbox("Exemple", [
        "Nœud stable", "Selle", "Centre", "Spirale stable", "Système non-linéaire simple"
    ])
    if st.sidebar.button("Analyser"):
        if ex == "Nœud stable":
            A = np.array([[-2, 1], [1, -3]])
            domain = (-3, 3, -3, 3)
            analyzer = DynamicsAnalyzer(A, domain)
        elif ex == "Selle":
            A = np.array([[1, 2], [2, 1]])
            domain = (-3, 3, -3, 3)
            analyzer = DynamicsAnalyzer(A, domain)
        elif ex == "Centre":
            A = np.array([[0, -1], [1, 0]])
            domain = (-3, 3, -3, 3)
            analyzer = DynamicsAnalyzer(A, domain)
        elif ex == "Spirale stable":
            A = np.array([[-1, -2], [2, -1]])
            domain = (-3, 3, -3, 3)
            analyzer = DynamicsAnalyzer(A, domain)
        else:
            def f1(x1, x2): return x1*(1 - x1 - x2)
            def f2(x1, x2): return x2*(x1 - 0.5)
            domain = (0, 1.5, 0, 1.5)
            analyzer = DynamicsAnalyzer((f1, f2), domain)
        st.session_state['analyzer'] = analyzer

analyzer = st.session_state.get('analyzer', None)

if analyzer:
    st.markdown("---")
    st.markdown("## 🔍 **Étape 1 : Analyse des Valeurs Propres**")
    try:
        eig = analyzer.analyze_eigenvalues()
        format_eigenvalues_display(eig)
    except Exception as e:
        st.warning(f"⚠️ **Non applicable** : {e}")
    
    st.markdown("---")
    st.markdown("## 🧮 **Étape 2 : Vecteurs Propres et Sous-espaces**")
    try:
        esp = analyzer.compute_eigenspaces(plot=False)
        format_subspaces_display(esp)
    except Exception as e:
        st.warning(f"⚠️ **Non applicable** : {e}")
    
    st.markdown("---")
    st.markdown("## 📊 **Étape 3 : Isoclines Orientées**")
    st.markdown("*Les isoclines séparent les régions de directions opposées du flux*")
    analyzer.plot_isoclines(plot=False)
    st.pyplot(plt.gcf())
    plt.close()
    
    st.markdown("---")
    st.markdown("## 🧭 **Étape 4 : Analyse par Quadrant**")
    quad = analyzer.quadrant_analysis()
    format_quadrant_analysis(quad)
    
    st.markdown("---")
    st.markdown("## 🎨 **Étape 5 : Portrait de Phase Final**")
    st.markdown("*Synthèse complète : isoclines, trajectoires, et analyse qualitative*")
    analyzer.plot_final_phase_portrait(plot=False)
    st.pyplot(plt.gcf())
    plt.close()
    
    st.markdown("---")
    st.markdown("## 📄 **Rapport PDF**")
    if st.button("📥 **Générer le rapport PDF**", type="primary"):
        with st.spinner("Génération du rapport en cours..."):
            filename = analyzer.generate_report()
            with open(filename, "rb") as f:
                st.download_button(
                    "📥 **Télécharger le rapport PDF**", 
                    f, 
                    file_name=filename, 
                    mime="application/pdf",
                    type="primary"
                )
