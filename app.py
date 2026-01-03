import streamlit as st
import pandas as pd
from PIL import Image
import os
import base64
import pickle
import bz2
import plotly.graph_objects as go

from padelpy import padeldescriptor
from rdkit import Chem
from rdkit.Chem import Descriptors

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="NeuroCureAI",
    page_icon="🧠",
    layout="wide"
)

# =========================
# HERO SECTION
# =========================
st.markdown(
    """
    <div style="text-align:center; padding: 30px 10px;">
        <h1 style="font-size:3rem;">🧠 NeuroCureAI</h1>
        <p style="font-size:1.2rem; color:#555;">
        AI-Powered Drug Discovery Platform for Alzheimer’s Disease
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("media/hero_brain_ai.png", use_column_width=True)

# =========================
# HOW TO USE
# =========================
st.markdown("---")
st.markdown("## 🚀 How to Use NeuroCureAI")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Upload Molecules")
    st.markdown(
        """
        Upload a `.txt` file containing **SMILES and molecule name/ID**
        separated by space.

        Example:
        ```
        CC(=O)Oc1ccccc1 Aspirin 
        ```
        """
    )

with col2:
    st.markdown("### 2️⃣ Run AI Prediction")
    st.markdown(
        """
        Click **Run Prediction** to:
        - Generate molecular fingerprints  
        - Apply trained ML model  
        - Rank compounds by predicted potency
        """
    )

with col3:
    st.markdown("### 3️⃣ Explore ADMET")
    st.markdown(
        """
        Analyze drug-likeness using:
        - Lipinski & Veber rules  
        - BBB likelihood  
        - Interactive radar visualization
        """
    )

# =========================
# WORKFLOW IMAGE
# =========================
st.markdown("---")
st.markdown("## 🔁 AI-Driven Drug Discovery Workflow")
st.image("media/workflow.jpg", use_container_width=True)
st.caption("From molecular structure to AI-based activity prediction and ADMET evaluation")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🛠 Control Panel")
    st.markdown("Upload molecule file and start analysis")

    uploaded = st.file_uploader(
        "📄 Upload molecule file (.txt)",
        type=["txt"]
    )

    if st.button("🚀 Run Prediction") and uploaded is not None:
        st.session_state.clear()
        st.session_state["run"] = True
        st.session_state["input_df"] = pd.read_table(uploaded, sep=" ", header=None)

# =========================
# Descriptor calculation
# =========================
def desc_calc():
    fp = {
        'AtomPairs2D': 'AtomPairs2DFingerprinter.xml',
        'CDK': 'Fingerprinter.xml',
        'CDKextended': 'ExtendedFingerprinter.xml',
        'CDKgraphonly': 'GraphOnlyFingerprinter.xml',
        'EState': 'EStateFingerprinter.xml',
        'KlekotaRoth': 'KlekotaRothFingerprinter.xml',
        'MACCS': 'MACCSFingerprinter.xml',
        'PubChem': 'PubchemFingerprinter.xml',
        'Substructure': 'SubstructureFingerprinter.xml'
    }

    common_params = dict(
        mol_dir='molecule.smi',
        detectaromaticity=True,
        standardizenitro=True,
        standardizetautomers=True,
        threads=2,
        removesalt=True,
        log=False,
        fingerprints=True
    )

    for name, xml in fp.items():
        padeldescriptor(
            d_file=f"{name}.csv",
            descriptortypes=f"./PaDEL-Descriptor/{xml}",
            **common_params
        )

    def load_fp_clean(path):
        df = pd.read_csv(path)
        return df.drop_duplicates("Name").set_index("Name")

    fps = [
        "AtomPairs2D.csv", "CDK.csv", "CDKextended.csv",
        "CDKgraphonly.csv", "EState.csv",
        "KlekotaRoth.csv", "MACCS.csv",
        "PubChem.csv", "Substructure.csv"
    ]

    X = pd.concat([load_fp_clean(f) for f in fps], axis=1)
    X.reset_index().to_csv("descriptors_output.csv", index=False)

    for f in fps:
        os.remove(f)
    os.remove("molecule.smi")

# =========================
# Utilities
# =========================
def load_model():
    with bz2.BZ2File("alzheimers_model.pbz2", "rb") as f:
        return pickle.load(f)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">📥 Download Predictions</a>'

# =========================
# ADMET calculation
# =========================
def compute_admet(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)

    return {
        "Lipinski": int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
        "Veber": int(tpsa <= 140 and rot <= 10),
        "BBB Likely": int(tpsa < 90 and logp >= 2)
    }

def plot_admet_radar(d):
    fig = go.Figure(go.Scatterpolar(
        r=list(d.values()),
        theta=list(d.keys()),
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False
    )
    return fig

# =========================
# MAIN TABS
# =========================
tab1, tab2 = st.tabs(["🔬 Prediction", "🧬 ADMET Analysis"])

# =========================
# Prediction Tab
# =========================
with tab1:
    st.markdown("## 🔬 Molecular Activity Prediction")

    if st.session_state.get("run", False):
        input_df = st.session_state["input_df"]

        st.info("📥 Molecule file uploaded successfully")

        with st.expander("📄 View Input Molecules"):
            st.dataframe(input_df)

        input_df.to_csv("molecule.smi", sep="\t", index=False, header=False)

        with st.spinner("🧪 Calculating molecular descriptors..."):
            desc_calc()

        st.success("✅ Descriptor calculation completed")

        desc = pd.read_csv("descriptors_output.csv")

        with st.expander("🧬 Descriptor Overview"):
            st.dataframe(desc.iloc[:, :40])

        Xlist = list(pd.read_csv("descriptor_list.csv").columns)
        desc_subset = desc[Xlist]

        model = load_model()
        preds = model.predict(desc_subset)

        results = pd.DataFrame({
            "Molecule": input_df[1],
            "SMILES": input_df[0],
            "Predicted pIC50": preds
        }).sort_values("Predicted pIC50", ascending=False)

        st.subheader("🏆 Prediction Results")
        st.dataframe(results)

        st.success(
            f"Best predicted compound: **{results.iloc[0]['Molecule']}** "
            f"(pIC₅₀ = {results.iloc[0]['Predicted pIC50']:.2f})"
        )

        st.markdown(filedownload(results), unsafe_allow_html=True)
        st.session_state["results"] = results

# =========================
# ADMET Tab
# =========================
with tab2:
    st.markdown("## 🧬 ADMET & Drug-Likeness Evaluation")
    st.markdown(
        "Assess pharmacokinetic suitability and BBB penetration potential."
    )

    if "results" in st.session_state:
        results = st.session_state["results"]
        mol = st.selectbox("Select compound", results["Molecule"])
        smi = results.loc[results["Molecule"] == mol, "SMILES"].values[0]

        admet = compute_admet(smi)
        if admet:
            st.plotly_chart(plot_admet_radar(admet), use_container_width=True)
            st.json(admet)

# =========================
# RESEARCH CONTEXT
# =========================
st.markdown("---")
st.markdown("## 🔗 Bridging AI with Benchwork")
st.image("media/portfolio.png", use_container_width=True)
st.caption("Integrating computational predictions with experimental validation")

# =========================
# USER REVIEWS SECTION
# =========================
st.markdown("---")
st.markdown("## 🌟 User Reviews & Feedback")

rev_col1, rev_col2, rev_col3 = st.columns(3)

with rev_col1:
    st.image("media/scott.jpeg", width=120) # Replace with your actual filename
    st.markdown("**Scott C. Schuyler**")
    st.markdown("⭐ 4.5/5")
    st.caption("Associate Professor, Chang Gung University, Taiwan")
    st.info("""“Excellent tool for lead optimization. We used NeuroCureAI to identify high-potential leads, and the transition from 'in silico' to 'in vitro' was seamless. The prediction model is highly reliable. Our lab results closely mirrored the platform's data. It is a powerful resource that has earned a permanent spot in our workflow.”""")

with rev_col2:
    st.image("media/toshiya.jpg", width=120) # Replace with your actual filename
    st.markdown("**Toshiya Senda**")
    st.markdown("⭐ 3.5/5")
    st.caption("Research Director, KEK, Japan")
    st.info("""“NeuroCureAI has changed the game for our lead discovery. I’m usually skeptical about AI in the lab, but our summer intern, Sohith, used the platform to create a model that provided results very close to our experimental values. He excelled at the lab work and then took it further with this technology. Sohith, you rock! For the next version, it would be great to see integrated MD simulations... We'll definitely use this for future tests!”""")

with rev_col3:
    st.image("media/brooks.png", width=120) # Replace with your actual filename
    st.markdown("**Brooks Robinson**")
    st.markdown("⭐ 4.2/5")
    st.caption("Program Director, UCCS, USA")
    st.info("""“NeuroCureAI has reduced our lead-picking time. This change allows the team to focus more on the actual science.”""")

# =========================
# FOOTER
# =========================
st.markdown("---")
col1, col2 = st.columns([1, 4])

with col1:
    st.image("sohith_dp.jpg", width=150)

with col2:
    st.markdown(
        """
        **Developed by:** [Sohith Reddy](https://sohithpydev.github.io/sohith/)  

        📧 **Contact:** sohith.bme@gmail.com  

        💡 *This platform demonstrates how AI can accelerate early-stage
        neurodegenerative drug discovery.*
        """
    )
