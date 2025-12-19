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
# Page config
# =========================
st.set_page_config(page_title="NeuroCureAI", layout="wide")

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
    return f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'

# =========================
# ADMET (rule-based)
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
        "BBB Likely": int(tpsa < 90 and logp >= 2),
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2)
    }

def plot_admet_radar(d):
    labels = list(d.keys())
    values = list(d.values())
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), showlegend=False)
    return fig

# =========================
# Header
# =========================
st.image("logo.png", use_column_width=True)
st.markdown("# 🧠 NeuroCureAI\nAI-Powered Platform for Alzheimer’s Drug Discovery")

# =========================
# Sidebar
# =========================
with st.sidebar:
    uploaded = st.file_uploader("Upload molecule file (.txt)", type=["txt"])
    if st.button("Predict") and uploaded is not None:
        st.session_state.clear()
        st.session_state["run"] = True
        st.session_state["input_df"] = pd.read_table(uploaded, sep=" ", header=None)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🔬 Prediction", "🧬 ADMET"])

# =========================
# Prediction tab
# =========================
with tab1:
    if st.session_state.get("run", False):

        input_df = st.session_state["input_df"]
        st.subheader("Input Molecules")
        st.dataframe(input_df)

        input_df.to_csv("molecule.smi", sep="\t", index=False, header=False)

        with st.spinner("Calculating molecular descriptors…"):
            desc_calc()

        desc = pd.read_csv("descriptors_output.csv")
        st.subheader("Calculated Molecular Descriptors")
        st.dataframe(desc)

        Xlist = list(pd.read_csv("descriptor_list.csv").columns)
        desc_subset = desc[Xlist]

        st.subheader("Descriptor Subset Used by Model")
        st.dataframe(desc_subset.iloc[:, :50])  # limit columns for UI

        model = load_model()
        preds = model.predict(desc_subset)

        results = pd.DataFrame({
            "Molecule": input_df[1],
            "SMILES": input_df[0],
            "Predicted pIC50": preds
        }).sort_values("Predicted pIC50", ascending=False)

        st.subheader("Prediction Output")
        st.dataframe(results)

        st.success(
            f"Best predicted compound: **{results.iloc[0]['Molecule']}** "
            f"(pIC₅₀ = {results.iloc[0]['Predicted pIC50']:.2f})"
        )

        st.markdown(filedownload(results), unsafe_allow_html=True)
        st.session_state["results"] = results

# =========================
# ADMET tab
# =========================
with tab2:
    if "results" in st.session_state:
        results = st.session_state["results"]
        mol = st.selectbox("Select compound", results["Molecule"])
        smi = results.loc[results["Molecule"] == mol, "SMILES"].values[0]
        admet = compute_admet(smi)

        if admet:
            st.plotly_chart(plot_admet_radar(admet), use_container_width=True)
            st.json(admet)

# =========================
# Research context
# =========================
st.markdown("---")
st.markdown("## 🔬 Bridging Computational Prediction with Experimental Research Environments")

cols = st.columns(4)
images = [
    ("media/japan_lab.jpg", "Cryo-EM data generation"),
    ("media/japan_lab_2.jpg", "Cryo-EM structural analysis"),
    ("media/japan_lab_3.jpg", "X-ray crystallography workflows"),
    ("media/taiwan_lab_1.JPG", "AI-driven bioinformatics research"),
]

for c, (img, cap) in zip(cols, images):
    with c:
        st.image(img, caption=cap, use_column_width=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.image("sohith_dp.jpg", width=120)
st.markdown("**Developed by:** K. Sohith Reddy  \n📧 sohith.bme@gmail.com")
