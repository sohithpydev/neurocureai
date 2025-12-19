import streamlit as st
import pandas as pd
from PIL import Image
import os
import base64
import pickle
import bz2
import numpy as np
import plotly.graph_objects as go

# =========================
# PaDEL
# =========================
from padelpy import padeldescriptor

# =========================
# RDKit (lightweight ADMET rules)
# =========================
from rdkit import Chem
from rdkit.Chem import Descriptors

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="NeuroCureAI",
    layout="wide"
)

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

    common_params = {
        'mol_dir': 'molecule.smi',
        'detectaromaticity': True,
        'standardizenitro': True,
        'standardizetautomers': True,
        'threads': 2,
        'removesalt': True,
        'log': False,
        'fingerprints': True
    }

    for name, xml in fp.items():
        padeldescriptor(
            d_file=f"{name}.csv",
            descriptortypes=f"./PaDEL-Descriptor/{xml}",
            **common_params
        )

    def load_fp_clean(path):
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset="Name", keep="first")
        return df.set_index("Name")

    fp_files = [
        "AtomPairs2D.csv",
        "CDK.csv",
        "CDKextended.csv",
        "CDKgraphonly.csv",
        "EState.csv",
        "KlekotaRoth.csv",
        "MACCS.csv",
        "PubChem.csv",
        "Substructure.csv"
    ]

    X = pd.concat([load_fp_clean(f) for f in fp_files], axis=1)
    X.reset_index().to_csv("descriptors_output.csv", index=False)

    for f in fp_files:
        os.remove(f)
    os.remove("molecule.smi")

# =========================
# Utilities
# =========================
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'

def load_qsar_model():
    with bz2.BZ2File("alzheimers_model.pbz2", "rb") as f:
        return pickle.load(f)

# =========================
# ADMET (rule-based, real chemistry)
# =========================
def compute_admet_rules(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot = Descriptors.NumRotatableBonds(mol)

    lipinski = int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    veber = int(tpsa <= 140 and rot <= 10)
    bbb = int(tpsa < 90 and logp >= 2)

    return {
        "Lipinski Pass": lipinski,
        "Veber Pass": veber,
        "BBB Likely": bbb,
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "HBD": hbd,
        "HBA": hba,
        "RotB": rot
    }

def admet_to_radar(admet):
    return {
        "Absorption": admet["Lipinski Pass"],
        "Permeability": admet["Veber Pass"],
        "BBB": admet["BBB Likely"],
        "Drug-likeness": int(admet["Lipinski Pass"] and admet["Veber Pass"]),
        "Safety Proxy": 1 - int(admet["LogP"] > 5)
    }

def plot_admet_radar(admet_dict):
    labels = list(admet_dict.keys())
    values = list(admet_dict.values())

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill='toself'
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1])),
        showlegend=False
    )
    return fig

# =========================
# Header
# =========================
st.image("logo.png", use_column_width=True)

st.markdown("""
# 🧠 NeuroCureAI  
### AI-Powered Platform for Alzheimer’s Drug Discovery

Predict **pIC₅₀** against **Amyloid Beta A4** and evaluate **ADMET-relevant properties**
using physicochemical and rule-based filters.
""")

st.info(
    "Predictions are for research purposes only and require experimental validation. "
    "ADMET properties are estimated using established early-stage drug-discovery rules."
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Upload Molecules")
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    st.markdown(
        "[Example input file]"
        "(https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)"
    )

# =========================
# Main logic
# =========================
if st.sidebar.button("Predict") and uploaded_file is not None:

    load_data = pd.read_table(uploaded_file, sep=" ", header=None)
    load_data.to_csv("molecule.smi", sep="\t", header=False, index=False)

    with st.spinner("Calculating molecular fingerprints..."):
        desc_calc()

    desc = pd.read_csv("descriptors_output.csv")
    Xlist = list(pd.read_csv("descriptor_list.csv").columns)
    desc_subset = desc[Xlist]

    model = load_qsar_model()
    preds = model.predict(desc_subset)

    results = pd.DataFrame({
        "Molecule": load_data[1],
        "SMILES": load_data[0],
        "Predicted pIC50": preds
    }).sort_values("Predicted pIC50", ascending=False)

    # =========================
    # Tabs
    # =========================
    tab1, tab2, tab3 = st.tabs(
        ["🔬 Bioactivity Prediction", "🧬 ADMET Analysis", "📊 Descriptors"]
    )

    # ---- Prediction tab ----
    with tab1:
        st.subheader("Prediction Results")
        st.dataframe(results)

        best = results.iloc[0]
        st.success(
            f"🏆 **Best Predicted Compound**\n\n"
            f"**{best['Molecule']}**\n\n"
            f"Predicted pIC₅₀: **{best['Predicted pIC50']:.2f}**"
        )

        st.markdown(filedownload(results), unsafe_allow_html=True)

    # ---- ADMET tab ----
    with tab2:
        st.subheader("ADMET Profile (Rule-Based)")

        selected = st.selectbox(
            "Select compound",
            results["Molecule"]
        )

        smiles = results.loc[
            results["Molecule"] == selected, "SMILES"
        ].values[0]

        admet_raw = compute_admet_rules(smiles)

        if admet_raw is None:
            st.error("Invalid SMILES — cannot compute ADMET")
        else:
            radar_vals = admet_to_radar(admet_raw)
            st.plotly_chart(
                plot_admet_radar(radar_vals),
                use_container_width=True
            )

            st.subheader("ADMET Details")
            st.json(admet_raw)

            st.caption(
                "ADMET properties estimated using Lipinski Rule of Five, "
                "Veber rules, and CNS permeability heuristics."
            )

    # ---- Descriptor tab ----
    with tab3:
        st.subheader("Descriptor Matrix Used for Prediction")
        st.write(desc_subset)
        st.write(desc_subset.shape)

# =========================
# Research experience
# =========================
st.markdown("---")
st.markdown("## 🌍 International Research Experience")

cols = st.columns(4)
images = [
    ("media/japan_lab.jpg", "Cryo-EM @ Japan 🇯🇵"),
    ("media/japan_lab_2.jpg", "Cryo-EM Analysis @ Japan 🇯🇵"),
    ("media/japan_lab_3.jpg", "XRC @ Japan 🇯🇵"),
    ("media/taiwan_lab_1.JPG", "AI & Bioinformatics Research – Taiwan 🇹🇼"),
]

for col, (img, cap) in zip(cols, images):
    with col:
        st.image(img, caption=cap, use_column_width=True)
        
# =========================
# Footer
# =========================
st.markdown("---")

col1, col2 = st.columns([1, 4])
with col1:
    st.image("sohith_dp.jpg", width=120)

with col2:
    st.markdown("""
    **Developed by:**  
    **K. Sohith Reddy**  
    UG Researcher | AI × Drug Discovery  

    📧 **Contact:** sohith.bme@gmail.com  
    🌍 International research experience: Japan 🇯🇵 · Taiwan 🇹🇼  
    """)
