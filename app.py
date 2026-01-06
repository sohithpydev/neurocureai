import streamlit as st
import pandas as pd
from PIL import Image
import os
import base64
import pickle
import bz2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
# EMAIL BACKEND
# =========================
def send_feedback_email(name, designation, rating, feedback):
    sender_email = "sohith.bme@gmail.com" 
    receiver_email = "sohith.bme@gmail.com"
    # Authenticated with your provided App Password
    password = "nlso orfq xnaa dzbd" 

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"NeuroCureAI Feedback: {rating}/5 Stars from {name}"

    body = f"New Review Received:\nName: {name}\nDesignation: {designation}\nRating: {rating}/5\nFeedback: {feedback}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email Error: {e}")
        return False

# =========================
# CHEMISTRY & ML LOGIC
# =========================
def desc_calc():
    fp = {'PubChem': 'PubchemFingerprinter.xml'}
    common_params = dict(mol_dir='molecule.smi', detectaromaticity=True, standardizenitro=True,
                        standardizetautomers=True, threads=2, removesalt=True, log=False, fingerprints=True)
    for name, xml in fp.items():
        padeldescriptor(d_file=f"{name}.csv", descriptortypes=f"./PaDEL-Descriptor/{xml}", **common_params)
    df = pd.read_csv("PubChem.csv").drop_duplicates("Name").set_index("Name")
    df.reset_index().to_csv("descriptors_output.csv", index=False)
    os.remove("PubChem.csv")
    os.remove("molecule.smi")

def load_model():
    with bz2.BZ2File("alzheimers_model.pbz2", "rb") as f:
        return pickle.load(f)

def compute_admet(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return {
        "Lipinski": int(Descriptors.MolWt(mol) <= 500 and Descriptors.MolLogP(mol) <= 5),
        "Veber": int(Descriptors.TPSA(mol) <= 140 and Descriptors.NumRotatableBonds(mol) <= 10),
        "BBB Likely": int(Descriptors.TPSA(mol) < 90 and Descriptors.MolLogP(mol) >= 2)
    }

def plot_admet_radar(d):
    fig = go.Figure(go.Scatterpolar(r=list(d.values()), theta=list(d.keys()), fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
    return fig

# =========================
# TABS NAVIGATION
# =========================
tab_home, tab_workflow, tab_discovery, tab_reviews, tab_contact = st.tabs([
    "🏠 Dashboard", "🔄 Pipeline", "🔬 Discovery Engine", "🌟 Testimonials", "📞 Inquiry"
])

# =========================
# 1. DASHBOARD (HOME)
# =========================
with tab_home:
    st.markdown("<h1 style='text-align:center;'>🧠 NeuroCureAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI-Powered Drug Discovery Platform for Alzheimer’s Disease</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("media/hero_brain_ai.png", use_column_width=True)

    st.markdown("---")
    st.markdown("## 🔗 Bridging AI with Benchwork")
    st.image("media/portfolio.png", use_container_width=True)
    st.caption("Integrating computational predictions with experimental validation")

# =========================
# 2. PIPELINE (WORKFLOW)
# =========================
with tab_workflow:
    st.header("🔁 AI-Driven Drug Discovery Workflow")
    st.image("media/workflow.jpg", use_container_width=True)

# =========================
# 3. DISCOVERY ENGINE
# =========================
with tab_discovery:
    st.header("🔬 Activity Prediction & ADMET Profiling")
    
    # Control section moved from sidebar to main page area
    st.markdown("### 🛠️ Control Panel")
    with st.container(border=True):
        uploaded = st.file_uploader("Upload molecule file (.txt)", type=["txt"])
        run_btn = st.button("🚀 Run Prediction Analysis")

    if run_btn and uploaded is not None:
        st.session_state["run"] = True
        st.session_state["input_df"] = pd.read_table(uploaded, sep=" ", header=None)

    if st.session_state.get("run", False):
        res_tab1, res_tab2 = st.tabs(["Potency Analysis", "ADMET Screening"])
        input_df = st.session_state["input_df"]
        input_df.to_csv("molecule.smi", sep="\t", index=False, header=False)
        
        with st.spinner("Calculating molecular descriptors..."):
            desc_calc()
            desc = pd.read_csv("descriptors_output.csv")
            Xlist = list(pd.read_csv("descriptor_list.csv").columns)
            model = load_model()
            preds = model.predict(desc[Xlist])
            results = pd.DataFrame({
                "Molecule": input_df[1], 
                "SMILES": input_df[0], 
                "Predicted pIC50": preds
            }).sort_values("Predicted pIC50", ascending=False)
        
        with res_tab1:
            st.subheader("Predicted Bioactivity (pIC50)")
            st.dataframe(results, use_container_width=True)
            
        with res_tab2:
            st.subheader("Pharmacokinetic Profiling")
            sel = st.selectbox("Select Compound for Radar Analysis", results["Molecule"])
            smi = results.loc[results["Molecule"] == sel, "SMILES"].values[0]
            st.plotly_chart(plot_admet_radar(compute_admet(smi)), use_container_width=True)

# =========================
# 4. TESTIMONIALS (REVIEWS)
# =========================
with tab_reviews:
    st.header("🌟 User Reviews")
    
    r1, r2, r3 = st.columns(3)
    with r1:
        # Set width to 150 for all to match size
        st.image("media/scott.jpeg", width=150) 
        st.markdown("**Scott C. Schuyler**")
        st.markdown("⭐ 4.5/5")
        st.caption("Associate Professor, Chang Gung University, Taiwan")
        st.info("“Excellent tool for lead optimization. The transition from 'in silico' to 'in vitro' was seamless.”")

    with r2:
        st.image("media/toshiya.jpg", width=150) 
        st.markdown("**Toshiya Senda**")
        st.markdown("⭐ 3.5/5")
        st.caption("Research Director, KEK, Japan")
        st.info("“NeuroCureAI has changed the game for our lead discovery. Sohith, you rock!”")

    with r3:
        st.image("media/brooks.png", width=150) 
        st.markdown("**Brooks Robinson**")
        st.markdown("⭐ 4.2/5")
        st.caption("Program Director, UCCS, USA")
        st.info("“NeuroCureAI has reduced our lead-picking time, allowing the team to focus more on the science.”")

    st.divider()
    st.subheader("✍️ Submit Your Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name")
        desig = st.text_input("Designation")
        rating = st.slider("Rating", 1, 5, 5)
        msg = st.text_area("Feedback")
        if st.form_submit_button("Send Feedback"):
            if name and msg:
                if send_feedback_email(name, desig, rating, msg):
                    st.success("Feedback sent successfully!")
                    st.balloons()
            else:
                st.warning("Please fill in required fields.")

# =========================
# 5. INQUIRY (CONTACT)
# =========================
with tab_contact:
    st.header("📞 Developer Profile")
    c1, c2 = st.columns([1, 4])
    with c1:
        st.image("sohith_dp.jpg", width=200)
    with c2:
        st.markdown("### **Sohith Reddy**")
        st.markdown("""
        Final year **B.E. Bioinformatics Undergraduate** at **Saveetha School of Engineering (SSE)**.  
        My research focus lies at the intersection of **Artificial Intelligence** and **Drug Discovery**.
        
        **Current Position:** Research Assistant at **Chang Gung University (CGU)**, Taoyuan, Taiwan.
        """)
        st.markdown("---")
        st.markdown("**Portfolio:** [sohithpydev.github.io/sohith/](https://sohithpydev.github.io/sohith/)")
        st.markdown("📧 **Contact:** sohith.bme@gmail.com")
