import streamlit as st
import pandas as pd
import os
import pickle
import bz2
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

from padelpy import padeldescriptor
from rdkit import Chem
from rdkit.Chem import Descriptors

# ==========================================
# PAGE CONFIGURATION & THEMING
# ==========================================
st.set_page_config(
    page_title="NeuroCureAI | Precision Drug Discovery",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS to move from "Entry Level" to "Design School" standards
st.markdown("""
    <style>
    /* Global Background and Typography */
    .main {
        background-color: #fcfcfc;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    
    /* Glassmorphism Card Style */
    div[data-testid="stExpander"], .stContainer, div[data-testid="stForm"] {
        border: none !important;
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(12px);
        border-radius: 24px !important;
        box-shadow: 0 10px 40px 0 rgba(31, 38, 135, 0.05) !important;
        padding: 25px;
        margin-bottom: 20px;
    }

    /* Modern Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #555 !important;
        transition: all 0.3s ease;
    }
    button[data-baseweb="tab"]:hover {
        color: #2a5298 !important;
        transform: translateY(-2px);
    }

    /* Professional Gradient Title */
    .title-text {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.8rem;
        letter-spacing: -1.5px;
    }

    /* Image Styling */
    img {
        border-radius: 18px;
        transition: transform 0.4s ease;
    }
    img:hover {
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# UTILITIES & ASSET LOADERS
# ==========================================
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# Animation for processing
lottie_running = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_96bov8.json")

def send_feedback_email(name, designation, rating, feedback):
    sender_email = "sohith.bme@gmail.com" 
    receiver_email = "sohith.bme@gmail.com"
    password = "nlso orfq xnaa dzbd" #
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
    except: return False

# ==========================================
# CHEMISTRY & ML ENGINE
# ==========================================
def desc_calc():
    fp = {
        'AtomPairs2D': 'AtomPairs2DFingerprinter.xml', 'CDK': 'Fingerprinter.xml',
        'CDKextended': 'ExtendedFingerprinter.xml', 'CDKgraphonly': 'GraphOnlyFingerprinter.xml',
        'EState': 'EStateFingerprinter.xml', 'KlekotaRoth': 'KlekotaRothFingerprinter.xml',
        'MACCS': 'MACCSFingerprinter.xml', 'PubChem': 'PubchemFingerprinter.xml',
        'Substructure': 'SubstructureFingerprinter.xml'
    } #
    common_params = dict(mol_dir='molecule.smi', detectaromaticity=True, standardizenitro=True,
                        standardizetautomers=True, threads=2, removesalt=True, log=False, fingerprints=True)
    for name, xml in fp.items():
        padeldescriptor(d_file=f"{name}.csv", descriptortypes=f"./PaDEL-Descriptor/{xml}", **common_params)
    def load_fp_clean(path):
        df = pd.read_csv(path)
        return df.drop_duplicates("Name").set_index("Name")
    fps = [f"{name}.csv" for name in fp.keys()]
    X = pd.concat([load_fp_clean(f) for f in fps], axis=1) #
    X.reset_index().to_csv("descriptors_output.csv", index=False)
    for f in fps: os.remove(f)
    os.remove("molecule.smi")

def load_model():
    with bz2.BZ2File("alzheimers_model.pbz2", "rb") as f:
        return pickle.load(f)

def compute_admet(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return {
        "Lipinski": int(Descriptors.MolWt(mol) <= 500 and Descriptors.MolLogP(mol) <= 5), #
        "Veber": int(Descriptors.TPSA(mol) <= 140 and Descriptors.NumRotatableBonds(mol) <= 10), #
        "BBB Likely": int(Descriptors.TPSA(mol) < 90 and Descriptors.MolLogP(mol) >= 2) #
    }

def plot_admet_radar(d):
    fig = go.Figure(go.Scatterpolar(r=list(d.values()), theta=list(d.keys()), fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
    return fig

# ==========================================
# MAIN NAVIGATION (Professional Names)
# ==========================================
tab_home, tab_workflow, tab_discovery, tab_reviews, tab_contact = st.tabs([
    "🏠 Dashboard", "🔄 Pipeline", "🔬 Discovery Engine", "🌟 Testimonials", "📞 Inquiry"
])

# 1. DASHBOARD
with tab_home:
    st.markdown('<h1 class="title-text" style="text-align:center;">NeuroCureAI</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.3rem; color:#666;'>Computational lead discovery for Alzheimer's Research.</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2: st.image("media/hero_brain_ai.png", use_column_width=True) #
    st.markdown("---")
    st.markdown("## 🔗 Bridging AI with Benchwork")
    st.image("media/portfolio.png", use_container_width=True) #

# 2. PIPELINE
with tab_workflow:
    st.header("🔁 Research Methodology")
    st.image("media/workflow.jpg", use_container_width=True) #

# 3. DISCOVERY ENGINE (With Animated Runner)
with tab_discovery:
    st.header("🔬 AI-Driven Molecular Discovery")
    with st.container():
        uploaded = st.file_uploader("Upload molecule file (.txt)", type=["txt"]) #
        run_btn = st.button("🚀 Execute Discovery Algorithm")

    if run_btn and uploaded is not None:
        st.session_state["run"] = True
        st.session_state["input_df"] = pd.read_table(uploaded, sep=" ", header=None) #

    if st.session_state.get("run", False):
        # ANIMATED RUNNER OVERLAY
        anim_placeholder = st.empty()
        with anim_placeholder.container():
            st.markdown("### 🧬 Algorithm Processing...")
            st_lottie(lottie_running, height=300, key="runner")
            st.info("Calculating comprehensive molecular fingerprints and descriptors...")
            desc_calc() #
            st.success("✅ Computational descriptors generated successfully.")

        anim_placeholder.empty() # Remove animation
        
        # SHOWCASE DATA TRANSPARENCY
        res_tab1, res_tab2, res_tab3 = st.tabs(["🧬 Data Transparency", "🏆 Predicted Activity", "🩸 ADMET Profile"])
        
        desc = pd.read_csv("descriptors_output.csv") #
        Xlist = list(pd.read_csv("descriptor_list.csv").columns) #
        
        with res_tab1:
            st.subheader("Raw Molecular Descriptors")
            st.write("A comprehensive matrix of all calculated chemical features:")
            st.dataframe(desc.head(10), use_container_width=True)
            
            st.subheader("Refined Model Features (Xlist Subset)")
            st.write("Specific features prioritized by the Random Forest model for prediction accuracy:")
            st.dataframe(desc[Xlist].head(10), use_container_width=True)

        with res_tab2:
            model = load_model() #
            preds = model.predict(desc[Xlist]) #
            results = pd.DataFrame({
                "Molecule": st.session_state["input_df"][1], 
                "SMILES": st.session_state["input_df"][0], 
                "Predicted pIC50": preds
            }).sort_values("Predicted pIC50", ascending=False)
            st.subheader("Lead Compound Ranking")
            st.dataframe(results, use_container_width=True)

        with res_tab3:
            st.subheader("Drug-Likeness & BBB Likelihood")
            sel = st.selectbox("Select molecule for pharmacokinetic analysis", results["Molecule"])
            smi_sel = results.loc[results["Molecule"] == sel, "SMILES"].values[0]
            st.plotly_chart(plot_admet_radar(compute_admet(smi_sel)), use_container_width=True)

# 4. TESTIMONIALS (With Uniform Styling)
with tab_reviews:
    st.header("🌟 Global Testimonials")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.image("media/scott.jpeg", width=150) #
        st.markdown("**Scott C. Schuyler**\n\n⭐ 4.5/5")
        st.caption("Associate Professor, Chang Gung University, Taiwan")
        st.info("“Excellent tool for lead optimization. Transition from 'in silico' to 'in vitro' was seamless.”") #

    with r2:
        st.image("media/toshiya.jpg", width=150) #
        st.markdown("**Toshiya Senda**\n\n⭐ 3.5/5")
        st.caption("Research Director, KEK, Japan")
        st.info("“NeuroCureAI has changed the game for our lead discovery. Sohith, you rock!”") #

    with r3:
        st.image("media/brooks_robinson.png", width=150) #
        st.markdown("**Brooks Robinson**\n\n⭐ 4.2/5")
        st.caption("Program Director, UCCS, USA")
        st.info("“NeuroCureAI has reduced our lead-picking time, allowing focus on the actual science.”") #

    st.divider()
    st.subheader("✍️ Submit Your Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        f_name = st.text_input("Name")
        f_desig = st.text_input("Designation")
        f_rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=5)
        f_msg = st.text_area("Observations")
        if st.form_submit_button("Submit Review"):
            if f_name and f_msg:
                if send_feedback_email(f_name, f_desig, f_rating, f_msg):
                    st.success("Feedback securely transmitted. Thank you!")
                    st.balloons()
            else: st.warning("Please complete the required fields.")

# 5. INQUIRY
with tab_contact:
    st.header("📞 Academic Profile")
    c1, c2 = st.columns([1, 4])
    with c1: st.image("sohith_dp.jpg", width=200) #
    with c2:
        st.markdown("### **Sohith Reddy**")
        st.markdown("""
        Final year **B.E. Bioinformatics Undergraduate** at **Saveetha School of Engineering (SSE)**.  
        Research specialization at the intersection of **Artificial Intelligence** and **Neurodegenerative Lead Discovery**.
        
        **Current Research Tenure:** Research Assistant at **Chang Gung University (CGU)**, Taoyuan, Taiwan.
        """)
        st.markdown("---")
        st.markdown("**Portfolio:** [sohithpydev.github.io/sohith/](https://sohithpydev.github.io/sohith/)")
        st.markdown("📧 **Direct Contact:** sohith.bme@gmail.com")
