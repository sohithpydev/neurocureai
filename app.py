import streamlit as st
import pandas as pd
import os
import pickle
import bz2
import base64
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

from padelpy import padeldescriptor
from rdkit import Chem
from rdkit.Chem import Descriptors

# ==========================================
# PAGE CONFIGURATION & PREMIUM DESIGN
# ==========================================
st.set_page_config(
    page_title="NeuroCureAI | AI Drug Discovery",
    page_icon="🧠",
    layout="wide"
)

# Advanced CSS for High-End Design (To impress Manidhar Singh)
st.markdown("""
    <style>
    /* Global Background and Typography */
    .main {
        background-color: #fcfcfc;
        font-family: 'Inter', sans-serif;
    }
    
    /* ENLARGING AND SPACING TABS - DESIGN SCHOOL STANDARDS */
    button[data-baseweb="tab"] {
        font-size: 22px !important; 
        font-weight: 600 !important;
        color: #555 !important;
        padding-left: 50px !important; 
        padding-right: 50px !important;
        margin-right: 20px !important; 
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border-bottom: 4px solid transparent !important;
        height: 60px !important;
    }

    /* Tab Hover & Active States */
    button[data-baseweb="tab"]:hover {
        color: #2a5298 !important;
        background-color: #f8f9fa !important;
        transform: translateY(-2px);
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1e3c72 !important;
        border-bottom: 4px solid #1e3c72 !important;
        background-color: transparent !important;
    }

    /* Glassmorphism Containers with extra White Space */
    div[data-testid="stExpander"], .stContainer, div[data-testid="stForm"], .stImage, .stVideo {
        border: none !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(15px);
        border-radius: 30px !important;
        box-shadow: 0 15px 50px 0 rgba(0, 0, 0, 0.05) !important;
        padding: 40px !important; 
        margin-bottom: 40px !important;
    }

    /* Professional Gradient Title */
    .title-text {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 4.2rem;
        letter-spacing: -2px;
        margin-bottom: 10px;
    }

    /* Custom Read More Styling */
    .description-box {
        font-size: 1.1rem;
        color: #444;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# ASSET LOADERS (Lottie & Email)
# ==========================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_running = load_lottieurl("https://lottie.host/808605c1-e705-407b-a010-062829b3c582/A0O9MclLAn.json")

def send_feedback_email(name, designation, rating, feedback):
    sender_email = "sohith.bme@gmail.com" 
    receiver_email = "sohith.bme@gmail.com"
    password = "nlso orfq xnaa dzbd" 
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"NeuroCureAI Feedback: {rating}/5 Stars from {name}"
    
    body = f"New Review Received:\n\nName: {name}\nDesignation: {designation}\nRating: {rating}/5\n\nFeedback:\n{feedback}"
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email failure. Error: {str(e)}")
        return False

# ==========================================
# CORE BACKEND LOGIC
# ==========================================
def desc_calc():
    fp = {
        'AtomPairs2D': 'AtomPairs2DFingerprinter.xml', 'CDK': 'Fingerprinter.xml',
        'CDKextended': 'ExtendedFingerprinter.xml', 'CDKgraphonly': 'GraphOnlyFingerprinter.xml',
        'EState': 'EStateFingerprinter.xml', 'KlekotaRoth': 'KlekotaRothFingerprinter.xml',
        'MACCS': 'MACCSFingerprinter.xml', 'PubChem': 'PubchemFingerprinter.xml',
        'Substructure': 'SubstructureFingerprinter.xml'
    }
    selection = os.path.abspath('molecule.smi')
    common_params = dict(mol_dir=selection, detectaromaticity=True, standardizenitro=True,
                        standardizetautomers=True, threads=2, removesalt=True, log=False, fingerprints=True)

    for name, xml in fp.items():
        padeldescriptor(d_file=f"{name}.csv", descriptortypes=f"./PaDEL-Descriptor/{xml}", **common_params)
    
    def load_fp_clean(path):
        df = pd.read_csv(path)
        return df.drop_duplicates("Name").set_index("Name")
    
    fps = [f"{name}.csv" for name in fp.keys()]
    X = pd.concat([load_fp_clean(f) for f in fps], axis=1)
    X.reset_index().to_csv("descriptors_output.csv", index=False)
    
    for f in fps: 
        if os.path.exists(f): os.remove(f)
    if os.path.exists('molecule.smi'): os.remove('molecule.smi')

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

# ==========================================
# MAIN NAVIGATION TABS
# ==========================================
tab_home, tab_workflow, tab_discovery, tab_reviews, tab_contact = st.tabs([
    "🏠 Dashboard", "🔄 Pipeline", "🔬 Discovery Engine", "🌟 Testimonials", "📞 Inquiry"
])

# 1. DASHBOARD
with tab_home:
    st.markdown('<h1 class="title-text" style="text-align:center;">NeuroCureAI</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.6rem; color:#666;'>Computational lead discovery for Alzheimer's Research.</p>", unsafe_allow_html=True)
    
    # HERO SECTION WITH GIF SUPPORT AND DESCRIPTION
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        col2_1, col2_2 = st.columns([1.2, 1])
        with col2_1:
            if os.path.exists("media/hero_brain_ai.gif"):
                st.image("media/hero_brain_ai.gif", use_column_width=True) # GIF Autoplay
            else:
                st.image("media/hero_brain_ai.png", use_column_width=True) # Fallback to static
        
        with col2_2:
            st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("""
                <div class='description-box'>
                <b>NeuroCureAI</b> leverages advanced Machine Learning and Molecular Fingerprinting to accelerate the identification of potent therapeutic inhibitors. 
                Our platform streamlines the journey from raw chemical data to high-confidence lead compounds.
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Read more.."):
                st.write("""
                    By integrating PaDEL-Descriptor technology with robust Random Forest regression models, 
                    NeuroCureAI predicts the pIC50 values of novel molecules with high precision. 
                    The tool further evaluates pharmacokinetic feasibility through ADMET profiling, ensuring 
                    that discovered leads are not just active, but drug-ready.
                """)
    
    st.markdown("---")
    st.markdown("## 🔗 Bridging AI with Benchwork")
    
    # Bottom section prefers Video/Gif
    if os.path.exists("media/portfolio.gif"):
        st.image("media/portfolio.gif", use_container_width=True)
    elif os.path.exists("media/portfolio.mp4"):
        st.video("media/portfolio.mp4", format="video/mp4", start_time=0)
    else:
        st.image("media/portfolio.png", use_container_width=True)
        
    st.caption("Integrating computational predictions with experimental validation")

# 2. PIPELINE
with tab_workflow:
    st.header("🔁 Research Methodology")
    st.image("media/workflow.jpg", use_container_width=True)

# 3. DISCOVERY ENGINE
with tab_discovery:
    st.header("🔬 AI-Driven Molecular Discovery")
    with st.container():
        uploaded = st.file_uploader("Upload molecule file (.txt)", type=["txt"])
        run_btn = st.button("🚀 Execute Discovery Algorithm")

    if run_btn and uploaded is not None:
        st.session_state["run"] = True
        st.session_state["input_df"] = pd.read_table(uploaded, sep=" ", header=None)

    if st.session_state.get("run", False):
        anim_placeholder = st.empty()
        with anim_placeholder.container():
            st.markdown("### 🧬 Algorithm Processing...")
            if lottie_running:
                st_lottie(lottie_running, height=300, key="runner")
            
            input_df = st.session_state["input_df"]
            input_df.to_csv("molecule.smi", sep="\t", index=False, header=False)
            time.sleep(1) 
            
            st.info("Calculating comprehensive molecular fingerprints...might take upto 1 minute, please wait!")
            desc_calc()
            st.success("✅ Computational descriptors generated successfully.")

        anim_placeholder.empty() 
        
        res_tab1, res_tab2, res_tab3 = st.tabs(["🧬 Data Transparency", "🏆 Predicted Activity", "🩸 ADMET Profile"])
        
        if os.path.exists("descriptors_output.csv"):
            desc = pd.read_csv("descriptors_output.csv")
            Xlist = list(pd.read_csv("descriptor_list.csv").columns)
            
            with res_tab1:
                st.subheader("1. Comprehensive Descriptor Matrix")
                st.dataframe(desc.head(10), use_container_width=True)
                st.subheader("2. Model-Specific Subset (Xlist)")
                st.dataframe(desc[Xlist].head(10), use_container_width=True)

            with res_tab2:
                model = load_model()
                preds = model.predict(desc[Xlist])
                results = pd.DataFrame({
                    "Molecule": st.session_state["input_df"][1], 
                    "SMILES": st.session_state["input_df"][0], 
                    "Predicted pIC50": preds
                }).sort_values("Predicted pIC50", ascending=False)
                st.subheader("Lead Compound Ranking")
                st.dataframe(results, use_container_width=True)

            with res_tab3:
                st.subheader("Pharmacokinetic Screening")
                sel = st.selectbox("Select molecule", results["Molecule"])
                smi_sel = results.loc[results["Molecule"] == sel, "SMILES"].values[0]
                st.plotly_chart(plot_admet_radar(compute_admet(smi_sel)), use_container_width=True)

# 4. TESTIMONIALS
with tab_reviews:
    st.header("🌟 Global Testimonials")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.image("media/scott.jpeg", width=150)
        st.markdown("**Scott C. Schuyler**\n\n⭐ 4.5/5")
        st.caption("Associate Professor, Chang Gung University, Taiwan")
        st.info("“Excellent tool for lead optimization. Transition from 'in silico' to 'in vitro' was seamless.”")

    with r2:
        st.image("media/toshiya.jpg", width=150)
        st.markdown("**Toshiya Senda**\n\n⭐ 3.5/5")
        st.caption("Research Director, KEK, Japan")
        st.info("“NeuroCureAI has changed the game for our lead discovery. Sohith, you rock!”")

    with r3:
        st.image("media/brooks_robinson.png", width=150)
        st.markdown("**Brooks Robinson**\n\n⭐ 4.2/5")
        st.caption("Program Director, UCCS, USA")
        st.info("“NeuroCureAI has reduced our lead-picking time, allowing focus on the actual science.”")

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
    st.header("📞 Developer Profile")
    c1, c2 = st.columns([1, 4])
    with c1: st.image("sohith_dp.jpg", width=200)
    with c2:
        st.markdown("### **Sohith Reddy**")
        st.markdown("""
        Final year **B.E. Bioinformatics Undergraduate** at **Saveetha School of Engineering (SSE)**.  
        Research specialization at the intersection of **Artificial Intelligence** and **Neurodegenerative Lead Discovery**.
        
        **Current Designation:** Research Assistant at **Chang Gung University (CGU)**, Taoyuan, Taiwan.
        """)
        st.markdown("---")
        st.markdown("**Portfolio:** [sohithpydev.github.io/sohith/](https://sohithpydev.github.io/sohith/)")
        st.markdown("📧 **Direct Contact:** sohith.bme@gmail.com")
