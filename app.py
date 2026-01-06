import streamlit as st
import pandas as pd
import os
import base64
import pickle
import bz2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import plotly.graph_objects as go

# Chemistry Imports
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
# EMAIL BACKEND LOGIC
# =========================
def send_feedback_email(name, designation, rating, feedback):
    sender_email = "sohith.bme@gmail.com" 
    receiver_email = "sohith.bme@gmail.com" 
    
    # PASTE YOUR 16-CHARACTER APP PASSWORD HERE
    password = "nlso orfq xnaa dzbd" 

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"NeuroCureAI Feedback: {rating}/5 from {name}"

    body = f"New Review Received:\n\nName: {name}\nDesignation: {designation}\nRating: {rating}/5\nFeedback: {feedback}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# =========================
# TABS NAVIGATION
# =========================
tab_home, tab_workflow, tab_discovery, tab_reviews, tab_contact = st.tabs([
    "🏠 Home", "🔄 Workflow", "🔬 Discovery Engine", "🌟 Reviews", "📞 Contact"
])

# =========================
# 1. HOME TAB
# =========================
with tab_home:
    st.markdown("<h1 style='text-align: center;'>🧠 NeuroCureAI</h1>", unsafe_allow_html=True)
    
    # Reduced Hero Image Size
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image("media/hero_brain_ai.png", width=300) # Smaller size

    st.markdown("---")
    st.markdown("## 🔗 Bridging AI with Benchwork")
    st.image("media/portfolio.png", use_container_width=True) # Benchwork image on home page
    st.caption("We bridge the gap between AI-driven predictions and experimental laboratory validation.")

# =========================
# 4. REVIEWS TAB
# =========================
with tab_reviews:
    st.header("🌟 Existing User Reviews")
    
    rev_col1, rev_col2, rev_col3 = st.columns(3)
    with rev_col1:
        st.image("media/scott.jpeg", width=120)
        st.markdown("**Scott C. Schuyler**")
        st.info("Excellent tool for lead optimization. Transition from 'in silico' to 'in vitro' was seamless.")

    with rev_col2:
        st.image("media/toshiya.jpg", width=120)
        st.markdown("**Toshiya Senda**")
        st.info("Sohith, you rock! The platform provided results very close to our experimental values.")

    with rev_col3:
        st.image("media/brooks.png", width=120)
        st.markdown("**Brooks Robinson**")
        st.info("Reduced our lead-picking time. Focus more on the actual science.")

    st.divider()
    
    st.subheader("✍️ Submit Your Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        f_name = st.text_input("Full Name")
        f_desig = st.text_input("Designation")
        f_rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=5)
        f_msg = st.text_area("Feedback Message")
        
        if st.form_submit_button("Send Feedback"):
            if f_name and f_msg:
                if send_feedback_email(f_name, f_desig, f_rating, f_msg):
                    st.success("Success! Feedback sent to sohith.bme@gmail.com")
                    st.balloons()
            else:
                st.warning("Please fill in your name and message.")

# (Remaining tabs: Workflow, Discovery, and Contact follow the same logic as previous version)
