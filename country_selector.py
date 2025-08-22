# country_selector.py
import streamlit as st

def country_dropdown():
    return st.selectbox(
        "Country (affects Google/Bing/YouTube)",
        ["us","au","in","uk","ca","de","fr","it","es","nl"],
        index=0
    )
