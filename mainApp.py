import App
import app2
import home

import streamlit as st

st.set_page_config(layout="wide")

PAGES = {
    "Home Page": home,
    "Team Overview Analysis": App,
    "Player Level Analysis": app2
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))


page = PAGES[selection]
page.app()