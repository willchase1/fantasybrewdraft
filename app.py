"""Fantasy Brewing — single entry point, two modes.

Run with:  streamlit run app.py

The app serves two deliberate use cases that share one engine (draft_core):
  * Run a Draft  — the live, single-commissioner draft board (draft_mode.py)
  * My Brewery   — personal recipe/prep sandbox, decoupled from any live
                   draft (brewery_mode.py)

st.navigation renders the mode switch; each mode is a normal Streamlit script.
set_page_config must be called here (once) before any other Streamlit command.
"""
import streamlit as st

st.set_page_config(page_title="Fantasy Brewing", layout="wide")

draft_page = st.Page("draft_mode.py", title="Run a Draft", icon="🍺", default=True)
brewery_page = st.Page("brewery_mode.py", title="My Brewery", icon="🧪")

pg = st.navigation([draft_page, brewery_page])
pg.run()
