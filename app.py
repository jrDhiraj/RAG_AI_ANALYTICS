import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title("AI ANALYTICS")

pages = [

    st.Page(r"main\\utils\\home\\home_page.py", title="Description", icon="🏠"),
    st.Page(r"main\\utils\\stats\\stats_page.py", title="Statistics", icon="📏"),
    st.Page(r"main\\utils\\charts\\chartsutils.py", title = "charts ", icon = "📊"),
    st.Page(r"main\\AI\\aiui.py", title="AI chart", icon = "🤖")

]
pg = st.navigation(pages, position="top")

pg.run()





