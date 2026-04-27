import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title("AI ANALYTICS")

import streamlit as st

pages = [
    st.Page(r"main\\utils\\home\\home_page.py", title="Description", icon="🏠"),
    st.Page(r"main\\utils\\stats\\stats_page.py", title="Statistics", icon="📊"),

]
pg = st.navigation(pages, position="top")

pg.run()





