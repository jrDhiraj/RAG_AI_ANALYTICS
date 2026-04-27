import pandas as pd

def load_df():
    
    # df = st.file_uploader("upload dataset in CSV formate", type="csv")
    df = pd.read_csv("synthetic_premium_dataset.csv")

    if df is not None:
        return df


