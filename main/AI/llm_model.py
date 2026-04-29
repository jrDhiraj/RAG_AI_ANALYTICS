import os
from langchain.chat_models import init_chat_model
import streamlit as st

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def build_prompt(query, docs):
    
    context = "\n\n".join([d.page_content for d in docs])

    return f"""
You are a senior data analyst with 10 years of experience.

Instructions:
- Use ONLY the provided data
- Do not make assumptions
- If data is insufficient, say "Not enough data"

Context:
{context}

User Question:
{query}

Tasks:
1. Find relationships between features
2. Give data-driven insights
3. Suggest ways to increase revenue

Answer:
"""


def model_llm(query, docs):

    model = init_chat_model(
        "google_genai:gemini-2.5-flash-lite",
        temperature=0.6,   # lower = better for analysis
        max_tokens=1000,
    )

    prompt = build_prompt(query, docs)

    response = model.invoke(prompt)

    return response.content


