import os
from langchain.chat_models import init_chat_model
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def build_prompt(query, docs):
    
    context_parts = []

    for d in docs:

        if hasattr(d, "page_content"):
            context_parts.append(d.page_content)

        else:
            context_parts.append(str(d))

    context = "\n\n".join(context_parts)
    
    return f"""
You are a senior data analyst with 10 years of experience.

Instructions:
- Use ONLY the provided data in smart way, 
- if some things is not given in provided data calculate if possible
- Do not make assumptions,
- If data is insufficient, say "Not enough data"
- IF Question from not given data provide exact question according to the given data.

Context:
{context}

User Question:
{query}

Answer:
"""

def model_llm(query, docs):

    model = init_chat_model(
        "google_genai:gemini-2.5-flash-lite",
        temperature=0.7,   # lower = better for analysis
        max_tokens=1000,
    )

    prompt = build_prompt(query, docs)

    response = model.invoke(prompt)

    return response.content


