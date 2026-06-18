import streamlit as st
from langchain.chat_models import init_chat_model
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def build_prompt_reports(docs):

    context_parts = []

    for d in docs:
        if hasattr(d, "page_content"):
            context_parts.append(d.page_content)
        else:
            context_parts.append(str(d))


    context = "\n\n".join(context_parts)

    return f"""
You are a Senior Data Analyst.

Create a complete professional report.

Sections:

1. Executive Summary
2. Dataset Overview
3. Missing Value Analysis
4. Outlier Analysis
5. Statistical Findings
6. Correlation Findings
7. Feature Importance
8. Key Insights
9. Recommendations
10. Conclusion

Rules:

- Use ONLY provided data.
- Never hallucinate.
- Explain findings in business language.
- Use markdown tables whenever possible.
- Mention risks and limitations.
- Provide actionable recommendations.

Dataset Information:

{context}

Generate the report:
"""

def model_llm_report(docs):

    model = init_chat_model(
        "google_genai:gemini-2.5-flash-lite",
        temperature=0.7,   # lower = better for analysis
        max_tokens=4000,
    )

    prompt = build_prompt_reports(docs)

    response = model.invoke(prompt)

    return response.content