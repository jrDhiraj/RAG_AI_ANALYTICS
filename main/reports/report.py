from fpdf import FPDF
import streamlit as st
from main.AI.aiui import collect_all_insights, build_vector_db
from main.AI.chunking import chunking_docs
from main.AI.vector_db import create_vector_db
from main.reports.pre_perot import model_llm_report
from io import BytesIO
from fpdf import FPDF
import streamlit as st

st.set_page_config(layout="wide")

pdf = FPDF()

def generate_pdf(report_text):
    pdf.set_author("Dhiraj Sharma")
    pdf.set_title("AI Analytics Report")

    pdf.add_page()

    pdf.set_font(
        "Arial",
        size=11
    )

    # FPDF latin-1 support
    clean_text = (
        report_text
        .encode("latin-1", "replace")
        .decode("latin-1")
    )

    pdf.multi_cell(
        0,
        8,
        clean_text
    )

    pdf_output = pdf.output(dest="S")

    if isinstance(pdf_output, str):
        pdf_output = pdf_output.encode("latin-1")

    return pdf_output


def ui():

    st.subheader("AI Analytics Report Generator")

    if st.button("Generate AI Report"):

          with st.spinner("Collecting Insights..."):

               knowledge = collect_all_insights()

          if len(knowledge) == 0:

            st.error(
               "No insights available. Run EDA, Statistics and Charts first."
            )

            return

          with st.spinner("Generating Report..."):

               report_text = model_llm_report(
               knowledge
            )

          st.success(
               "Report Generated Successfully"
          )

          st.subheader(
               "Generated Report"
          )

          st.markdown(report_text)

          pdf_data = generate_pdf(
               report_text
          )

          pdf_data = bytes(pdf.output(dest="S"))

          st.download_button(
               label="📥 Download Report",
               data=pdf_data,
               file_name="AI_Analytics_Report.pdf",
               mime="application/pdf"
               )


ui()