import streamlit as st
import replicate
import PyPDF2 as pdf
from dotenv import load_dotenv
load_dotenv()

# Streamlit setup
st.set_page_config(page_title="ATS Resume Expert", layout='centered')
st.header("ATS Resume Expert with LLAMA 2")

# Input Section
job_description = st.text_area("Job Description:")
uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
submit_resume = st.button("Analyze Resume against Job Description")

# Function to extract text from uploaded PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to get response from LLAMA 2 model for analyzing resumes
def get_api_response(job_description, resume_text):
    input_data = {
        "top_p": 1,
        "prompt": f"provide the keyword similarity percentage match(out of 100%) with this provided Job description\n\n{job_description}\n\n and resume text\n\n{resume_text}\n\n i want the response in one single line having the structure is....Match=keyword similarity percentage",
        "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        "max_new_tokens": 500
    }

    output = replicate.run("meta/llama-2-13b-chat", input=input_data)
    return "".join(output)

# Resume Analysis Section
if submit_resume:
    if job_description:
        if uploaded_file is not None:
            resume_text = input_pdf_text(uploaded_file)
            response = get_api_response(job_description, resume_text)
            st.subheader("Review:")
            st.write(response)
        else:
            st.write("Please upload the resume")
    else:
        st.write("Please provide the job description")
