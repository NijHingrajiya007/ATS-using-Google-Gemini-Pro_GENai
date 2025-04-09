import streamlit as st
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
import PyPDF2 as pdf

# Streamlit setup
st.set_page_config(page_title="ATS Resume Expert", layout='centered')
st.header("ATS Resume Expert with LLAMA 2")

# Input Section
job_discription = st.text_area("Job Description:")
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
def get_llama_resume_review(job_discription, resume_text):
    try:
        llama_model_path = "C:\\Users\\nijhi\\PycharmProjects\\ATS using Google Gemini Pro\\llama-2-7b-chat.Q8_0.gguf"
        llm = CTransformers(model=llama_model_path, model_type='llama', config={'max_new_tokens': 256, 'temperature': 0.01})

        input_prompt =f"""provide the keyword similarity percentage match(out of 100%) with this provided Job description\n\n{job_discription}\n\n and resume text\n\n{resume_text}\n\n
        i want the response in one single line having the structure is....Match=keyword similarity percentage
        """
    #\n\nplease review candidate's resume against the job description:\n\n{job_discription}\n\nPlease provide your evaluation on whether the candidate's profile aligns with the job description.
        prompt = PromptTemplate(input_variables=["job_discription", "resume_text"],
                                template=input_prompt)
        response = llm(prompt.format(job_discription=job_discription, resume_text=resume_text))
        print(response)

        return response
    except Exception as e:
        return str(e)


# Resume Analysis Section
if submit_resume:
    if job_discription:
        if uploaded_file is not None:
            resume_text = input_pdf_text(uploaded_file)
            response = get_llama_resume_review(job_discription, resume_text)
            st.subheader("Review:")
            st.write(response)
            #st.write(resume_text)
        else:
            st.write("Please upload the resume")
    else:
        st.write("Please provide the job description")
