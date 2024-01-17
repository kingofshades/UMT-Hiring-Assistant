import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid

#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''

def main():
    load_dotenv()

    st.set_page_config(page_title="UMT Hiring Assistant")
    
    col1, col2, col3 = st.columns([6,2,2])

    with col1:
     st.header("**UMT Hiring Assistant** ", divider='rainbow')

    with col2:
     st.write("")

    with col3:
     st.image('https://upload.wikimedia.org/wikipedia/commons/c/c8/Umt_logo.png', width = 80)

        
    st.subheader("Lets start Resume short listing...")
    st.divider()
    
    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...",key="1", height=300)
    #document_count = st.text_input("No.of 'RESUMES' to return",key="2")
    document_count = st.number_input('No.of RESUMES to Short List:', key=2, value=0   , placeholder="Type a number...")
    # Upload the Resumes (pdf files)
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)
    st.markdown(
    """ <style>
            button {
                height: auto;
                padding-top: 10px !important;
                padding-bottom: 10px !important;
                padding-left: 30px !important;
                padding-right: 30px !important;
                
            }
        </style>
    """,
    unsafe_allow_html=True,
)
    submit=st.button("**Start**",type="primary")

    if submit:
        with st.spinner('Wait for it...'):

            #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            #Create a documents list out of all the user uploaded pdf files
            final_docs_list=create_docs(pdf,st.session_state['unique_id'])

            #Displaying the count of resumes that have been uploaded
            st.write("*Resumes uploaded* :"+str(len(final_docs_list)))

            #Create embeddings instance
            embeddings=create_embeddings_load_data()

            #Push data to PINECONE
            push_to_pinecone("086c0473-8f3a-4a81-bbf4-c457e7e01629","gcp-starter","test",embeddings,final_docs_list)

            #Fecth relavant documents from PINECONE
            relavant_docs=similar_docs(job_description,document_count,"086c0473-8f3a-4a81-bbf4-c457e7e01629","gcp-starter","test",embeddings,st.session_state['unique_id'])

            #Introducing a line separator
            st.divider()

            #For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(relavant_docs)):
                
                st.subheader("⭐ Result: "+str(item+1))

                #Displaying Filepath
                st.write("**File** : "+relavant_docs[item][0].metadata['name'])

                #Introducing Expander feature
                with st.expander('Show Summary 👀'):
                    
                    
                    st.info("**Match Score** : "+str(relavant_docs[item][1]))
                    
                    #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                    summary = get_summary(relavant_docs[item][0])
                    
                    st.write("**Summary** : "+summary)
                    
                    st.markdown('##')
                    
                    score = f"{relavant_docs[item][1]*100:.2f}"
                    st.info(f"**NOTE:** The candidate possesses a qualification level of **{score}%** for the role.")
                    
        st.success("Hope I was able to save your time❤️")


#Invoking main function
if __name__ == '__main__':
    main()
