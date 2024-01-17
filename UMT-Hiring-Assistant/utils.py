
from langchain.vectorstores import Pinecone
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
import time
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain

OPENAI_API_KEY="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"
HUGGINGFACEHUB_API_TOKEN="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"

#Extract Information from PDF file                                                      
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# iterate over files in 
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


#Function to push data to Vector Store - Pinecone here
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )
    
    Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)
    


#Function to pull infrmation from Vector Store - Pinecone here
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):
    # For some of the regions allocated in pinecone which are on free tier, the data takes upto 10secs for it to available for filtering
    #so I have introduced 20secs here, if its working for you without this delay, you can remove it :)
    #https://docs.pinecone.io/docs/starter-environment
    print("20secs delay...")
    time.sleep(20)
    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index



#Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,k,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,unique_id):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    return similar_docs



#summarization using BART

""" def get_summary(current_doc):
    
    # Load the best available text summarization pipeline from Hugging Face
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Load a high-performance LLM
    text_content = current_doc.page_content
    # Extract the text content from the Document object
    # Generate the summary
    summary = summarizer(text_content, max_length=150, min_length=80, do_sample=False)[0]["summary_text"] # type: ignore

    return summary """
    
#summarization using OpenAssistant/osasst
def get_summary(current_doc):

    # Define prompt
    prompt_template = """Assist me with cleaning, extracting information and summarizing a resume given below in 50 to 150 words:
    "{text}"
    SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Define LLM chain
    llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b", model_kwargs={"temperature":0.1})
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    summary = stuff_chain.run([current_doc])

    return summary
