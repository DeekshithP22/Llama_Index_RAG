#Importing necessary frameworks and libraries
import os
import sys
import os.path
import logging
import pandas as pd
import streamlit as st

from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.agents import create_pandas_dataframe_agent 

from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Prompt
from llama_index.core import GPTListIndex
from llama_index.core import ComposableGraph
from llama_index.readers.file import CSVReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Check if storage already exists
PERSIST_DIR = "./storage"

# Define a dictionary of users and their credentials
user_credentials = {
    "user1@example.com": {"password": "password1", "persona": ["ANY_EMPLOYEE"]},
    "user2@example.com": {"password": "password2", "persona": ["SERVICE_LINE_HEAD"]},
    "user3@example.com": {"password": "password3", "persona": ["CXO"]},
    "user4@example.com": {"password": "password4", "persona": ["CORPORATE_STRATEGY"]},
    "admin@example.com": {"password": "adminpassword", "persona": ["ADMIN"]}
}

# Authenticate user based on email, password, and persona
def authenticate(email, password, persona):
    try:
        if email in user_credentials:
            user_data = user_credentials[email]
            if user_data["password"] == password and persona in user_data["persona"]:
                return True
        return False
    
    except Exception as e:
        st.error(f"Error occured during authentication {str(e)}")
    
# Setup logging
def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize LLM 
def setup_llm(api_key, azure_endpoint, api_version):
    try:
        # prompt = """You are Q and A assistant named as corporate performance geek An AI Assistant that can 
        #                   answer every question related to a corporate’s financial performance while adhering to the 
        #                   boundaries of corporate information classification and access levels of various enterprise personas,
        #                   you should handle different information classification levels and data access rights appropriately,
        #                   should redact information based on the persona or navigate them to appropriate human moderator for any 
        #                   profanity or unethical questions.

        #                   Your goal is to answer question related to the coroprates perfromance as accurately as possible based on the 
        #                   instructions and the context, 
        #                   respond with "Answer is not available in the context or you do not have access to this data as this may be confidential, 
        #                   you should have respective authorization or credentials to aceess the confidential data."""
       
        llm = AzureOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            deployment_name=os.getenv("OPENAI_DEPLOYMENT"),
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            # system_prompt= prompt,
            temperature=0.5,
        )
        return llm
    
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")   
        return None
    
def struct_llm(api_key, azure_endpoint, api_version):
    try:
        llm = AzureChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            azure_deployment=os.getenv("OPENAI_STRUCT_DEPLOYMENT"),
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return llm
    
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")   
        return None
      
# Initialize embedding model
def setup_embedding(api_key, azure_endpoint, api_version):
    try:
        embed_model = AzureOpenAIEmbedding(
            model=os.getenv("OPENAI_EMBEDDING_MODEL"),
            deployment_name= os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return embed_model
    
    except Exception as e:
        st.error(f"Error setting up embeddings: {str(e)}")  
        return None

# Setup new index based on classification and data access rights 
def setup_new_index(documents, persona):
    try:
        Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        print("creating embeddings again")
        index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)])

        # Create directories for public and confidential if they don't exist
        if not os.path.exists(os.path.join(PERSIST_DIR, "Public")):
            os.makedirs(os.path.join(PERSIST_DIR, "Public"))
        if not os.path.exists(os.path.join(PERSIST_DIR, "Confidential")):
            os.makedirs(os.path.join(PERSIST_DIR, "Confidential"))

        # Save the VectorStoreIndex for public data
        if persona == "CORPORATE_STRATEGY" or persona == "CXO":
            index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "Confidential"))

        # Save the VectorStoreIndex for confidential data
        elif persona == "ANY_EMPLOYEE" or persona == "SERVICE_LINE_HEAD":
            # Save the index embeddings for both public and confidential data
            index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "Public"))

        return index      
    
    except Exception as e:
        st.error(f"Error setting up embeddings: {str(e)}")  
        return None


# Setup new index based on classification and data access rights 
def new_index(directory_path, storage_type):
    try:
        st.info("Loading documents for creating the VectorStoreIndex...")
        documents = SimpleDirectoryReader(input_dir=directory_path).load_data(num_workers=4)
        
        # Update index based on storage type
        if storage_type == "Public":
            persona = ["ANY_EMPLOYEE", "SERVICE_LINE_HEAD"]
        elif storage_type == "Confidential":
            persona = ["CXO", "CORPORATE_STRATEGY"] # Or whichever persona is applicable for confidential storage
        else:
            st.error("Invalid storage type selected.")
            return None
        
        index = setup_new_index(documents, persona)
        storage_dir = os.path.join(PERSIST_DIR, storage_type.lower())
        
        st.info("Done updating the VectorStoreIndex.")
        st.info("Saving the VectorStoreIndex to the persistent storage...")
        index.storage_context.persist(persist_dir=storage_dir)
        st.success("Documents loaded and VectorStoreIndex updated successfully.")
        return index
     
    except Exception as e:
        st.error(f"Error updating index: {str(e)}")
        return None

# Update exisitng index based on classification and data access rights 
def update_index(directory_path, storage_type):
    try:
        Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        # files = os.listdir(directory_path)
        # loader = CSVReader()
        # dict=[]
        # documents = loader.load_data(file=directory_path,extra_info=dict)
        # print("Done reading CSV document")
        documents = SimpleDirectoryReader(input_dir=directory_path).load_data(num_workers=4)
        
        if storage_type == "Public":
            p_storage_dir = os.path.join(PERSIST_DIR, "Public")
            p_storage_context = StorageContext.from_defaults(persist_dir=p_storage_dir)
            p_index = load_index_from_storage(p_storage_context)
            c_storage_dir = os.path.join(PERSIST_DIR, "Confidential")
            c_storage_context = StorageContext.from_defaults(persist_dir=c_storage_dir)
            c_index = load_index_from_storage(c_storage_context)            
            parser = SimpleNodeParser()
            new_nodes = parser.get_nodes_from_documents(documents)
            p_index.insert_nodes(new_nodes)
            c_index.insert_nodes(new_nodes)
            p_index.storage_context.persist(persist_dir=p_storage_dir)
            c_index.storage_context.persist(persist_dir=c_storage_dir)
            st.success("Documents loaded and VectorStoreIndex updated successfully.")
            return p_index
        
        elif storage_type == "Confidential":
            storage_dir = os.path.join(PERSIST_DIR, "Confidential")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)        
            parser = SimpleNodeParser()
            new_nodes = parser.get_nodes_from_documents(documents)
            index.insert_nodes(new_nodes)
            storage_dir = os.path.join(PERSIST_DIR, storage_type.lower())
            index.storage_context.persist(persist_dir=storage_dir)
            st.success("Documents loaded and VectorStoreIndex updated successfully.")
            return index
        
        else:
            st.error("Invalid storage type selected.")
            return None

        
    except Exception as e:
        st.error(f"Error updating existing index: {str(e)}")  
        return None
    

# Initialize llm credentials and parameters
def initialize(api_key, azure_endpoint, api_version):
    try:
        setup_logging()
        logging.info("Initializing AzureOpenAI...")
        llm = setup_llm(api_key, azure_endpoint, api_version)
        logging.info("Initializing Azure OpenAI Embeddings...")
        embed_model = setup_embedding(api_key, azure_endpoint, api_version)
        Settings.llm = llm
        Settings.embed_model = embed_model
        logging.info("Done initializing both the OpenAI LLM and Embeddings...")

    except Exception as e:
        st.error(f"Error initializing LLM and embeddings model: {str(e)}")
        return None

# Load VectorStoreIndex from storage based on classification and data access rights
def load_index(persona):
    try:
        if persona in ["ANY_EMPLOYEE","SERVICE_LINE_HEAD"]:
            storage_dir = os.path.join(PERSIST_DIR, "public")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            print("Loaded the inde from the public storage")
            return index
            
        elif persona in ["CXO", "CORPORATE_STRATEGY"]:
            c_storage_dir = os.path.join(PERSIST_DIR, "Confidential")
            c_storage_context = StorageContext.from_defaults(persist_dir=c_storage_dir)
            c_index = load_index_from_storage(c_storage_context)
            return c_index

        else:
            st.error("Invalid persona...")
            return None
        
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None

def create_agent(llm):
    df = pd.read_csv(r'C:\Users\deekshith.p\Data_AI_Hackathon\storage\Structured_Data\Dummy_Financial_Summary_Q4FY22.csv')
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type="openai-tools")
    return agent


def run_agent(agent, query):
    try:
        response = agent.run(query)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    return response

def struct_query():
    # Create a directory for storing Excel files if not exists
    if not os.path.exists(os.path.join(PERSIST_DIR, "Structured_Data")):
        os.makedirs(os.path.join(PERSIST_DIR, "Structured_Data"))

    api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION")
    
    llm = struct_llm(api_key, azure_endpoint, api_version)

    # Query input box
    query = st.text_input("Enter question to query structured data")

    # "Query Excel" button
    if query:
        agent = create_agent(llm)
        response = run_agent(agent, query)
        st.write("Response:", response)


# UI for admin login
def admin_page():
    try:
        st.title("🤖  Corporate Performance Geek")
        st.sidebar.title("Admin Login")
        email = st.sidebar.text_input("Email:", key = "admin_email")
        password = st.sidebar.text_input("Password:", type="password", key = "admin_password")
        login_button = st.sidebar.button("Login")

        if login_button:
            try:
                if authenticate(email, password, "ADMIN"):
                    st.session_state.admin_authenticated = True
                    st.sidebar.success("Authentication successful!")

                else:
                    st.error("Incorrect email or password. Please try again.")
            except Exception as e:
              st.error(f"Error occured during authentication: {str(e)}")

        # Check if authenticated
        if st.session_state.get("admin_authenticated"):
            display_admin_content()

    except Exception as e:
        st.error(f"Error loading admin page: {str(e)}")
        return None           

# UI for admin page
def display_admin_content():
    try:
        st.write("Upload files and update indexes here.")

        directory_path = st.text_area("Enter the directory path of the files:")
        storage_type = st.selectbox("Select storage type to update:", ["Public", "Confidential"])

        if st.button("Create New Index"):
            with st.spinner("Creating new index...."):
                st.session_state.index = new_index(directory_path, storage_type)
            if st.session_state.index:
                st.success(f"Index updated successfully for {storage_type.lower()} storage.")
            else:
                st.error("Failed to create index.")


        elif st.button("Update exisitng index"):
            with st.spinner("Updating exisitng index...."):
                st.session_state.index = update_index(directory_path, storage_type)
            if st.session_state.index:
                st.success(f"Index updated successfully for {storage_type.lower()} storage.")
            else:
                st.error("Failed to update existing index.")
               
    except Exception as e:
        st.error(f"Error displaying admin page content: {str(e)}")
        return None         
      
# UI for user login
def home_page():
    try:
        st.title("🤖  Corporate Performance Geek")

        # Sidebar for authentication
        persona = st.sidebar.selectbox("Select Persona:", ["ANY_EMPLOYEE", "SERVICE_LINE_HEAD", "CXO", "CORPORATE_STRATEGY"])
        st.sidebar.title("Authentication")
        email = st.sidebar.text_input("Email:")
        password = st.sidebar.text_input("Password:", type="password")
        

        # Check if persona selection changed
        if "selected_persona" not in st.session_state or st.session_state.selected_persona != persona:
            st.session_state.authenticated = False
            st.session_state.selected_persona = persona

        # Check if login button is clicked
        if st.sidebar.button("Login"):
            try:
                if authenticate(email, password, persona):
                    st.sidebar.success("Authentication successful!")
                    st.session_state.authenticated = True
                else:
                    st.sidebar.error("Authentication failed. Please check your credentials and persona selection.")
            except Exception as e:
                st.error(f"authentication failed: {str(e)}")
            
        # Display main page content if authenticated
        if st.session_state.get("authenticated"):
            display_home_main_page(persona)

    except Exception as e:
        st.error(f"Error loading home page: {str(e)}")
        return None      

# UI for user page
def display_home_main_page(persona):

    index = load_index(persona)
    # Main page content
    question = st.text_area("Ask a question:", height=100, max_chars=500)


    if persona == "CORPORATE_STRATEGY" or persona == "CXO":   
        struct_query()

    if question:
        try:
            query_engine = index.as_query_engine()
            answer = query_engine.query(question)
            answer.get_formatted_sources()
            with st.container():
                st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                pprint_response(answer, show_source=True)
                st.write("Reply: \n", answer.response)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

    # UI setup
api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
    
initialize(api_key, azure_endpoint, api_version)

# Main function
def main():

    primary_color = "#3366ff"
    secondary_color = "#66ff66"

    # CSS styling
    st.markdown(
        f"""
        <style>
            /* Custom CSS styles */
            body {{
                background-color: #f5f5f5;
                color: #333;
                font-family: Arial, sans-serif;
            }}
            .btn-primary {{
                background-color: {primary_color};
                color: #fff;
                padding: 0.5rem 1rem;
                border-radius: 0.25rem;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}
            .btn-primary:hover {{
                background-color: #254EDA;
            }}
            .text-input {{
                background-color: #fff;
                padding: 0.5rem;
                border: 1px solid #ccc;
                border-radius: 0.25rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


    with st.sidebar:
        st.title("Menu")
        st.sidebar.markdown("---")

    # Page selection
    page = st.sidebar.radio("Go to", ["Home Page", "Admin Page"])

    if page == "Home Page":
        home_page()
    elif page == "Admin Page":
        admin_page()

if __name__ == "__main__":
    main()
