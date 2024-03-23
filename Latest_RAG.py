import logging
import sys
import os.path
import streamlit as st
import plotly.express as px

from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Define color scheme
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
        st.error(f"Error occurred during authentication {str(e)}")
    
# Setup logging
def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize OpenAI models and embeddings
def setup_llm(api_key, azure_endpoint, api_version):
    try:
        llm = AzureOpenAI(
            model="gpt-35-turbo-16k",
            deployment_name="gpt-35-turbo",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return llm
    
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")   
        return None

def setup_embedding(api_key, azure_endpoint, api_version):
    try:
        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="text-embedding-ada-002",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return embed_model
    
    except Exception as e:
        st.error(f"Error setting up embeddings: {str(e)}")  
        return None

# Setup index with separate storage for public and confidential data
def setup_index(documents, persona):
    try:
        Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)])

        # Create directories for public and confidential if they don't exist
        if not os.path.exists(os.path.join(PERSIST_DIR, "Public")):
            os.makedirs(os.path.join(PERSIST_DIR, "Public"))
        if not os.path.exists(os.path.join(PERSIST_DIR, "Confidential")):
            os.makedirs(os.path.join(PERSIST_DIR, "Confidential"))

        if persona == "CORPORATE_STRATEGY" or persona == "CXO":
            # Save the index embeddings for public data
            index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "Confidential"))
        elif persona == "ANY_EMPLOYEE" or persona == "SERVICE_LINE_HEAD":
            # Save the index embeddings for both public and confidential data
            index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "Public"))

        return index
    
    except Exception as e:
        st.error(f"Error setting up embeddings: {str(e)}")  
        return None

# Update index with new documents
def update_index(directory_path, storage_type):
    try:
        st.info("Loading documents for updating the VectorStoreIndex...")
        documents = SimpleDirectoryReader(input_dir=directory_path).load_data(num_workers=4)
        
        # Update index based on storage type
        if storage_type == "Public":
            persona = ["CORPORATE_STRATEGY", "SERVICE_LINE_HEAD"]
        elif storage_type == "Confidential":
            persona = ["CXO", "CORPORATE_STRATEGY"] # Or whichever persona is applicable for confidential storage
        else:
            st.error("Invalid storage type selected.")
            return None
        
        index = setup_index(documents, persona)
        storage_dir = os.path.join(PERSIST_DIR, storage_type.lower())
        
        st.info("Done updating the VectorStoreIndex.")
        st.info("Saving the VectorStoreIndex to the persistent storage...")
        index.storage_context.persist(persist_dir=storage_dir)
        st.success("Documents loaded and VectorStoreIndex updated successfully.")
        return index
    
    except Exception as e:
        st.error(f"Error updating index: {str(e)}")
        return None

# Initialize
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

# Load index from storage based on the selected persona
def load_index(persona):
    try:
        if persona in ["ANY_EMPLOYEE","SERVICE_LINE_HEAD"]:
            storage_dir = os.path.join(PERSIST_DIR, "public")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            return index
            
        elif
        elif persona in ["CXO", "CORPORATE_STRATEGY"]:
            storage_dir = os.path.join(PERSIST_DIR, "confidential")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            return index

        else:
            st.error("Invalid persona...")
            return None
        
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None


def admin_page():
    try:
        st.title("Admin Page")
        st.sidebar.title("Admin Login")
        email = st.sidebar.text_input("Email:")
        password = st.sidebar.text_input("Password:", type="password")
        login_button = st.sidebar.button("Login")

        if login_button:
            try:
                if authenticate(email, password, "ADMIN"):
                    st.session_state.authenticated = True
                    st.sidebar.success("Authentication successful!")
                else:
                    st.error("Incorrect email or password. Please try again.")
            except Exception as e:
                st.error(f"Error occurred during authentication: {str(e)}")

        # Check if authenticated
        if st.session_state.get("authenticated"):
            st.title("Admin Page")
            st.write("Upload files and update indexes here.")

            directory_path = st.text_area("Enter the directory path of the files:")
            storage_type = st.selectbox("Select storage type to update:", ["Public", "Confidential"])

            if st.button("Update Index"):
                with st.spinner("Updating index...."):
                    # Process uploaded files and update indexes
                    st.session_state.index = update_index(directory_path, storage_type)
                if st.session_state.index:
                    st.success(f"Index updated successfully for {storage_type.lower()} storage.")
                else:
                    st.error("Failed to update index.")

    except Exception as e:
        st.error(f"Error loading admin page: {str(e)}")
        return None           


def home_page():
    try:
        st.title("🤖 Corporate Performance Geek")
        st.sidebar.title("Authentication")
        persona = st.sidebar.selectbox("Select Persona:", ["ANY_EMPLOYEE", "SERVICE_LINE_HEAD", "CXO", "CORPORATE_STRATEGY"])
        email = st.sidebar.text_input("Email:")
        password = st.sidebar.text_input("Password:", type="password")
        login_button = st.sidebar.button("Login")

        if login_button:
            try:
                if authenticate(email, password, persona):
                    st.sidebar.success("Authentication successful!")
                    st.session_state.authenticated = True
                else:
                    st.sidebar.error("Authentication failed. Please check your credentials and persona selection.")
            except Exception as e:
                st.error(f"Error occurred during authentication: {str(e)}")

        if st.session_state.get("authenticated"):
            st.markdown("---")
            question = st.text_area("Ask a question:", height=100, max_chars=500)
            if st.button("Submit"):
                index = load_index(persona)
                if index:
                    query_engine = index.as_query_engine()
                    answer = query_engine.query(question)
                    answer.get_formatted_sources()
                    with st.container():
                        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                        pprint_response(answer, show_source=True)
                        st.write("Answer: \n", answer.response)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("Failed to load index.")

    except Exception as e:
        st.error(f"Error loading home page: {str(e)}")
        return None


def main():
    api_key = "a6efd3fae742488ebbcaefeaaddb0aff"
    azure_endpoint = "https://openai-ppcazure017.openai.azure.com/"
    api_version = "2023-03-15-preview"
    initialize(api_key, azure_endpoint, api_version)

    with st.sidebar:
        st.title("Menu")
        st.sidebar.write("---")

    page = st.sidebar.radio("Go to", ["Home Page", "Admin Page"])

    if page == "Home Page":
        home_page()
    elif page == "Admin Page":
        admin_page()


if __name__ == "__main__":
    main()
