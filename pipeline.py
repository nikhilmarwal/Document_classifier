#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from typing import Dict, Any, Tuple
from pypdf import PdfReader
from dotenv import load_dotenv

# --- AGENT 1: ClassifierScraperAgent ---

def extract_text_tool(file_path: str) -> str:
    """Tool: Extracts all text content from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n\n---PAGE {page_num + 1} START---\n\n"
            text += page.extract_text()
        return text
    except Exception as e:
        return f"ERROR: Could not read PDF. Details: {e}"

class ClassifierScraperAgent:
    """Handles file access, scraping, and classification."""
    
    def __init__(self, llm_model: str = "gemini-2.5-flash", temperature: float = 0.0, api_key: str=None):
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, google_api_key=api_key)
        self.classification_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert document classifier. Classify the provided text snippet into one of: 'Medical Record', 'Research Paper', 'Legal Document', or 'Other/Unclassified'. Return ONLY the classified type name."),
                ("user", "Text Snippet to Classify:\n\n---\n{raw_text_snippet}\n---"),
            ]
        )
    
    def classify_document(self, raw_text_snippet: str) -> str:
        chain = self.classification_prompt | self.llm 
        response = chain.invoke({"raw_text_snippet": raw_text_snippet}).content
        return response.strip()

    # The run method is designed to be called by the LCEL pipeline
    def run(self, file_path: str) -> Tuple[str, str]:
        """Returns (doc_type, raw_content) tuple."""
        raw_content = extract_text_tool(file_path)

        if raw_content.startswith("ERROR"):
            # Use a tuple for error handoff in LCEL
            return "Extraction Error", raw_content 

        snippet = raw_content[:4000]
        doc_type = self.classify_document(snippet)
        return doc_type, raw_content

# --- AGENT 2: ProcessingAgent ---

class ProcessingAgent:
    """Handles translation and contextual summarization."""
    
    def __init__(self, llm_model: str = "gemini-2.5-flash", temperature: float = 0.2, api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature,google_api_key=api_key)
        
        self.translation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Translate the entire document provided below into {target_language}. Maintain the original tone and technical terminology."),
                ("user", "Document to Translate:\n\n---\n{raw_content}\n---"),
            ]
        )
        
        self.summarization_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert summarizer. Summarize the following {doc_type} into a detailed, five-paragraph summary in {target_language}. Focus on: Main thesis/goal, methodology/key clauses, main results/findings, and conclusion."),
                ("user", "Translated Document:\n\n---\n{translated_content}\n---"),
            ]
        )
    
    # This run method expects named arguments from the LCEL map
    def run(self, doc_type: str, raw_content: str, target_language: str) -> Tuple[str, str]:
        """Returns (translated_content, summary) tuple."""
        
        # 1. TRANSLATION STEP
        translation_chain = self.translation_prompt | self.llm
        translated_content = translation_chain.invoke({
            "target_language": target_language,
            "raw_content": raw_content
        }).content
        
        # 2. SUMMARIZATION STEP
        summarization_chain = self.summarization_prompt | self.llm
        summary_content = summarization_chain.invoke({
            "doc_type": doc_type,
            "target_language": target_language,
            "translated_content": translated_content
        }).content
        
        return translated_content, summary_content

# --- AGENT 3: TutorAgent (RAG System Builder) ---

class TutorAgent:
    """
    Agent responsible for ingesting processed content into a RAG system 
    and answering user queries based on that content.
    """
    
    def __init__(self, llm_model: str = "gemini-2.5-flash", embedding_model: str = "text-embedding-004", api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.0,google_api_key=api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model,google_api_key=api_key)
        self.vectorstore = None
        
    def _ingest_documents(self, translated_content: str, doc_type: str):
        """Internal method to split text and populate the FAISS VectorStore."""
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        
        # Create a document object with metadata
        doc = {"page_content": translated_content, "metadata": {"source": "uploaded_document", "type": doc_type}}
        
        # Split the content
        split_docs = text_splitter.create_documents([doc["page_content"]], metadatas=[doc["metadata"]])

        # Create the VectorStore (In-memory FAISS)
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        
    def run(self, translated_content: str, summary: str, user_query: str, doc_type: str) -> str:
        """The main method to execute the RAG workflow (Ingestion + Query)."""
        
        # STEP 1: INGESTION
        self._ingest_documents(translated_content, doc_type)

        # STEP 2: RAG QUERY
        retriever = self.vectorstore.as_retriever(k=3)
        retrieved_docs = retriever.invoke(user_query)
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert tutor. Answer the user's question only based on the following context derived from a {doc_type}. "
                    "Use the provided summary for high-level context, but use the detailed context for specific answers. "
                    "If the answer is not in the context, state clearly: 'I cannot find the answer in the provided document.'"
                    "\n\n--- SUMMARY ---\n{summary}"
                    "\n\n--- DETAILED CONTEXT ---\n{context}"
                ),
                ("user", "User Question: {user_query}"),
            ]
        )
        
        rag_chain = rag_prompt | self.llm
        
        final_answer = rag_chain.invoke({
            "doc_type": doc_type,
            "summary": summary,
            "context": context,
            "user_query": user_query
        }).content
        
        return final_answer

# --- LCEL MAPPING FUNCTIONS (The Handoff Glue) ---

def map_agent_1_output_to_state(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Agent 1's tuple output and maps to state keys."""
    doc_type, raw_content = data["agent_1_output"]
    
    if doc_type == "Extraction Error":
        # Raise an error to halt the entire chain in case of file failure
        raise ValueError(f"Pipeline Halted: Extraction Error - {raw_content}")

    return {
        "doc_type": doc_type,
        "raw_content": raw_content,
        **data # Pass the rest of the original state (file_path, target_language, user_query)
    }

def map_agent_2_output_to_state(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Agent 2's tuple output and maps to state keys."""
    translated_content, summary = data["agent_2_output"]
    
    return {
        "translated_content": translated_content,
        "summary": summary,
        **data # Pass the rest of the state
    }

# --- THE ORCHESTRATOR ---

def create_lcel_pipeline(classifier_agent, processor_agent, tutor_agent):
    """
    Creates the complete, sequential LCEL pipeline.
    """
    
    # 1. Define the input schema for the whole pipeline
    # The chain starts with a single dictionary {"file_path": str, "target_language": str, "user_query": str}
    pipeline_input = RunnablePassthrough()

    # 2. Agent 1: Classification and Scraping
    # Runs the classifier_agent.run(file_path) and stores the (doc_type, raw_content) tuple
    agent_1_chain = RunnablePassthrough.assign(
        agent_1_output=itemgetter("file_path") | RunnableLambda(classifier_agent.run)
    )
    
    # 3. Map Agent 1 Output to state keys
    mapper_1_chain = RunnableLambda(map_agent_1_output_to_state)

    # 4. Agent 2: Processing (Translation & Summarization)
    # Extracts the required inputs (doc_type, raw_content, target_language) and runs processor_agent
    agent_2_chain = RunnablePassthrough.assign(
        agent_2_output=RunnableLambda(
            lambda x: processor_agent.run(
                doc_type=x["doc_type"],
                raw_content=x["raw_content"],
                target_language=x["target_language"]
            )
        )
    )

    # 5. Map Agent 2 Output to state keys
    mapper_2_chain = RunnableLambda(map_agent_2_output_to_state)

    # 6. Agent 3: Tutor (RAG)
    # Extracts the required inputs and runs tutor_agent
    agent_3_chain = RunnablePassthrough.assign(
        final_answer=RunnableLambda(
            lambda x: tutor_agent.run(
                translated_content=x["translated_content"],
                summary=x["summary"],
                user_query=x["user_query"],
                doc_type=x["doc_type"]
            )
        )
    )

    # 7. Final Output Selector
    final_output_selector = {
        "document_type": itemgetter("doc_type"),
        "raw_text": itemgetter("raw_content"),
        "translated_content": itemgetter("translated_content"),
        "summary": itemgetter("summary"),
        "final_answer": itemgetter("final_answer")
    }

    # Assembly of the Full Pipeline
    full_pipeline = (
        pipeline_input 
        | agent_1_chain 
        | mapper_1_chain
        | agent_2_chain
        | mapper_2_chain
        | agent_3_chain
        | final_output_selector
    )
    
    return full_pipeline

# --- MAIN EXECUTION ---

def run_pipeline(file_path: str, target_language: str, user_query: str) -> dict:
    
    # Check environment variable
    load_dotenv()
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        return {"status": "Failed", "error": "GEMINI_API_KEY environment variable not set."}

    # 1. Initialize Agents
    classifier_agent = ClassifierScraperAgent(api_key=GEMINI_KEY)
    processor_agent = ProcessingAgent(api_key=GEMINI_KEY)
    tutor_agent = TutorAgent(api_key=GEMINI_KEY)

    # 2. Create and Invoke Pipeline
    pipeline = create_lcel_pipeline(classifier_agent, processor_agent, tutor_agent)
    
    # Define the starting input for the chain
    start_input = {
        "file_path": file_path,
        "target_language": target_language,
        "user_query": user_query
    }
    
    print("\n Running Multi-Agent LCEL Pipeline...")
    
    try:
        final_result = pipeline.invoke(start_input)
        final_result["status"] = "Success"
        return final_result
    
    except ValueError as e:
        # Catch the intentional pipeline halt error
        if "Pipeline Halted" in str(e):
            return {"status": "Failed", "error": str(e).replace("Pipeline Halted: ", "")}
        raise e

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    # NOTE: Replace 'sample_doc.pdf' with the path to a test PDF file on your system
    TEST_FILE_PATH = "CV_Nikhil.pdf"  
    
    # Check if a dummy file exists for the demonstration (replace this with a real check if needed)
    if not os.path.exists(TEST_FILE_PATH):
         print(f" ERROR: Please create a dummy file named '{TEST_FILE_PATH}' to run the demo, or update the path.")
    else:
        results = run_pipeline(
            file_path=TEST_FILE_PATH, 
            target_language="English",
            user_query="Describe the skills of Nikhil from this pdf."
        )
        
        print("\n" + "="*50)
        print("FINAL ORCHESTRATION RESULTS")
        print("="*50)
        print(f"Status: {results.get('status')}")
        if results.get('status') == 'Failed':
             print(f"Error: {results.get('error')}")
        else:
            print(f"Document Type: {results.get('document_type')}")
            print(f"Raw Text Length: {len(results.get('raw_text', ''))} characters")
            print(f"Summary (Snippet): {results.get('summary', '')[:250]}...")
            print("\n--- TUTOR'S ANSWER ---")
            print(results.get('final_answer'))
            print("="*50)


# In[ ]:




