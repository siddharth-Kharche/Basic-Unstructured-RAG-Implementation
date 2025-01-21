import os
from langchain_openai import OpenAIEmbeddings
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document
from langchain.vectorstores import FAISS
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def load_and_chunk_pdf(filename):
    """Loads, extracts, and chunks PDF content."""
    pdf_elements = partition_pdf(
    filename=filename,
    extract_images_in_pdf=True,
    strategy = "hi_res",
    hi_res_model_name="yolox",
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=3000,
    combine_text_under_n_chars=200,
    )
    return pdf_elements

def create_langchain_documents(pdf_elements, filename):
    """Converts unstructured elements to langchain documents."""
    documents = [Document(page_content=el.text, metadata={"source": filename}) for el in pdf_elements]
    return documents

def create_vectorstore(documents, embeddings):
    """Creates FAISS vectorstore from documents and embeddings."""
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def create_retriever(vectorstore):
    """Creates a retriever from the vectorstore."""
    retriever = vectorstore.as_retriever()
    return retriever

def create_rag_chain(retriever, llm):
    """Creates the RAG chain."""
    template = """
    You are a helpful assistant that answers questions based on the provided context, which can include text and tables.
    Use the provided context to answer the question.
    Question: {input}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever,  "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def prepare_data_for_evaluation(rag_chain, retriever, questions):
    """Prepares data for evaluation with Athina."""
    responses = []
    contexts = []

    for query in questions:
        responses.append(rag_chain.invoke(query))
        contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

    data = {
        "query": questions,
        "response": responses,
        "context": contexts,
    }
    return data
def create_embeddings():
  """Creates OpenAI embeddings"""
  embeddings = OpenAIEmbeddings()
  return embeddings
def check_unique_categories(pdf_elements):
    """Checks unique categories of parsed elements"""
    category_counts = Counter(str(type(element)) for element in pdf_elements)
    return category_counts

def extract_unique_types(pdf_elements):
    """Extracts unique types of elements."""
    unique_types = {el.to_dict()['type'] for el in pdf_elements}
    return unique_types