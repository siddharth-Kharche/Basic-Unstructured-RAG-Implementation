import os
import streamlit as st
import pandas as pd
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.loaders import Loader
from athina.evals import DoesResponseAnswerQuery
from google.colab import userdata
import utils  # Import your utils module
from datasets import Dataset

# Set API keys
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')
OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

# Setup Streamlit App
st.title("Unstructured RAG with Athina Evaluation")

# Sidebar for file upload
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

# Main content area
if uploaded_file:
    with st.spinner("Processing Document..."):
        # 1. Load and Chunk PDF
        filename = "temp_upload.pdf"
        with open(filename, "wb") as f:
          f.write(uploaded_file.getbuffer())

        pdf_elements = utils.load_and_chunk_pdf(filename)
        os.remove(filename)

        # Display unique categories of parsed elements
        category_counts = utils.check_unique_categories(pdf_elements)
        st.write("Element Categories:", category_counts)

        # Display unique element types
        unique_types = utils.extract_unique_types(pdf_elements)
        st.write("Unique Element Types:", unique_types)

        # 2. Create Langchain Documents
        documents = utils.create_langchain_documents(pdf_elements, uploaded_file.name)

        # 3. Embedding
        embeddings = utils.create_embeddings()

        # 4. Create Vectorstore and Retriever
        vectorstore = utils.create_vectorstore(documents, embeddings)
        retriever = utils.create_retriever(vectorstore)
        # 5. Load LLM
        llm = utils.ChatOpenAI()
        # 6. Create RAG Chain
        rag_chain = utils.create_rag_chain(retriever, llm)


        # Input question
        question = st.text_input("Ask a question about the document:", value="Compare all the Training Results on MATH Test Set")

        if question:
           # 7. Prepare Data for Evaluation
            data_dict = utils.prepare_data_for_evaluation(rag_chain, retriever, [question])
           
            # Convert the dictionary to a pandas dataframe
            df = pd.DataFrame(data_dict)
             # Convert the DataFrame to a dictionary
            df_dict = df.to_dict(orient='records')

            # Convert context to list
            for record in df_dict:
              if not isinstance(record.get('context'), list):
                  if record.get('context') is None:
                      record['context'] = []
                  else:
                      record['context'] = [record['context']]
            
            dataset = Loader().load_dict(df_dict)

           # Display response
            with st.spinner("Generating response..."):
              response = rag_chain.invoke(question)
              st.write("Response:", response)
           # Evaluate and display results
            with st.spinner("Evaluating..."):
               eval_df = DoesResponseAnswerQuery(model="gpt-4o").run_batch(data=dataset).to_df()
               st.write("Evaluation Results:", eval_df)
else:
    st.write("Please upload a PDF document to get started.")