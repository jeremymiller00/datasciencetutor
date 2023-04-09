import pathlib
import os.path
import logging
import sys
import glob

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def setup_logging(log_dir=None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_dir:
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_dir, 'qanda.log')
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_format)


def build_index(library_dir=None):
    embedding=OpenAIEmbeddings(
        # this is the embedding model for searching for related chunks
        document_model_name="text-embedding-ada-002",
        query_model_name="text-embedding-ada-002",
        chunk_size=1024, max_retries=1
        )
    
    if os.path.isdir('datasciencetutor/db') and os.path.isfile('datasciencetutor/db/chroma-embeddings.parquet'):
        logging.info("Loading database from disk")
        db = Chroma(
            persist_directory='datasciencetutor/db', 
            embedding_function=embedding
            )
        logging.info("Database loaded from disk")

    else:
        logging.info("Building database")
        one_book = "/Users/Jeremy/Documents/Data_Science/Projects/datasciencetutor/datasciencetutor/library/ISLR Seventh Printing.pdf"
        loader = PyPDFLoader(one_book)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(documents)
        db = Chroma.from_documents(
            documents=texts, 
            embedding=embedding, 
            persist_directory="db"
            )
        db.persist()
        logging.info("Database persisted")

    retriever = db.as_retriever(search_kwargs={"k":2})
    # retriever=VectorStoreRetriever(
    #     vectorstore=db, 
    #     search_kwargs={"filter":{"type":"filter"},"k":3}
    #     )

    qa = RetrievalQA.from_chain_type(
        # this is the chat model that provides the response from the context
        llm=OpenAI(model_name="text-ada-001", n=1, best_of=1,
        temperature=0.7, max_tokens=256, top_p=1, 
        frequency_penalty=0, presence_penalty=0),
        chain_type="stuff", 
        retriever=retriever
        )
    return db, qa

def ask(qa):
    while True:
        print("Type '(exit)' to exit\n")
        question = input("Query:\n\n")
        if question == "(exit)":
            return
        else:
            logging.info(f"Question asked: {question}")
            answer = qa.run(question)
            logging.info(f"Answer provided: {answer}")
            print("\n" + answer + "\n")

def main(log_dir=None):
    setup_logging(log_dir=log_dir)
    db, qa = build_index()
    ask(qa)

######################
if __name__ == '__main__':
    main(log_dir='logs')