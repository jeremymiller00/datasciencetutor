import pathlib
import os.path
import logging
import sys
import glob

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
# from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


config = {
    'embedding_model': 'text-embedding-ada-002',
    'qanda_model': 'text-ada-001',
    'k_chunks': 2,
    'chunk_size': 1024,
    'chunk_overlap': 32
    }


def setup_logging(log_dir=None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_dir:
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_dir, 'qanda.log')
        logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)


def build_index(library_dir=None):
    embedding=OpenAIEmbeddings(
        # this is the embedding model for searching for related chunks
        document_model_name=config['embedding_model'],
        query_model_name=config['embedding_model'],
        chunk_size=config['chunk_size'], max_retries=1
        )
    
    # load the db from disk
    if os.path.isdir('db') and os.path.isfile('db/chroma-embeddings.parquet'):
        logging.info("Loading database from disk")
        db = Chroma(
            persist_directory='db', 
            embedding_function=embedding
            )
        logging.info("Database loaded from disk")

    # build the db if does not exist
    else:
        logging.info("Getting books")
        books = glob.glob(library_dir + '/*.pdf')
        documents = []
        # one_book = "/Users/Jeremy/Documents/Data_Science/Projects/datasciencetutor/datasciencetutor/library/ISLR Seventh Printing.pdf"
        for book in books:
            logging.info(f"Loading book: {book}")
            loader = PyPDFLoader(book)
            docs = loader.load()
            documents.extend(docs)
        text_splitter = CharacterTextSplitter(
            chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])
        texts = text_splitter.split_documents(documents)
        logging.info("Building database")
        db = Chroma.from_documents(
            documents=texts, 
            embedding=embedding, 
            persist_directory="db"
            )
        db.persist()
        logging.info("Database persisted")

    retriever = db.as_retriever(search_kwargs={"k":config['k_chunks']})

    qa = RetrievalQA.from_chain_type(
        # this is the chat model that provides the response from the context
        llm=OpenAI(model_name=config['qanda_model'], n=1, best_of=1,
        temperature=0.7, max_tokens=256, top_p=1, 
        frequency_penalty=0, presence_penalty=0),
        chain_type="stuff", 
        retriever=retriever
        )
    return db, qa

def ask(qa):
    while True:
        print("Type '(exit)' to exit\n")
        question = input("Question:\n\n")
        if question == "(exit)":
            return
        else:
            logging.info(f"Question asked: {question}")
            try:
                answer = qa.run(question)
                logging.info(f"Answer provided: {answer}")
                print("\n" + answer + "\n")
            except BaseException as e:
                print(f"An exception has occurred: {e}\n")

def main(log_dir=None, library_dir=None):
    setup_logging(log_dir=log_dir)
    db, qa = build_index(library_dir=library_dir)
    ask(qa)

######################
if __name__ == '__main__':
    main(log_dir='logs', library_dir='datasciencetutor/library')