import pathlib
import os.path
import logging
import sys
import glob
import argparse
import subprocess

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, ObsidianLoader
# from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


# Model pricing https://openai.com/pricing

config = {
    'embedding_model': 'text-embedding-ada-002',
    # 'qanda_model': 'gpt-3.5-turbo', # use k=4
    'qanda_model': 'text-ada-001', # use k=2
    'k_chunks': 4,
    'chunk_size': 1024,
    'chunk_overlap': 32
    }


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
        document_model_name=config['embedding_model'],
        query_model_name=config['embedding_model'],
        chunk_size=config['chunk_size'], max_retries=1
        )
    
    # load the db from disk
    if os.path.isdir('dbs/clarivatevaultdb') and os.path.isfile('dbs/clarivatevaultdb/chroma-embeddings.parquet'):
        logging.info("Loading database from disk")
        db = Chroma(
            persist_directory="dbs/clarivatevaultdb", 
            embedding_function=embedding
            )
        logging.info("Database loaded from disk")

    # build the db if does not exist
    else:
        # code block for folder of pdfs
        # logging.info("Getting books")
        # books = glob.glob(library_dir + '/*.pdf')
        # documents = []
        # one_book = "/Users/Jeremy/Documents/Data_Science/Projects/datasciencetutor/datasciencetutor/library/ISLR Seventh Printing.pdf"
        # for book in books:
        #     logging.info(f"Loading book: {book}")
        #     loader = PyPDFLoader(book)
        #     docs = loader.load()
        #     documents.extend(docs)

        # code block for obsidian vault
        logging.info("Getting vault files")
        # loader = ObsidianLoader("/Users/Jeremy/Clarivate-Vault")
        loader = ObsidianLoader("/Users/jeremymiller/Desktop/Clarivate-Vault")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])
        texts = text_splitter.split_documents(documents)
        logging.info("Building database")
        db = Chroma.from_documents(
            documents=texts, 
            embedding=embedding, 
            persist_directory="dbs/clarivatevaultdb"
            )
        db.persist()
        logging.info("Database persisted")

    retriever = db.as_retriever(search_kwargs={"k":config['k_chunks']})

    # this block uses a gpt-3 model
    qa = RetrievalQA.from_chain_type(
        # this is the chat model that provides the response from the context
        llm=OpenAI(model_name=config['qanda_model'], n=1, best_of=1,
        temperature=0.7, max_tokens=256, top_p=1, 
        frequency_penalty=0, presence_penalty=0),
        chain_type="stuff", 
        retriever=retriever
        )

    # # this block uses a gpt-3.5 model
    # qa = RetrievalQA.from_chain_type(
    #     # this is the chat model that provides the response from the context
    #     llm=ChatOpenAI(model_name=config['qanda_model'], n=1,
    #     temperature=0.7, max_tokens=256, top_p=1, 
    #     frequency_penalty=0, presence_penalty=0),
    #     chain_type="stuff", 
    #     retriever=retriever
    #     )

    return db, qa

def ask(qa):
    while True:
        print("Type '(exit)' to exit\n")
        question = input("Question:\n\n")
        if question == "(exit)":
            sys.exit(0)
        else:
            logging.info(f"Question asked: {question}")
            try:
                # local alpaca response model - no context yet
                logging.info("Writing file")
                with open("temp", "w") as file:
                    file.write(question)
                logging.info("Executing alpaca subprocess")
                cmd = "/Users/jeremymiller/alpaca.cpp/chat -f temp"
                result = subprocess.run(cmd, shell=True, text=True,
                                        capture_output=True)
                answer = result.stdout
                print(answer)
                # OpenAI response model
                # answer = qa.run(question)
                logging.info(f"Answer provided: {answer}")
                print("\n" + answer + "\n")
            except BaseException as e:
                print(f"An exception has occurred: {e}\n")

def search(db):
    while True:
        print("Type '(exit)' to exit\n")
        query = input("Query:\n\n")
        if query == "(exit)":
            sys.exit(0)
        else:
            logging.info(f"Query searched: {query}")
            try:
                docs = db.similarity_search_with_score(query, k=config["k_chunks"])
                for doc in docs:
                    print("\n<<<>>>\n")
                    print(doc)
                    print("\n<<<>>>\n")
            except BaseException as e:
                print(f"An exception has occurred: {e}\n")

def parse():
    parser = argparse.ArgumentParser(
        description="Query or ask a question over a document set")
    parser.add_argument("-f", "--function", choices=["ask", "search"], default='ask')
    args = parser.parse_args()
    return args

def main(log_dir=None, library_dir=None, function=None):
    setup_logging(log_dir=log_dir)
    db, qa = build_index(library_dir=library_dir)
    if function == "ask":
        ask(qa)
    elif function == "search":
        search(db)
    else:
        print(f"Unknown function: {function}. Available functions are [ask, search]")
        sys.exit(1)

######################
if __name__ == '__main__':
    args = parse()
    main(
        log_dir='logs', 
        library_dir='datasciencetutor/library', 
        function=args.function)