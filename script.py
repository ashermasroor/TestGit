from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader, SeleniumURLLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from textgen import TextGen

from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.openapi.docs import get_swagger_ui_html

from pydantic import BaseModel, Field
import os
import re
import requests

from bs4 import BeautifulSoup


class RAGLlm(BaseModel):
    model_url: str | None = Field(default="http://localhost:5000")
    context: str | None = Field(default="https://www.aqa.org.uk/subjects/science/as-and-a-level/chemistry-7404-7405/subject-content/organic-chemistry", description="PDF file path on local. (eg. orient-context.pdf, webermeyer-context.pdf)")
    prompt: str | None = Field(default=None, description="User input")

class RAGLlm2(BaseModel):
    model_url: str | None = Field(default="http://localhost:5000")
    context: list| None = Field(default =[
    "https://studio.orient-telecoms.com/publishing/67534-case-1.docx",
    "https://studio.orient-telecoms.com/publishing/8198-webermeyer-context.pdf",
    "https://studio.orient-telecoms.com/publishing/29447-creating-template.txt"
    ],
    description="list of document URLS or File paths, the file name can be found at the end of the link(eg. 67534-case-1.docx,8198-webermeyer-context.pdf)")
    prompt: str | None = Field(default=None, description="User input")

class RAGLlm3(BaseModel):
    model_url: str | None = Field(default="http://localhost:5000")
    character_id: str | None = Field(default="OT-PLC/150424022018")
    context: list| None = Field(default =[
    "https://gist.githubusercontent.com/EdwardRayl/3436572afde8ce9e3faf5b7b95356a49/raw/6b25895fce480713560829dec31ac8220ffe5272/gists.txt",
    "https://www.rcrc-resilience-southeastasia.org/wp-content/uploads/2017/12/Contracts-Act-1950.pdf",
    "https://www.eng.uc.edu/~beaucag/Classes/AdvancedMaterialsThermodynamics/Books/Modern%20Physics%20for%20Scientists%20and%20Engineers.pdf",
    "https://sites.ntc.doe.gov/partners/tr/Training%20Textbooks/07-Chemistry/1-Module%201-Fundamentals%20of%20Chemistry.pdf"
    ],
    description="PDF file path on local. (eg. orient-context.pdf, webermeyer-context.pdf)")
    prompt: str | None = Field(default=None, description="User input")

app = FastAPI(docs_url=None)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.options("/")
async def options_route():
    return JSONResponse(content="OK")

@app.post("/rag", summary="Production RAG")
async def rag(request: Request, request_data: RAGLlm):
    """
        Naive RAG implementation with no data persistence.
    """
    
    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct", temperature=0.1, repetition_penalty=1.1, max_new_tokens=1000, truncation_length=32768, do_sample=True)

    loader = PyPDFLoader(request_data.context)
    pages = loader.load_and_split()
    
    db = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    # jinaai/jina-embeddings-v2-base-en, sentence-transformers/all-mpnet-base-v2, all-MiniLM-L6-v2, intfloat/e5-large-v2

    retriever_pdf = db.as_retriever(
        search_kwargs={'k': 5},
        # search_type="similarity",
    )

    rag_chain = ( 
    {"context": retriever_pdf, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )
    print(pages)
    response = rag_chain.invoke(f"{request_data.prompt}")
    return response

@app.post("/v2/rag", summary="Testing RAG")
async def rag2(request: Request, request_data: RAGLlm):
    """
        Naive RAG with added functionality of data persistence in local file ./store/
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
 
    match = re.search(r'\/([^\/]+)\.pdf$', request_data.context)
    doc_name = match.group(1) if match else None
    doc_dir = f'./store/{doc_name}'
    
    if os.path.isdir(doc_dir):
        print(f"Data file: '{doc_name}' already existed.")
        db = FAISS.load_local(doc_dir, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
        # jinaai/jina-embeddings-v2-base-en, sentence-transformers/all-mpnet-base-v2, all-MiniLM-L6-v2, intfloat/e5-large-v2
    
    else:
        print(f"Data file: '{doc_name}' does not exists.")
        loader = PyPDFLoader(request_data.context)
        pages = loader.load_and_split()
        db = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
        db.save_local(doc_dir)

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct", temperature=0.1, repetition_penalty=1.1, max_new_tokens=1000, truncation_length=32768, do_sample=True)

    retriever_pdf = db.as_retriever(search_kwargs={'k': 5})
    
    rag_chain = ( 
    {"context": retriever_pdf, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response

@app.post("/v3/rag", summary="Testing RAG")
async def rag3(request: Request, request_data: RAGLlm):
    """
        Naive RAG
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct", temperature=0.1, repetition_penalty=1.1, max_new_tokens=1000, truncation_length=32768, do_sample=True)
 
    #central store for FAISS index
    central_index_dir = './store/faiss_index/'
  
    #load or create new FAISS index
    if os.path.exists(central_index_dir):
        print("loaded vectors")
        db = FAISS.load_local(central_index_dir, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    else:
        print("create vectors")
        db = None
        
        
    #load new pdf doc,log doc name,process and update index
    context = request_data.context
    filenamePDF = os.path.basename(request_data.context)
    
    #URLloader Function and context retrieval
    def URLTitle(context):
        response = requests.get(context)
        soup = BeautifulSoup(response.content,'html.parser')
        title = soup.title.string
        return title
    filenameURL = URLTitle(context)

    filePDF = re.search(r"pdf$",context)
    fileURL =  re.search(r"^http",context)
    
    if filePDF: 
        print("file type is PDF")
        loader = PyPDFLoader(request_data.context)
        pages = loader.load_and_split()
        filename = filenamePDF

    elif fileURL:
        print("file type is URl")
        loader = SeleniumURLLoader(urls = [request_data.context])
        pages = loader.load()
        filename = filenameURL

    else:
        print("file type is doc/docx")
        loader= UnstructuredFileLoader(request_data.context)
        pages = loader.load_and_split()
        filename = filenamePDF

    new_db = FAISS.from_documents(pages,HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    if db:
        db.merge_from(new_db)
    else:
        db = new_db

    db.save_local(central_index_dir)

    #                                                   script to log file names
    log_file_path = './store/faiss_index/FI_doclog.txt'
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    try:
        # Initialize set to keep existing filenames from the log
        existing_files = set()
        # Read the current log to prevent duplicate entries (optimized to open once)
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as file:
                existing_files.update(file.read().splitlines())

        
        # Check if the PDF is already logged
        if filename not in existing_files:
            with open(log_file_path, 'a') as file:
                file.write(filename + '\n')
            print(f"{filename} added to the log.")
        else:
            print(f"{filename} is already in the log.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#                                                      end of file logging script

    retriever_pdf = db.as_retriever(search_kwargs={'k':5})
    rag_chain = (
        {"context":retriever_pdf, "question":RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response



    # match = re.search(r'\/([^\/]+)\.pdf$', request_data.context)
    # doc_name = match.group(1) if match else None
    # doc_dir = f'./store/{doc_name}'
    
    # if os.path.isdir(doc_dir):
    #     print(f"Data file: '{doc_name}' already existed.")
    #     db = FAISS.load_local(doc_dir, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    #     # jinaai/jina-embeddings-v2-base-en, sentence-transformers/all-mpnet-base-v2, all-MiniLM-L6-v2, intfloat/e5-large-v2
    
    # else:
    #     print(f"Data file: '{doc_name}' does not exists.")
    #     loader = PyPDFLoader(request_data.context)
    #     pages = loader.load_and_split()
    #     db = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    #     db.save_local(doc_dir)

    # textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct", temperature=0.1, repetition_penalty=1.1, max_new_tokens=1000, truncation_length=32768, do_sample=True)

    # # new_db = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    # # new_db.merge_from(db)
    # # new_db.save_local("faiss_index")
    # # retriever_pdf = new_db.as_retriever(search_kwargs={'k': 5})
    
    # rag_chain = ( 
    # {"context": retriever_pdf, "question": RunnablePassthrough()}
    #     | prompt
    #     | textgen_llm
    # )

    # response = rag_chain.invoke(f"{request_data.prompt}")
    # return response
@app.post("/v4/rag", summary="Testing RAG V4")
async def rag4(request_data: RAGLlm3):
    """
        Naive RAG with added functionality:
        - Vector database saved based on character ID and documents for each character ID can be tracked inside character/document_logs.txt
        - Only accepts documents in URL
        - Able to handle pdf, doc, docx, txt file types
        - Able to query multiple documents
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct", temperature=0.1, repetition_penalty=1.1, max_new_tokens=1000, truncation_length=32768, do_sample=True)
    
    char_dir = f'character/{request_data.character_id}'
    doc_log = f'{char_dir}/document_logs.txt'
    temp_dir = f'./temp/{request_data.character_id}'
    os.makedirs(temp_dir, exist_ok=True)

    if os.path.isdir(char_dir): # Check if character exists
        print("Character exists.")

        f = open(doc_log, "r") # TODO read file
        cur_list = set()
        for x in f:
            cur_list.add(x.rstrip('\n'))
        f.close()

        new_list = set()
        doc_map = {}
        for idx, value in enumerate(request_data.context):
            match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', value)
            doc_name = match.group(1) if match else None
            new_list.add(doc_name)
            doc_map[doc_name] = value # Store doc url and accessible by document name

        if new_list == cur_list:
            print(f"Same set: {cur_list} && {new_list}")
            faiss_index = FAISS.load_local(char_dir, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

        elif cur_list.issubset(new_list):
            print(f"{cur_list} is a subset of {new_list}")
            faiss_index = FAISS.load_local(char_dir, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
            
            new_doc = new_list - cur_list
            for doc in new_doc:
                f = open(doc_log, "a") # TODO append new document name at doc_log directory
                f.write(f'{doc_name}\n')
                f.close()
                response = requests.get(doc_map[doc])
                if response.status_code == 200:
                    doc_path = os.path.join(temp_dir, os.path.basename(doc_map[doc]))
                    with open(doc_path, 'wb') as f:
                        f.write(response.content)
                        print ("File successfully downloaded.")

                else:
                    print("Unsuccessful file download")
                    doc_path = doc_map[doc]

                filePDF = re.search(r"pdf$", doc_path)
                print(doc_path)
                if filePDF: # Checks if file type is pdf, else .doc/.docx/.txt
                    loader = PyPDFLoader(doc_path)
                else:
                    loader = UnstructuredFileLoader(doc_path)

                pages = loader.load_and_split()
                
                faiss_index_i = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                faiss_index.merge_from(faiss_index_i)
                os.remove(doc_path)
            faiss_index.save_local(char_dir)

        else:
            print(f"Not a subset and different list: {cur_list} && {new_list}")
            with open(doc_log, 'w') as f: # TODO create a new empty document_logs.txt
                pass
            for idx, value in enumerate(request_data.context):
                f = open(doc_log, "a") # TODO append new document name at doc_log directory
                f.write(f'{doc_name}\n')
                f.close()

                response = requests.get(value)
                if response.status_code == 200:
                    doc_path = os.path.join(temp_dir, os.path.basename(value))
                    with open(doc_path, 'wb') as f:
                        f.write(response.content)
                    print ("File successfully downloaded.")
                else:
                    print("Unsuccessful file download. Resorting to local path.")
                    doc_path = value
            
                # Only for printing documents name and no other functions
                match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', doc_path)
                doc_name = match.group(1) if match else None
                print(doc_name)

                filePDF = re.search(r"pdf$", doc_path)
                print(doc_path)
                if filePDF: # Checks if file type is pdf, else .doc/.docx/.txt
                    loader = PyPDFLoader(doc_path)
                else:
                    loader = UnstructuredFileLoader(doc_path)

                pages = loader.load_and_split()

                if idx == 0:
                    faiss_index = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                else:
                    faiss_index_i = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                    faiss_index.merge_from(faiss_index_i)

                os.remove(doc_path)
            faiss_index.save_local(char_dir)
    
    else:
        print("Character does not exist.")
        os.makedirs(char_dir)
        for idx, value in enumerate(request_data.context):
            response = requests.get(value)
            if response.status_code == 200:
                doc_path = os.path.join(temp_dir, os.path.basename(value))
                with open(doc_path, 'wb') as f:
                    f.write(response.content)
                print ("File successfully downloaded.")
            else:
                print("Unsuccessful file download. Resorting to local path.")
                doc_path = value
            
            # Only for printing documents name and no other functions
            match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', doc_path)
            doc_name = match.group(1) if match else None
            print(doc_name)
            f = open(doc_log, "a+") # TODO append new document name at doc_log directory
            f.write(f'{doc_name}\n')
            f.close()

            filePDF = re.search(r"pdf$", doc_path)
            print(doc_path)
            if filePDF: # Checks if file type is pdf, else .doc/.docx/.txt
                loader = PyPDFLoader(doc_path)
            else:
                loader = UnstructuredFileLoader(doc_path)

            pages = loader.load_and_split()

            if idx == 0:
                faiss_index = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
            else:
                faiss_index_i = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                faiss_index.merge_from(faiss_index_i)

            os.remove(doc_path)
        faiss_index.save_local(char_dir)
    
 
    os.rmdir(temp_dir)


    retriever = faiss_index.as_retriever(search_kwargs={'k': 5})

    rag_chain = ( 
    {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response


@app.post("/v4.1/rag", summary="Testing RAG V4")
async def rag4(request_data: RAGLlm3):
    """
        Naive RAG with added functionality:
        - Vector database saved based on character ID and documents for each character ID can be tracked inside character/document_logs.txt
        - Only accepts documents in URL
        - Able to handle pdf, doc, docx, txt file types
        - Able to query multiple documents
        - Testing Implementation of Selenium URL Loader
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct", temperature=0.1, repetition_penalty=1.1, max_new_tokens=1000, truncation_length=32768, do_sample=True)
    
    #Function to retrieve website title from URL
    def URLTitle(value):
        response = requests.get(value)
        soup = BeautifulSoup(response.content,'html.parser')
        title = soup.title.string
        return title
    
    context = request_data.context
    
    char_dir = f'character/{request_data.character_id}'
    doc_log = f'{char_dir}/document_logs.txt'
    temp_dir = f'./temp/{request_data.character_id}'
    os.makedirs(temp_dir, exist_ok=True)
    
    if os.path.isdir(char_dir): # Check if character exists
        print("Character exists.")

        f = open(doc_log, "r") # TODO read file
        cur_list = set()
        for x in f:
            cur_list.add(x.rstrip('\n'))
        f.close()

        new_list = set()
        doc_map = {}
        for idx, value in enumerate(request_data.context):
            #file type checking
            filePDF = re.search(r"pdf$", value)
            fileDOC = re.search (r"\.(doc|docx|txt)$",value)
            #if function to log doc name or website title depending on the context
            if filePDF or fileDOC:
                match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', value)
            else:
                match = URLTitle(value)

            doc_name = match.group(1) if match else None
            new_list.add(doc_name)
            doc_map[doc_name] = value # Store doc url and accessible by document name

        if new_list == cur_list:
            print(f"Same set: {cur_list} && {new_list}")
            faiss_index = FAISS.load_local(char_dir, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

        elif cur_list.issubset(new_list):
            print(f"{cur_list} is a subset of {new_list}")
            faiss_index = FAISS.load_local(char_dir, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
            
            new_doc = new_list - cur_list
            for doc in new_doc:
                f = open(doc_log, "a") # TODO append new document name at doc_log directory
                f.write(f'{doc_name}\n')
                f.close()
                response = requests.get(doc_map[doc])
                if response.status_code == 200:
                    doc_path = os.path.join(temp_dir, os.path.basename(doc_map[doc]))
                    with open(doc_path, 'wb') as f:
                        f.write(response.content)
                        print ("File successfully downloaded.")

                else:
                    print("Unsuccessful file download")
                    doc_path = doc_map[doc]

                filePDF = re.search(r"pdf$", doc_path)
                fileDOC = re.search (r"\.(doc|docx|txt)$",doc_path)

                print(doc_path)
                if filePDF: # Checks if file type is pdf, else if .doc/.docx/.txt, else Non-recursive URL loader
                    loader = PyPDFLoader(doc_path)
                elif fileDOC:
                    loader = UnstructuredFileLoader(doc_path)
                else:
                    loader = SeleniumURLLoader(urls = [request_data.context])

                pages = loader.load_and_split()
                
                faiss_index_i = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                faiss_index.merge_from(faiss_index_i)
                os.remove(doc_path)
            faiss_index.save_local(char_dir)

        else:
            print(f"Not a subset and different list: {cur_list} && {new_list}")
            with open(doc_log, 'w') as f: # TODO create a new empty document_logs.txt
                pass
            for idx, value in enumerate(request_data.context):
                f = open(doc_log, "a") # TODO append new document name at doc_log directory
                f.write(f'{doc_name}\n')
                f.close()

                response = requests.get(value)
                if response.status_code == 200:
                    doc_path = os.path.join(temp_dir, os.path.basename(value))
                    with open(doc_path, 'wb') as f:
                        f.write(response.content)
                    print ("File successfully downloaded.")
                else:
                    print("Unsuccessful file download. Resorting to local path.")
                    doc_path = value
            
                
                filePDF = re.search(r"pdf$", doc_path)
                fileDOC = re.search (r"\.(doc|docx|txt)$",doc_path)
                
                # Only for printing documents name and no other functions
                #if function to log doc name or website title depending on the context
                if filePDF or fileDOC:
                    match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', value)
                else:
                    match = URLTitle(value)
                doc_name = match.group(1) if match else None
                print(doc_name)

                print(doc_path)
                if filePDF: # Checks if file type is pdf, else if .doc/.docx/.txt, else Non-recursive URL loader
                    loader = PyPDFLoader(doc_path)
                elif fileDOC:
                    loader = UnstructuredFileLoader(doc_path)
                else:
                    loader = SeleniumURLLoader(urls = [request_data.context])

                pages = loader.load_and_split()

                if idx == 0:
                    faiss_index = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                else:
                    faiss_index_i = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                    faiss_index.merge_from(faiss_index_i)

                os.remove(doc_path)
            faiss_index.save_local(char_dir)
    
    else:
        print("Character does not exist.")
        os.makedirs(char_dir)
        for idx, value in enumerate(request_data.context):
            response = requests.get(value)
            if response.status_code == 200:
                doc_path = os.path.join(temp_dir, os.path.basename(value))
                with open(doc_path, 'wb') as f:
                    f.write(response.content)
                print ("File successfully downloaded.")
            else:
                print("Unsuccessful file download. Resorting to local path.")
                doc_path = value
            
            filePDF = re.search(r"pdf$", doc_path)
            fileDOC = re.search (r"\.(doc|docx|txt)$",doc_path)

            # Only for printing documents name and no other functions
            #if function to log doc name or website title depending on the context                                   
            if filePDF or fileDOC:
                match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', value)
            else:
                match = URLTitle(context)
            doc_name = match.group(1) if match else None
            print(doc_name)
            f = open(doc_log, "a+") # TODO append new document name at doc_log directory
            f.write(f'{doc_name}\n')
            f.close()

            print(doc_path)
            if filePDF: # Checks if file type is pdf, else if .doc/.docx/.txt, else Non-recursive URL loader
                loader = PyPDFLoader(doc_path)
            elif fileDOC:
                loader = UnstructuredFileLoader(doc_path)
            else:
                loader = SeleniumURLLoader(urls = [request_data.context])

            pages = loader.load_and_split()

            if idx == 0:
                faiss_index = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
            else:
                faiss_index_i = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                faiss_index.merge_from(faiss_index_i)

            os.remove(doc_path)
        faiss_index.save_local(char_dir)
    
 
    os.rmdir(temp_dir)


    retriever = faiss_index.as_retriever(search_kwargs={'k': 5})

    rag_chain = ( 
    {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response