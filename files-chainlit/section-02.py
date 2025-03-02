import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from typing import List
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Verificar que la API key esté configurada
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada. Por favor, configúrala en el archivo .env")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@cl.on_chat_start
async def on_chat_start():
    files = None
    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Por favor, sube un archivo PDF para comenzar!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    
    file = files[0]
    msg = cl.Message(content=f"Procesando `{file.name}`...")
    await msg.send()
    
    # Guardamos temporalmente el archivo
    with open(file.path, "rb") as f:
        # Usamos PyPDFLoader para cargar el archivo PDF
        loader = PyPDFLoader(file.path)
        documents = loader.load()
    
    # Dividimos los documentos
    splits = text_splitter.split_documents(documents)
    
    # Extraemos textos y metadatos de los documentos
    texts = [doc.page_content for doc in splits]
    metadatas = [
        {
            "source": f"{i}-pl", 
            "page": doc.metadata.get("page", "unknown")
        } 
        for i, doc in enumerate(splits)
    ]
    
    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    # Let the user know that the system is ready
    msg.content = f"Procesamiento de `{file.name}` finalizado. ¡Ahora puedes hacer preguntas!"
    await msg.update()
    
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List
    
    text_elements = []  # type: List[cl.Text]
    
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            page_info = f" (Página {source_doc.metadata.get('page', 'desconocida')})"
            
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content,
                    name=source_name + page_info,
                    display="side"
                )
            )
        
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\nFuentes: {', '.join(source_names)}"
        else:
            answer += "\nNo se encontraron fuentes"
    
    await cl.Message(content=answer, elements=text_elements).send()