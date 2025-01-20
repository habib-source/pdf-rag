
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

class ChatPDF:
    my_embedder = "./embed.gguf"
    llm_emb=LlamaCppEmbeddings(
        model_path=my_embedder,
        n_gpu_layers=13,
        n_batch=1810,
        )
    my_model = "./model.gguf"
    llm = ChatLlamaCpp(
        model_path=my_model,
        n_ctx=1000,
        max_tokens=1512,
        n_gpu_layers=33,
        n_batch=925,
    )
    def __init__(self):
        self.chat_history = []
        self.rag_chain = None
        self.retriever = None
        dim = len(self.llm_emb.embed_query("hello world"))
        index = faiss.IndexFlatL2(dim)
        self.vectorstore = FAISS(
            embedding_function=self.llm_emb,
            index=index,
            docstore= InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        self.vectorstore.add_documents(documents=splits)
        self.retriever = self.vectorstore.as_retriever()
        question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)

    def ask(self, query: str):
        if not self.rag_chain:
            return "Please, add a PDF document first."
        return self.rag_chain.stream({"input": query, "chat_history": self.chat_history})

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.raf_chain = None
