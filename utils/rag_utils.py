import bs4
from typing import List

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


class RAGPipeline:
    def __init__(self, open_model: str = "gpt-3.5-turbo-0125") -> None:
        self.llm = ChatOpenAI(model=open_model)
        self.docs = self.load_documents()
        splits = self.chunk_documents(self.docs)
        self.vector_store = self.create_vector_store(splits)
        self.rag_chain = self.create_rag_chain(self.vector_store, self.llm)

    def load_documents(self) -> List[Document]:
        loader = WebBaseLoader(
            web_paths=(
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://www.gov.pl/",
                "https://isap.sejm.gov.pl/",
                "https://e-justice.europa.eu/6/PL/national_legislation",
            ),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        return loader.load()

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        return text_splitter.split_documents(docs)

    def create_vector_store(self, splits: List[Document]) -> Chroma:
        return Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    def create_rag_chain(self, vector_store: Chroma, llm: ChatOpenAI):
        retriever = vector_store.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    def run(self, question: str) -> str:
        return self.rag_chain.invoke(question)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    print(pipeline.run("What is Task Decomposition?"))
