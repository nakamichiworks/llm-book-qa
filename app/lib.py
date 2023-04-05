import streamlit as st
from langchain import OpenAI
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import (
    VectorStoreInfo,
    VectorStoreToolkit,
    create_vectorstore_agent,
)
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore
from lxml import etree
from unstructured.documents.html import HTMLDocument
from unstructured.file_utils.file_conversion import convert_epub_to_html
from unstructured.partition.common import add_element_metadata, document_to_element_list


class UnstructuredEPubLoader_(UnstructuredEPubLoader):
    """Workaround for https://github.com/Unstructured-IO/unstructured/issues/436"""

    def _get_elements(self) -> list:
        html_text = convert_epub_to_html(filename=self.file_path)
        document = HTMLDocument.from_string(html_text)
        document.document_tree = etree.fromstring(
            html_text, parser=etree.HTMLParser(remove_comments=True)
        )
        layout_elements = document_to_element_list(document, include_page_breaks=False)
        return add_element_metadata(
            layout_elements,
            include_page_breaks=True,
            filename=self.file_path,
        )


@st.cache_resource
def create_book_store(file: str, key: str) -> VectorStore:
    loader = UnstructuredEPubLoader_(file)
    documents = loader.load()

    # TODO: Try SpacyTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # TODO: Try HuggingFaceEmbeddings
    embeddings = OpenAIEmbeddings()  # type: ignore
    book_store = Chroma.from_documents(texts, embeddings, collection_name=key)
    return book_store


def create_book_qa_agent(
    book_file: str,
    book_store_key: str,
    book_description: str,
    temperature: float = 0.1,
    verbose: bool = False,
) -> AgentExecutor:
    llm = OpenAI(temperature=temperature)  # type: ignore
    book_store = create_book_store(book_file, book_store_key)
    vectorstore_info = VectorStoreInfo(
        name=book_store_key, description=book_description, vectorstore=book_store
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=verbose)
    return agent_executor
