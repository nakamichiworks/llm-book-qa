import os

import streamlit as st

from lib import create_book_qa_agent

BOOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BOOKS = {
    "書籍を選択してください": None,
    "書籍名": {
        "file": "書籍名.epub",
        "key": "book_store_key",
        "description": "book content",
    },
}


st.title("Book Q&A")

book_name = st.selectbox("書籍選択", BOOKS.keys())
book_data = None
agent = None
if book_name and BOOKS[book_name]:
    with st.spinner("AIを準備中…"):
        book_data = BOOKS[book_name]
        agent = create_book_qa_agent(
            os.path.join(BOOK_DIR, book_data["file"]),
            book_data["key"],
            book_data["description"],
        )

question = st.text_input("質問", placeholder="この本の概要を教えてください")
answer = ""
if agent is not None and question is not None and question.strip() != "":
    with st.spinner("回答を生成中…"):
        answer = agent.run(question)

st.subheader("回答")
st.write(answer)
