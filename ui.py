import os
import tempfile
import streamlit as st
from loed import ChatPDF
st.set_page_config(page_title="ChatPDF")

def stream_data(msg):
    for chunk in msg:
        if chunk.get("answer")!= None:
            yield chunk.get("answer")
def display_messages():
    st.subheader("Chat")
    for i,(msg, is_user) in enumerate(st.session_state["messages"]):
        if is_user:
            with st.chat_message("user"):
                st.write(msg)
        else:
            with st.chat_message("ai"):
                if  isinstance(msg, str):
                    st.write(msg)
                else:
                    msg = st.write_stream(stream_data(msg))
                    st.session_state["messages"][i]=(msg, False)

    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    user_input = st.session_state.get("user_input")
    if user_input and len(user_input) > 0:
        user_text = user_input
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    st.session_state["assistant"].clear()

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("ChatPDF")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.chat_input("Message", key="user_input", on_submit=process_input)


if __name__ == "__main__":
    page()
