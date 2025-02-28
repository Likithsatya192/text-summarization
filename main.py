import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader, YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="LangChain: Summarize Text From URL or PDF", page_icon="🦜")
st.title("🦜Likith's: Summarize Text From URL or PDF")
st.subheader("Summarize URL or PDF")

with st.sidebar:
    api_key = st.text_input("Enter your Groq API key: ", type="password")

chunks_prompt="""
    Please summarize the below content:
    Content:{text}
    Summary:
"""
map_prompt_template=PromptTemplate(input_variables=['text'],
                                    template=chunks_prompt)

final_prompt='''
    Provide the final summary of the entire content with these important points.
    Add a Motivation Title,Start the precise summary with an introduction and provide the summary in number 
    points for the content.
    Content:{text}
'''
final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)

if api_key:

    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

    options = st.selectbox("Select: ", ["","URL", "PDF"], format_func=lambda x: "Select an option" if x == "" else x)

    if options == "URL":
        generic_url = st.text_input("Paste your URL: ",label_visibility="collapsed")

        if st.button("Summarize the Content: "):
            if not generic_url.strip():
                st.error("Please provide the information to get started.")
            elif not validators.url(generic_url):
                st.error("Please enter a valid Url. It can may be a YT video url or website url")
            else:
                try:
                    with st.spinner("Waiting..."):
                        if "youtube.com" in generic_url:
                            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                        else:
                            loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                        docs = loader.load()

                        chain = load_summarize_chain(llm, chain_type="refine")
                        output_summary = chain.run(docs)

                        st.success(output_summary)

                except Exception as e:
                    st.exception(f"Exception: {e}")

    elif options == "PDF":
        generic_pdf = st.file_uploader("Upload your pdf: ", type=["pdf"])
        if st.button("Summarize the Content: "):
            if generic_pdf is None:
                st.error("Please upload the valid pdf to get started.")
            else:
                try:
                    with st.spinner("Waiting..."):
                        temppdf = f"./temp.pdf"
                        with open(temppdf, "wb") as file:
                            file.write(generic_pdf.getvalue())
                            file_name = generic_pdf.name

                        loader = PyPDFLoader(temppdf)

                        docs = loader.load()

                        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
                        documents = splitter.split_documents(docs)

                        chain = load_summarize_chain(llm=llm, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=final_prompt_template, verbose=False)
                        output_summary = chain.run(documents)

                        st.success(output_summary)

                except Exception as e:
                    st.exception(f"Exception: {e}")

else:
    st.error("Please Enter your Groq API Key.")