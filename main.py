import pandas as pd
import streamlit as st
import numpy as np
import summarize
import time
import chunks
from models.vertexai_model import TextBisonModel
from models.openai_model import LangChainModel

def run(url, count, model, user_prompt, chunk_size, chunk_overlap):
    progress = st.progress(0, "Preparing custom summary. Please wait")
    def progressBar(percent):
        progress.progress(percent, "Loading...")

    start_time = time.time()
    summaries = summarize.custom_summary(url, count, progressBar, model, user_prompt, chunk_size, chunk_overlap)
    duration = time.time() - start_time

    progress.empty()
    df = pd.DataFrame.from_dict(summaries)
    df.index = np.arange(1, len(df) + 1)
    st.table(df)
    st.write(f"Summary generated in {duration:.2f} seconds.")

    st.button("Another one!", on_click=main)


def main():
    st.set_page_config(layout="wide")
    st.title("Summarization App")
    llmOption = st.sidebar.selectbox("LLM",["OpenAI", "VertexAI", "Other (open source in the future)"])
    chunk_size = st.sidebar.slider("Chunk Size", min_value=20, max_value = 10000,
                                    step=10, value=2000)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=5, max_value = 5000,
                                    step=10, value=200)

    if st.sidebar.checkbox("Debug chunk size"):
        st.header("Interactive Text Chunk Visualization")

        tempFile = open("chunk_visualizer.txt","r+")
        text_input = st.text_area(tempFile.read())

        html_code = chunks.color_chunks(text_input, chunk_size, chunk_overlap)
        st.markdown(html_code, unsafe_allow_html=True)            

    else:
        user_prompt = st.text_input("Enter the custom summary prompt")
        url = st.text_input('Enter the RSS Feed URL')
        temperature = st.sidebar.number_input("Set the model Temperature (Only for OpenAI)",
                                                min_value = 0.0,
                                                max_value=1.0,
                                                step=0.1,
                                                value=0.5)
        count = st.sidebar.number_input("Number of articles to parse",
                                                min_value = 1, 
                                                max_value = 10,
                                                step = 1,
                                                value = 3)
        
        if (llmOption == 'OpenAI'):
            model = LangChainModel()
        elif (llmOption == 'VertexAI'):
            model = TextBisonModel()
        else:
            st.write("Using OpenAI while open source models are not implemented!")  
            model = LangChainModel()  

        if st.button('Summarize'):
            run(url, count, model, user_prompt, chunk_size, chunk_overlap) 


if __name__=="__main__":
    main()