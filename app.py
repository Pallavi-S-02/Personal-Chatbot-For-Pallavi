__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from chatbot import get_base64
from chatbot import load_pdf_file, create_chunks_and_embeddings, load_llm, create_prompt_template

def main():
    st.set_page_config(page_title="Chat with Pallavi")
    st.header("*:White[Feel free to know more about Pallavi]*")
    st.markdown("<h5 style='color: #0000CD;'>You can ask me a question and my personal AI will answer you</h5>", unsafe_allow_html=True)
    
    user_question = st.text_input("Type your question below", "")
    
    def set_background(png_file):
        bin_str = get_base64(png_file)
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.8); /* Optional: Add a semi-transparent white background to make text more readable */
        }
        .stMarkdown {
            color: black; /* Set default text color to black */
            font-weight: bold; /* Make text bold */
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)
    
    set_background('image.jfif')
    
    #for input text box
    st.markdown(
    """
    <style>
    /* Change the border color and box shadow color */
    input[type="text"] {
        border: 1px solid #FFB6C1 !important;
        box-shadow: 0 0 0 1px #FFB6C1 !important;
        background-color: #FFB6C1 !important; /* Baby pink color as fallback */
        color: black !important; /* Text color */
    }

    /* Change the text color of the placeholder */
    input[type="text"]::placeholder {
        color: #FFB6C1 !important; /* Placeholder text color */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    #for response to be in box
    st.markdown(
        """
        <style>
        .response {
            color: black;
            background-color: #E6E6FA;
            padding: 10px;
            border-radius: 5px;
            margin-top: -40px; /* Negative margin to pull the box up */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if user_question:
        # Wrap the entire app content in a div
        with st.markdown("<div>", unsafe_allow_html=True):
            pdf_data = load_pdf_file('bio.pdf')
            Ensemble_Retriever = create_chunks_and_embeddings(pdf_data) 
            llm_model = load_llm()
            chain = create_prompt_template(Ensemble_Retriever, llm_model)
            response = chain.invoke(user_question)
            response_with_prefix = f"<span style='font-weight:bold'>Answer:</span> <span style='font-weight:normal'>{response}</span>"
            st.write(f"<div class='response'>{response_with_prefix}</div>", unsafe_allow_html=True)

    st.markdown(
    """
    <br><br><br>
    <div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: #FFFAFA; text-align: center; padding: 10px;">
        Pallavi Sindkar <br>
        <a href="https://www.linkedin.com/in/pallavi-sindkar-83b583203" target="_blank">LinkedIn</a> |
        <a href="https://github.com/Pallavi-S-02" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
    )

    # Close the outer div
    st.markdown("</div>", unsafe_allow_html=True)

main()


