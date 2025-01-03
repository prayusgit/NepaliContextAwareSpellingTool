import streamlit as st
import paragraph  # Import paragraph.py
import sentence   # Import sentence.py

def main():
    mode = st.sidebar.selectbox("Select Mode", ("Sentence", "Paragraph", "Speech-to-text"))
    # if mode == "Sentence":
    #     st.write("### Paragraph Mode")
    #     sentence.main()  # Call the paragraph mode function

    if mode == "Paragraph":
        st.write("### Paragraph Mode")
        paragraph.main()  # Call the paragraph mode function

    # elif mode == "Speech-to-text":
    #     st.write("### Speech To Text Mode")
    #

if __name__ == "__main__":
    main()
