import streamlit as st
import paragraph  # Import paragraph.py
import sentence   # Import sentence.py

def main():
    mode = st.sidebar.selectbox("Select Mode", ("Paragraph"))

    if mode == "Paragraph":
        st.write("### Paragraph Mode")
        paragraph.main()  # Call the paragraph mode function

if __name__ == "__main__":
    main()
