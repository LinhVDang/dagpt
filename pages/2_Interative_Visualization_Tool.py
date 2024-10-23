import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer


def main():
    # Set up Streamtlit interface
    st.set_page_config(
        layout="wide", 
        page_title="Interactive Visualization Tool", 
        page_icon="📈"
        )
    
    st.header("📈Interactive Visualization Tool")
    # Fix: Changed st.writer to st.write
    st.write("### Welcome to interactive visualization tool, please enjoy!!")
    
    # Render pygwalker
    if st.session_state.get("df") is not None:
        pyg_app = StreamlitRenderer(st.session_state.df)
        pyg_app.explorer()
    else:
        st.info("Please upload a dataset to begin using the interactive visualization tools")    


if __name__ == "__main__":
    main()