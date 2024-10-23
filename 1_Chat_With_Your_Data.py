import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Allows you to interact with OpenAI's language models
from langchain_openai import ChatOpenAI
# Provide Tool for LLM
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# Logging is essential in both development and production environments for tracking events and diagnosing issues.
from src.logger.base import BaseLogger
from src.models.llms import load_llm
from src.utils import execute_plt_code


from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
# Now you can access the API key
api_key = os.getenv("OPENAI_API_KEY")



# Logger initialization
logger = BaseLogger()
MODEL_NAME = "GPT-3.5-turbo"

def process_query(da_agent, query):
    response = da_agent(query)
    action = response["intermediate_steps"][-1][0].tool_input['query']
    
    if "plt" in action:
        st.write(response["output"])
        fig = execute_plt_code(action, df=st.session_state.df)
        if fig:
            st.pyplot(fig)

        st.write("**Executed code:**")
        st.code(action)

        to_display_string = response["output"] + "\n" + f"```python\n{action}\n```"
        st.session_state.history.append((query, to_display_string))
    else:
        st.write(response["output"])
        st.session_state.history.append((query, response["output"]))

def display_chat_history():
    st.markdown('## Chat History:')
    for i, (q, r) in enumerate(st.session_state.history):
        st.markdown(f"**Query {i + 1}:** {q}")
        st.markdown(f"**Response {i + 1}:** {r}")
        st.markdown("---")

def load_llm(model_name):
    """
    Load Large Language Model.
    
    Args:
        model_name (str): The name of the model to load.
    
    Raises:
        ValueError: If an unknown model name is provided.
    
    Returns:
        ChatOpenAI: The initialized language model.
    """
    model_name = model_name.lower()  # Normalize model name to lowercase
    
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1000
        )
    elif model_name == "gemini-pro":
        pass  # Implement this case when required
    else:
        raise ValueError(
            "Unknown model. Please choose from ['gpt-3.5-turbo', 'gpt-4',...]"
        )

def main():
    # Setting up Streamlit interface
    st.set_page_config(page_title="📊 Smart Data Analysis Tool", page_icon="📊", layout="wide")
    st.header("📊 Smart Data Analysis Tool")
    st.write("### Welcome to our data analysis tool. This tool can assist with your daily data analysis tasks.")

    # Loading language model and setting up OpenAI API key
    llm = load_llm(model_name=MODEL_NAME)
    logger.info(f"### Successfully loaded {MODEL_NAME}! ###")

    # Implementing file upload feature
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Creating chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Read csv file
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("### Your uploaded data: ", st.session_state.df.head())
    
        # Create data analysis agent to query with our data
        da_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state.df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            verbose=True,
            return_intermediate_steps=True,
        )
        logger.info("### Successfully loaded data analysis agent! ###")

        # Input query and processing user queries
        query = st.text_input("Enter your question:")

        if st.button("Run Query"):
            with st.spinner("Processing..."):
                process_query(da_agent, query)

    # Displaying results and visualizations
    st.divider()
    display_chat_history()

if __name__ == "__main__":
    main()
