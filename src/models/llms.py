from langchain_openai import ChatOpenAI

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
    api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from environment variable
    
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1000,
            openai_api_key=api_key  # Pass the API key here
        )
    elif model_name == "gemini-pro":
        pass  # Implement this case when required
    else:
        raise ValueError(
            "Unknown model. Please choose from ['gpt-3.5-turbo', 'gpt-4',...]"
        )





# from langchain_openai import ChatOpenAI

# def load_llm(model_name):
#     """
#     Load Large Language Model.
    
#     Args:
#         model_name (str): The name of the model to load.
    
#     Raises:
#         ValueError: If an unknown model name is provided.
    
#     Returns:
#         ChatOpenAI: The initialized language model.
#     """
#     model_name = model_name.lower()  # Normalize model name to lowercase
    
#     if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
#         return ChatOpenAI(
#             model=model_name,
#             temperature=0.0,
#             max_tokens=1000
#         )
#     elif model_name == "gemini-pro":
#         pass  # Implement this case when required
#     else:
#         raise ValueError(
#             "Unknown model. Please choose from ['gpt-3.5-turbo', 'gpt-4',...]"
#         )




# # from langchain_openai import ChatOpenAI

# # def load_llm(model_name):
# #     """
# #     Load Large Language Model.
    
# #     Args:
# #         model_name (str): The name of the model to load.
    
# #     Raises:
# #         ValueError: If an unknown model name is provided.
    
# #     Returns:
# #         ChatOpenAI: The initialized language model.
# #     """
# #     if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
# #         return ChatOpenAI(
# #             model=model_name,
# #             temperature=0.0,
# #             max_tokens=1000
# #         )
# #     elif model_name == "gemini-pro":
# #         pass  # Implement this case when required
# #     else:
# #         raise ValueError(
# #             "Unknown model. Please choose from ['gpt-3.5-turbo', 'gpt-4',...]"
# #         )







# # from langchain_openai import ChatOpenAI

# # def load_llm(model_name):
# #     """
# #     Load Large Language Model.
    
# #     Args:
# #         model_name (_type_): _description_
    
# #     Raises:
# #         ValueError: _description_
    
# #     Returns:
# #         _type_: _description_
# #     """
# #     if model_name=="gpt-3.5-turbo":
# #         return ChatOpenAI(
# #             model=model_name
# #             temperature=0.0,
# #             max_tokens=1000
# #         )
# #     elif model_name=="gpt-4":
# #         return ChatOpenAI(
# #             model=model_name
# #             temperature=0.0,
# #             max_tokens=1000
# #         )
# #     elif model_name=="gemini-pro":
# #         pass
# #     else:
# #         raise ValueError(
# #               "Unknown model.\
# #                 Please choose from ['gpt3.5-turbo', 'gpt-4',...]"
# #         )