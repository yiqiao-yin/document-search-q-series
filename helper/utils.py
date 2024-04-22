import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import PyPDF2
from openai import OpenAI


# Credit
def current_year():
    now = datetime.now()
    return now.year


# def read_and_textify(
#     files: List[str],
# ) -> Tuple[List[str], List[str]]:
#     """
#     Reads PDF files and extracts text from each page.

#     This function iterates over a list of uploaded PDF files, extracts text from each page,
#     and compiles a list of texts and corresponding source information.

#     Args:
#     files (List[st.uploaded_file_manager.UploadedFile]): A list of uploaded PDF files.

#     Returns:
#     Tuple[List[str], List[str]]: A tuple containing two lists:
#         1. A list of strings, where each string is the text extracted from a PDF page.
#         2. A list of strings indicating the source of each text (file name and page number).
#     """

#     # Initialize lists to store extracted texts and their sources
#     text_list = []  # List to store extracted text
#     sources_list = []  # List to store source information

#     # Iterate over each file
#     for file in files:
#         pdfReader = PyPDF2.PdfReader(file)  # Create a PDF reader object
#         # Iterate over each page in the PDF
#         for i in range(len(pdfReader.pages)):
#             pageObj = pdfReader.pages[i]  # Get the page object
#             text = pageObj.extract_text()  # Extract text from the page
#             pageObj.clear()  # Clear the page object (optional, for memory management)
#             text_list.append(text)  # Add extracted text to the list
#             # Create a source identifier and add it to the list
#             sources_list.append(file.name + "_page_" + str(i))

#     # Return the lists of texts and sources
#     return [text_list, sources_list]


def read_and_textify(
    files: List[str], chunk_size: int = 2  # Default chunk size set to 50
) -> Tuple[List[str], List[str]]:
    """
    Reads PDF files and extracts text from each page, breaking the text into specified segments.

    This function iterates over a list of uploaded PDF files, extracts text from each page,
    and compiles a list of texts and corresponding source information, segmented into smaller parts
    of approximately 'chunk_size' words each.

    Args:
    files (List[st.uploaded_file_manager.UploadedFile]): A list of uploaded PDF files.
    chunk_size (int): The number of words per text segment. Default is 50.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing two lists:
        1. A list of strings, where each string is a segment of text extracted from a PDF page.
        2. A list of strings indicating the source of each text segment (file name, page number, and segment number).
    """

    text_list = []  # List to store extracted text segments
    sources_list = []  # List to store source information

    # Iterate over each file
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)  # Create a PDF reader object
        # Iterate over each page in the PDF
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]  # Get the page object
            text = pageObj.extract_text()  # Extract text from the page
            if text:
                # Split text into chunks of approximately 'chunk_size' words
                words = text.split(". ")
                for j in range(0, len(words), chunk_size):
                    chunk = ". ".join(words[j : j + chunk_size]) + "."
                    text_list.append(chunk)
                    # Create a source identifier for each chunk and add it to the list
                    sources_list.append(f"{file.name}_page_{i}_chunk_{j // chunk_size}")
            else:
                # If no text extracted, still add a placeholder
                text_list.append("")
                sources_list.append(f"{file.name}_page_{i}_chunk_0")
            pageObj.clear()  # Clear the page object (optional, for memory management)

    return text_list, sources_list


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def list_to_nums(sentences: List[str]) -> List[List[float]]:
    """
    Converts a list of sentences into a list of numerical embeddings using OpenAI's embedding model.

    Args:
    - sentences (List[str]): A list of sentences (strings).

    Returns:
    - List[List[float]]: A list of lists of numerical embeddings.
    """

    # Initialize the list to store embeddings
    embeddings = []

    # Loop through each sentence to convert to embeddings
    for sentence in sentences:
        # Use the OpenAI API to get embeddings for the sentence

        response = client.embeddings.create(
            input=sentence, model="text-embedding-3-small"
        )

        embeddings.append(response.data[0].embedding)

    return embeddings


def call_gpt(prompt: str, content: str) -> str:
    """
    Sends a structured conversation context including a system prompt, user prompt,
    and additional background content to the GPT-3.5-turbo model for a response.

    This function is responsible for generating an AI-powered response by interacting
    with the OpenAI API. It puts together a preset system message, a formatted user query,
    and additional background information before requesting the completion from the model.

    Args:
        prompt (str): The main question or topic that the user wants to address.
        content (str): Additional background information or details relevant to the prompt.

    Returns:
        str: The generated response from the GPT model based on the given prompts and content.

    Note: 'client' is assumed to be an already created and authenticated instance of the OpenAI
          client, which should be set up prior to calling this function.
    """

    # Generates a response from the model based on the interactive messages provided
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # The AI model being queried for a response
        messages=[
            # System message defining the assistant's role
            {"role": "system", "content": "You are a helpful assistant."},
            # User message containing the prompt
            {"role": "user", "content": f"I want to ask you a question: {prompt}"},
            # Assistant message asking for background content
            {"role": "assistant", "content": "What is the background content?"},
            # User providing the background content
            {"role": "user", "content": content},
        ],
    )

    # Extracts and returns the response content from the model's completion
    return response.choices[0].message.content


def quantize_to_kbit(arr: Union[np.ndarray, Any], k: int = 16) -> np.ndarray:
    """Converts an array to a k-bit representation by normalizing and scaling its values.

    Args:
        arr (Union[np.ndarray, Any]): The input array to be quantized.
        k (int): The number of levels to quantize to. Defaults to 16 for 4-bit quantization.
    Returns:
        np.ndarray: The quantized array with values scaled to 0 to k-1.
    """
    if not isinstance(arr, np.ndarray):  # Check if input is not a numpy array
        arr = np.array(arr)  # Convert input to a numpy array
    arr_min = arr.min()  # Calculate the minimum value in the array
    arr_max = arr.max()  # Calculate the maximum value in the array
    normalized_arr = (arr - arr_min) / (
        arr_max - arr_min
    )  # Normalize array values to [0, 1]
    return np.round(normalized_arr * (k - 1)).astype(
        int
    )  # Scale normalized values to 0-(k-1) and convert to integer


def quantized_influence(
    arr1: np.ndarray, arr2: np.ndarray, k: int = 16, use_dagger: bool = False
) -> Tuple[float, List[float]]:
    """
    Calculates a weighted measure of influence based on quantized version of input arrays and optionally applies a transformation.

    Args:
        arr1 (np.ndarray): First input array to be quantized and analyzed.
        arr2 (np.ndarray): Second input array to be quantized and used for influence measurement.
        k (int): The quantization level, defaults to 16 for 4-bit quantization.
        use_dagger (bool): Flag to apply a transformation based on local averages, defaults to False.
    Returns:
        Tuple[float, List[float]]: A tuple containing the quantized influence measure and an optional list of transformed values based on local estimates.
    """
    # Quantize both arrays to k levels
    arr1_quantized = quantize_to_kbit(arr1, k)
    arr2_quantized = quantize_to_kbit(arr2, k)

    # Find unique quantized values in arr1
    unique_values = np.unique(arr1_quantized)

    # Compute the global average of quantized arr2
    total_samples = len(arr2_quantized)
    y_bar_global = np.mean(arr2_quantized)

    # Compute weighted local averages and normalize
    weighted_local_averages = [
        (np.mean(arr2_quantized[arr1_quantized == val]) - y_bar_global) ** 2
        * len(arr2_quantized[arr1_quantized == val]) ** 2
        for val in unique_values
    ]
    qim = np.sum(weighted_local_averages) / (
        total_samples * np.std(arr2_quantized)
    )  # Calculate the quantized influence measure

    if use_dagger:
        # If use_dagger is True, compute local estimates and map them to unique quantized values
        local_estimates = [
            np.mean(arr2_quantized[arr1_quantized == val]) for val in unique_values
        ]
        daggers = {
            unique_values[i]: v for i, v in enumerate(local_estimates)
        }  # Map unique values to local estimates

        def find_val_(i: int) -> float:
            """Helper function to map quantized values to their local estimates."""
            return daggers[i]

        # Apply transformation based on local estimates
        daggered_values = list(map(find_val_, arr1_quantized))
        return qim, daggered_values
    else:
        # If use_dagger is False, return the original quantized arr1 values
        daggered_values = arr1_quantized.tolist()
        return qim


def query_search(
    prompt: str,
    sentences: list[str],
    query_database: list[list[float]],
    sources: list[str],
    levels: int,
) -> pd.DataFrame:
    """
    Takes a text prompt and searches a predefined database by converting the prompt
    and database entries to embeddings, and then calculating a quantized influence metric.

    Args:
    - prompt (str): A text prompt to search for in the database.

    Returns:
    - pd.DataFrame: A pandas DataFrame sorted by the quantized influence metric in descending order.
                     The DataFrame contains the original sentences, their embeddings, and the computed scores.
    """
    # Convert the prompt to its numerical embedding
    prompt_embed_ = list_to_nums([prompt])

    # Calculate scores for each item in the database using the quantized influence metric
    scores = [
        [
            sentences[i],  # The sentence itself
            # query_database[i],  # Embedding of the sentence
            sources[i],  # Source of the sentence
            quantized_influence(
                prompt_embed_[0], query_database[i], k=levels, use_dagger=False
            ),  # Score calculation
        ]
        for i in range(len(query_database))
    ]

    # Convert the list of scores into a DataFrame
    refs = pd.DataFrame(scores)
    # Rename columns for clarity
    refs = refs.rename(
        # columns={0: "sentences", 1: "query_embeddings", 2: "page no", 3: "qim"}
        columns={0: "sentences", 1: "page no", 2: "qim"}
    )
    # Sort the DataFrame based on the 'qim' score in descending order
    refs = refs.sort_values(by="qim", ascending=False)

    return refs
