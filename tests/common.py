import os


def read_hugging_face_token() -> str:
    """
    Reads the Hugging Face token from the environment variables.

    Returns:
        str: The Hugging Face token.
    """
    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        raise ValueError("Hugging Face token not found in environment variables.")

    return hf_token
