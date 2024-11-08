import time
from typing import Dict, Union
import os
import openai
import tiktoken


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
        if "o1" in model:
            config = {
            "model": model,
            # "max_completion_tokens": max_tokens,
            "n": batch_size,
            "messages": [system_message + "\n\n" + message],
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }

        if "o1" in model:
            config = {
            "model": model,
            # "max_completion_tokens": max_tokens,
            "n": batch_size,
            "messages": [
                {"role": "user", "content": system_message + "\n\n" + message},
            ],
            }
    

    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    model = config.get("model")

    if model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        base_url = "http://localhost:6789/v1"
    elif model == "meta-llama/Meta-Llama-3.1-70B-Instruct":
        base_url = "http://localhost:6778/v1"
    else:
        base_url = None

    client = openai.OpenAI(base_url=base_url)

    if model == "Meta-Llama-3-1-70B-Instruct-htzs":
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint = "https://api-ai-sandbox.princeton.edu",   
            api_version="2024-02-01" 
        )
    elif model == "Meta-Llama-3-1-8B-Instruct-nwxcg":
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint = "https://api-ai-sandbox.princeton.edu",   
            api_version="2024-02-01" 
        )


    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            if "o1" in model:
                batch_size = config.get("n", 1)
                # remove n from config
                config.pop("n")
                outputs = []
                for _ in range(batch_size):
                    ret = client.chat.completions.create(**config)
                    outputs.append(ret)
                
                # merge choices
                choices = [output.choices[0] for output in outputs]
                ret.choices = choices
            else:
                ret = client.chat.completions.create(**config)
            


        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    prefill_message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": prefill_message},
            ],
        }
    return config


def request_anthropic_engine(client, config, logger, max_retries=40, timeout=100):
    ret = None
    retries = 0

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10)
        retries += 1

    return ret
