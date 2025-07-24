import tiktoken
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Any

# Borrowed from : https://github.com/openai/whisper
LANGUAGES = {
    "en": "english",
    "zh-hans": "simplified chinese",
    "zh": "simplified chinese",
    "zh-hant": "traditional chinese",
    "zh-yue": "cantonese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}


def prompt_config_to_kwargs(prompt_config):
    prompt_config = prompt_config or {}
    return dict(
        prompt_template=prompt_config.get("user", None),
        prompt_sys_msg=prompt_config.get("system", None),
    )


# ref: https://platform.openai.com/docs/guides/chat/introduction
def num_tokens_from_text(text, model="gpt-4.1"):
    messages = [
        {
            "role": "user",
            "content": text,
        },
    ]

    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to cl100k_base encoding for newer models including gpt-4.1
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Updated token counting for modern OpenAI models (gpt-3.5-turbo, gpt-4, gpt-4.1, etc.)
    if model in ["gpt-3.5-turbo-0301"]:
        # Legacy model handling
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        # Modern models (gpt-3.5-turbo, gpt-4, gpt-4.1, etc.)
        num_tokens = 0
        for message in messages:
            num_tokens += 3  # every message follows <|start|>{role/name}\n{content}<|end|\>\n
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 1  # role is always required and always 1 token
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens


def process_concurrently(items: List[Any], func: Callable, max_workers: int = 8) -> List[Any]:
    """
    Process a list of items concurrently using ThreadPoolExecutor.
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        max_workers: Maximum number of concurrent workers (default: 8)
    
    Returns:
        List of results in the same order as input items
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    return results


async def process_concurrently_async(items: List[Any], async_func: Callable, max_concurrent: int = 8) -> List[Any]:
    """
    Process a list of items concurrently using asyncio semaphore.
    
    Args:
        items: List of items to process
        async_func: Async function to apply to each item
        max_concurrent: Maximum number of concurrent operations (default: 8)
    
    Returns:
        List of results in the same order as input items
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_func(item):
        async with semaphore:
            return await async_func(item)
    
    tasks = [limited_func(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
