"""
Utility functions for data cleaning

"""
from typing import Union
import re
import pandas as pd
import numpy as np

CONCLUSIONS_LIST = ["conclusion*", "discussion*"]
INTRO_LIST = ["introdu*"]
RW_LIST = ["related work*"]
ACK_LIST = ["acknow*", "thank*"]


def parse_list_to_string(text: Union[str, list]) -> str:
    """Convert lists of strings to a single string"""
    if isinstance(text, list):
        return " ".join(text)
    return text


def get_wordcount(text: Union[list, str]) -> int:
    """Rough wordcount based on assuming each word is separated by a space"""
    text = parse_list_to_string(text)
    return len(text.split(" "))


def wordcount_stats(df: pd.DataFrame):
    """Print out average wordcount per column in a dataset"""
    for col in df.columns:
        wordcounts = df[col].apply(get_wordcount)
        print(f"\nColumn: {col}")
        print(f"Mean wordcount: {np.mean(wordcounts)}")
        print(f"Median: {np.median(wordcounts)}")
        print(f"Min: {np.min(wordcounts)}")
        print(f"Max: {np.max(wordcounts)}")


def compile_search(*args):
    """Compile regex search from multiple list inputs"""
    full_list = []
    for arg in args:
        if isinstance(arg, str):
            arg = [arg]
        full_list += arg

    compiled_search = re.compile("|".join(full_list), re.IGNORECASE)
    return compiled_search
