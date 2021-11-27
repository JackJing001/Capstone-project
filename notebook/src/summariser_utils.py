"""
Classes for text summarisation


"""
import os
from typing import Union
import numpy as np
import pandas as pd
from rouge import Rouge
from summarizer import Summarizer
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BartTokenizer,
    BartForConditionalGeneration,
    LongformerTokenizer,
    LongformerModel,
    AutoConfig,
    AutoModel,
)


# Update for your Colab setup
COLAB_BASE_PATH = "/content/drive/MyDrive/Data Science Capstone files/"


class BaseSummariser:
    """Base Summarisers with utilities for generic summarisation tasks"""

    # Insert specific model parameters
    MODEL_PATH = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

    token_limit = 1024
    max_length = 300

    rouge = Rouge()
    rouges = ["rouge-1", "rouge-2", "rouge-l"]

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def batch_sentences(self, text: list) -> list:
        """Create batches of sentences within the token limit"""
        batches = []
        sent = []
        length = 0
        for sentence in text:

            length += len(sentence)
            if length < self.token_limit:
                sent.append(sentence)
            else:
                batches.append(sent)
                sent = [sentence]
                length = len(sentence)
        if sent:
            batches.append(sent)

        return batches

    def generate_summary(self, batches: list) -> list:
        """Generate summary. https://github.com/huggingface/transformers/issues/4224 """
        summaries = []
        device = "cuda"

        for batch in batches:
            if batch == []:
                pass
            else:
                inputs = self.tokenizer.encode(
                    " ".join(batch), truncation=True, return_tensors="pt"
                )
                inputs = inputs.to(device)
                summary_ids = self.model.to(device).generate(
                    inputs,
                    length_penalty=3.0,
                    min_length=30,
                    max_length=self.max_length,
                    early_stopping=True,
                )
                output = [
                    self.tokenizer.decode(
                        g,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for g in summary_ids
                ]
                summaries.append(output)
        summaries = [sentence for sublist in summaries for sentence in sublist]
        return summaries

    @staticmethod
    def flatten_batches(batches):
        """Flatten batches into a single list"""
        return [word for sent in batches for word in sent]

    def get_rouge(self, text_1: Union[str, list], text_2: Union[str, list]):
        """Get rouge score for two text passages"""
        input_1 = parse_list_to_string(text_1)
        input_2 = parse_list_to_string(text_2)

        return self.rouge.get_scores(input_1, input_2)[0]

    def get_rouge_f(self, text_1: Union[str, list], text_2: Union[str, list]):
        """Extract only the f metrics from ROUGE scores"""
        scores = self.get_rouge(text_1, text_2)
        return [scores[rouge]["f"] for rouge in self.rouges]

    @staticmethod
    def format_scores(score: float):
        """Format ROUGE scores to 2 dp """
        return float(f"{score * 100:.2f}")

    def process_row(
        self,
        row: int,
        column: str = None,
        batches: bool = True,
        preprocess: bool = False,
    ):
        """Generate summary and rouge scores for a single row"""
        if column:
            text = self.df.iloc[row][column]
        else:
            text = self.df.iloc[row].full_text

        # Optionally batch the text and feed into the model
        if batches:
            text = self.batch_sentences(text)
        try:
            model_summary = self.generate_summary(text)
        except RuntimeError:
            # Remove last sentence, which seems buggy on one or two examples
            text = self.df.iloc[row][column]
            text = text[:-1]
            if batches:
                text = self.batch_sentences(text)
            model_summary = self.generate_summary(text)

        if preprocess:
            paper_id = self.df.iloc[row].paper_id
            related_works = self.df.iloc[row].related_works
            abstract = self.df.iloc[row].abstract
            return (paper_id, model_summary, abstract, related_works)
        else:
            # Get rouge f scores
            human_summary = self.df.iloc[row].human_summary
            rouge_scores = self.get_rouge_f(human_summary, model_summary)
            rouge_scores = [
                self.format_scores(score) for score in rouge_scores
            ]
            return (
                human_summary,
                model_summary,
                rouge_scores[0],
                rouge_scores[1],
                rouge_scores[2],
            )

    def process_all(
        self,
        column: str = None,
        batches: bool = True,
        print_results: bool = True,
        preprocess: bool = False,
    ):
        """
        Generate summary and rouge scores for all rows in the data

        Args:
            column: Which column to use as model input
            batches: Naively batch the input text into chunks of max_length
            print_results: Print out average rouge scores

        Returns:
            results: dataframe with human summary, model summary and ROUGE scores
        """
        results_list = []
        for i in tqdm(range(len(self.df)), position=0, leave=True):
            results_list.append(
                self.process_row(
                    i, column=column, batches=batches, preprocess=preprocess
                )
            )
        results = self.results_to_df(results_list, preprocess=preprocess)

        if print_results:
            self.printout(results)
        return results

    def results_to_df(self, results_list, preprocess):
        """Format results as a dataframe"""
        if preprocess:
            return pd.DataFrame(
                results_list,
                columns=[
                    "paper_id",
                    "model_summary",
                    "abstract",
                    "related_works",
                ],
            )
        return pd.DataFrame(
            results_list,
            columns=["human_summary", "model_summary", "R1", "R2", "RL"],
        )

    def printout(self, results):
        """Display summary metrics. """
        self.print_mean_rouge(results)
        try:
            display(results.head())
        except:
            print(results.head())

    @staticmethod
    def print_mean_rouge(df: pd.DataFrame):
        """Print mean rouge scores"""
        print("")
        print("Mean rouge scores:")
        print(f"    R1: {np.mean(df.R1)}")
        print(f"    R2: {np.mean(df.R2)}")
        print(f"    RL: {np.mean(df.RL)}")


class Bart(BaseSummariser):
    """Bart using general purpose pretrained model"""

    # Pretrained BART model
    MODEL_PATH = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)


#
# class Long(Bart):
#     """Longformer summariser"""
#
#     # Pretrained Longformer data
#     MODEL_PATH = "allenai/longformer-base-4096"
#     tokenizer = LongformerTokenizer.from_pretrained(MODEL_PATH)
#     model = LongformerModel.from_pretrained(MODEL_PATH)
#
#     # Longformer can accept a higher token limit
#     token_limit = 4096
#     max_length = 100


class SciBERT(BaseSummariser):
    """SciBERT vocab with BERT extractive summariser"""

    # Credit: https://github.com/Nikoschenk/bert-extractive-summarizer
    MODEL_PATH = "allenai/scibert_scivocab_uncased"
    custom_config = AutoConfig.from_pretrained(MODEL_PATH)
    custom_config.output_hidden_states = True
    custom_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    custom_model = AutoModel.from_pretrained(MODEL_PATH, config=custom_config)
    model = Summarizer(
        custom_model=custom_model, custom_tokenizer=custom_tokenizer
    )

    def generate_summary(self, batches: list):
        """Generate summary"""

        if self.target_length:
            ratio = self.calculate_ratio(batches)
            self.ratio_list.append(ratio)

        summaries = []
        for batch in batches:
            text = parse_list_to_string(batch)
            # Adjust compression ratio to output summaries of target_length
            if self.target_length:
                output = self.model(text, ratio=ratio)

            # Otherwise, use default ratio
            else:
                text = text[: self.token_limit]
                output = self.model(text)

            summaries.append(output)
        return summaries

    def calculate_ratio(self, text):
        """Adjust compression ratio to get summaries of approx target length"""
        if self.batches:
            text = self.flatten_batches(text)

        total_length = get_wordcount(text)

        ratio = self.target_length / total_length
        if ratio >= 1.0:
            ratio = 1.0
        return ratio

    def process_all(
        self,
        column: str = None,
        batches: bool = True,
        print_results: bool = True,
        target_length: int = None,
        preprocess: bool = False,
    ):
        """
        Generate summary and rouge scores for all rows in the data

        Args:
            column: Which column to use as model input
            batches: Naively batch the input text into chunks of max_length
            print_results: Print out average rouge scores
            target_length: Target number of words in the summary

        Returns:
            results: dataframe with human summary, model summary and ROUGE scores
        """
        self.batches = batches
        self.target_length = target_length
        self.ratio_list = []

        results_list = []
        for i in tqdm(range(len(self.df)), position=0, leave=True):
            results_list.append(
                self.process_row(
                    i, column=column, batches=batches, preprocess=preprocess
                )
            )
        results = self.results_to_df(results_list, preprocess=preprocess)

        if print_results:
            self.printout(results)

        return results


class SciBERTFinetuned(SciBERT):
    """
    Using a finetuned model from
    https://github.com/Santosh-Gupta/ScientificSummarizationDataSets
    Trained for 30,000 steps on a computer science paper dataset.
    The first 5000 steps were trained on a batch size of 1024,
    and the rest were trained on a batch size of 4096.
    """

    MODEL_PATH = os.path.join(
        COLAB_BASE_PATH,
        "pretrained_models",
        "ScientificSummarizationDataSets.pt",
    )


def parse_list_to_string(text: Union[str, list]) -> str:
    """Convert lists of strings to a single string"""
    if isinstance(text, list):
        return " ".join(text)
    return text


def get_wordcount(text: Union[list, str]) -> int:
    """Rough wordcount based on assuming each word is separated by a space"""
    text = parse_list_to_string(text)
    return len(text.split(" "))
