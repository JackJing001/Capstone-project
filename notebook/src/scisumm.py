"""
Data extraction for SciSumm dataset, used for the summarisation step
Place data from https://github.com/WING-NUS/scisumm-corpus/tree/master/data/Training-Set-2019/Task2/From-ScisummNet-2019
into the DATA_PATH

"""

import json, os, re, itertools
from glob import glob
from tqdm import tqdm

import xml.etree.ElementTree as ET
import pandas as pd

from src.data_utils import (
    CONCLUSIONS_LIST,
    INTRO_LIST,
    ACK_LIST,
    compile_search,
    get_wordcount,
)


class SciSumm:
    """Parsing functions for SciSumm dataset"""

    BASE_PATH = "../../data/"
    DIRECTORY = "From-ScisummNet-2019"
    DATA_PATH = os.path.join(BASE_PATH, DIRECTORY)
    REF_FOLDER = "Reference_XML"
    SUMMARY_FOLDER = "summary"
    ABSTRACT_TEXT_SUFFIX = ".abstract.txt"
    HUMAN_TEXT_SUFFIX = ".scisummnet_human.txt"

    conclusions_search = compile_search(CONCLUSIONS_LIST)
    intro_search = compile_search(INTRO_LIST)
    ack_search = compile_search(ACK_LIST)

    min_words = 20

    def __init__(self, data_path=None):
        """Initialise with path to data folder"""
        self.data_path = data_path or self.DATA_PATH
        self._paper_paths = None

    @property
    def paper_paths(self):
        """Get all paper id paths"""
        if self._paper_paths is None:
            self._paper_paths = glob(os.path.join(self.data_path, "*"))
            print(f"{len(self._paper_paths)} papers found")
        return self._paper_paths

    def get_xml_path(self, paper_path):
        """Path to paper xml """
        return glob(os.path.join(paper_path, self.REF_FOLDER, "*"))[0]

    def get_human_summary_path(self, paper_path):
        """Path to human summary file"""
        return glob(
            os.path.join(
                paper_path, self.SUMMARY_FOLDER, "*" + self.HUMAN_TEXT_SUFFIX
            )
        )[0]

    def get_abstract_path(self, paper_path):
        """Path to abstract file"""
        return glob(
            os.path.join(
                paper_path,
                self.SUMMARY_FOLDER,
                "*" + self.ABSTRACT_TEXT_SUFFIX,
            )
        )[0]

    def get_root(self, xml_path):
        """Return parsed XML root"""
        return ET.parse(xml_path).getroot()

    def search_section(self, root, start_search, stop_search=None):
        """Identify start and end indexes for specific sections"""
        extract_on = False
        extract_off = False
        section = []
        for i in root.iter():
            # Start extracting text as soon as we encounter a section with
            # the title that matches the start_search
            temp = i.attrib.get("title")
            if temp and (start_search.search(temp)):
                extract_on = True

            # Aim to isolate conclusion section
            else:
                if extract_on:
                    # Disable extraction at the next title section
                    if temp:
                        extract_off = True

                    # Disable extraction if we encounter the stop_search section
                    if stop_search:
                        if stop_search.search(i.text):
                            extract_off = True

            if extract_on and not extract_off:
                if i.text:
                    # Remove any blanks
                    text = i.text.strip()
                    if text is not "":
                        section.append(text)

        return section

    def parse_xml(self, xml_path):
        """Extract full text from xml"""
        root = self.get_root(xml_path)
        raw_text = [elem.text for elem in root.iter() if elem.text is not None]
        clean_text = [
            text.strip() for text in raw_text if text.strip() is not ""
        ]
        title, full_text = clean_text[0], clean_text[1:]
        conclusions = self.search_section(
            root, self.conclusions_search, self.ack_search
        )
        introduction = self.search_section(root, self.intro_search)
        return title, full_text, introduction, conclusions

    def parse_txt(self, txt_path, remove_title=False):
        """Load txt files"""
        data = []
        with open(txt_path, "r") as f:
            for line in f.readlines():
                data.append(line.replace("\n", ""))

        # Remove title as the first line
        if remove_title:
            data = data[1:]
        return data

    def flag_formatting(self, text):
        """Flag any strange formatting for removal"""

        # Some papers seem to have text with all of the periods removed
        number_periods = " ".join(text).count(".")
        number_sentences = len(text)
        if number_periods < number_sentences:
            return 1
        return 0

    def flag_text(self, text):
        """Flag papers where text is missing or too short"""
        wordcount = get_wordcount(text)
        if wordcount < self.min_words:
            return 1
        return 0

    def get_data(self, paper_path):
        """Generate data for each paper"""
        paper_id = paper_path.split("/")[-1]
        xml_path = self.get_xml_path(paper_path)
        title, full_text, introduction, conclusions = self.parse_xml(xml_path)

        hs_path = self.get_human_summary_path(paper_path)
        hs = self.parse_txt(hs_path, remove_title=True)

        abstract_path = self.get_abstract_path(paper_path)
        abstract = self.parse_txt(abstract_path, remove_title=False)

        return (
            paper_id,
            title,
            full_text,
            introduction,
            conclusions,
            abstract,
            hs,
            self.flag_formatting(full_text),
            self.flag_text(hs),
            self.flag_text(introduction),
            self.flag_text(conclusions),
        )

    def __call__(self):
        """Create dataset"""
        dataset = [self.get_data(path) for path in self.paper_paths]
        output = pd.DataFrame(dataset)
        output.columns = [
            "paperid",
            "title",
            "full_text",
            "introduction",
            "conclusions",
            "abstract",
            "human_summary",
            "flag_formatting",
            "flag_hs",
            "flag_intro",
            "flag_conclusions",
        ]
        return output

    def clean(self, output):
        # Drop rows that are flagged as strange
        print("")
        print(f"{len(output)} papers in the raw dataset")
        print(
            f"    {output.flag_formatting.sum()} papers with corrupted formatting  "
            f"removed"
        )
        output = output[output.flag_formatting == 0]

        print(
            f"    {output.flag_hs.sum()} papers with human summary "
            f"shorter than {self.min_words} words removed"
        )
        output = output[output.flag_hs == 0]

        print(
            f"    {output.flag_conclusions.sum()} papers with conclusions "
            f"shorter than {self.min_words} words removed"
        )
        output = output[output.flag_conclusions == 0]

        print(
            f"    {output.flag_intro.sum()} papers introductions shorter than "
            f"{self.min_words} words removed"
        )
        output = (
            output[output.flag_intro == 0]
            .reset_index()
            .drop(columns=["index"])
        )

        print(f"{len(output)} papers in the final cleaned dataset")
        return output[
            [
                "paperid",
                "title",
                "full_text",
                "introduction",
                "conclusions",
                "abstract",
                "human_summary",
            ]
        ]
