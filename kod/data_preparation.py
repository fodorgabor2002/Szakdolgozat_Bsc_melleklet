import os
from typing import Dict, Union, Generator, List, Tuple
from tqdm import tqdm
import statistics as st
import spacy

from doc_data_types import RawDocInfo, WordLists
from doc_data_utils import tokenize_batch
import math
from benepar import Parser
from hunspell import Hunspell
import benepar
from doc_data import create_doc_data

import pandas as pd
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import random
import re


def remove_texts_with_only_one_author(path: str):
    """
    Delete every text file from path that has an author that does not appear anywhere else in the corpus.

    :param path: The path to the txt files.
    """

    confirm = input(f"Tuti biztos ki akarod tisztítani a {path} mappát? Írd, hogy 'igen' ha igen, bármi mást ha nem.\n")
    author_freqs = dict()
    if confirm in {"'igen'", "igen"}:
        for filename in os.listdir(path):
            author = filename.split("_")[0]
            if not filename.endswith(".txt"):
                raise ValueError("Rossz mappa, ebben nem a megfelelő txt-k vannak.")

            if author in author_freqs:
                author_freqs[author] += 1
            else:
                author_freqs[author] = 1
        for filename in os.listdir(path):
            author = filename.split("_")[0]
            if author_freqs[author] == 1:
                os.remove(os.path.join(path, filename))
        print("Törlés sikeres.")
    else:
        print("Nem lett semmi sem törölve.")
        return


def corpus_analysis_simple(path: str) -> Dict[str, Union[int, float]]:
    """
    Perform statistical analysis on the text files found on the path. Only performs simple analysis that does not
    require tokenization. The following metrics are analysed:
    - the number of texts
    - max, min, mean, median, mode, standard_deviation of text char numbers
    - how many different authors there are
    - max, min, mean, median, mode, standard_deviation of the number of texts each author has

    :param path: The path of the corpus.
    :return: A dictionary containing the analysis data.
    """
    print("Performing simple corpus analysis...")

    stats = dict()
    files = os.listdir(path)
    stats["no_of_texts"] = len([file for file in files if file.endswith(".txt")])

    text_lens = []
    no_of_texts_per_author = dict()

    for filename in tqdm(files, total=len(files), desc="Processing files..."):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(path, filename), encoding="UTF-8") as f:
            text = f.read()
        text_lens.append(len(text))
        author = filename.split("_")[0]
        if author in no_of_texts_per_author:
            no_of_texts_per_author[author] += 1
        else:
            no_of_texts_per_author[author] = 1

    stats["max_len"] = max(text_lens)
    stats["min_len"] = min(text_lens)
    stats["mean_len"] = round(st.mean(text_lens))
    stats["median_len"] = round(st.median(text_lens))
    stats["mode_len"] = round(st.mean(st.multimode(text_lens)))
    stats["stdev_len"] = round(st.stdev(text_lens))
    stats["no_of_authors"] = len(no_of_texts_per_author)
    stats["max_author_num"] = max(no_of_texts_per_author.values())
    stats["min_author_num"] = min(no_of_texts_per_author.values())
    stats["mean_author_num"] = round(st.mean(no_of_texts_per_author.values()))
    stats["median_author_num"] = round(st.median(no_of_texts_per_author.values()))
    stats["mode_author_num"] = round(st.mean(st.multimode(no_of_texts_per_author.values())))
    stats["stdev_author_num"] = round(st.stdev(no_of_texts_per_author.values()))

    return stats


def corpus_analysis_tokens(path: str, doc_data_path: str, alternate_doc_data_path: str = "") -> Dict[
    str, Union[int, float]]:
    """
    Perform statistical analysis on the text files found on the path. Only performs analysis that requires tokenization.
    The following metrics are analysed:
    - max, min, mean, median, mode, standard_deviation of text token numbers

    :param path: The path of the corpus.
    :param doc_data_path: The path to the doc_data analysis of the corpus. The DocData effectively contains the
    tokenization info too, which is why it's needed.
    :param alternate_doc_data_path: A folder with extra DocDatas, if needed. Optional.
    :return: A dictionary containing the analysis data.
    """

    print("Performing token-related corpus analysis...")

    stats = dict()
    files = os.listdir(path)
    token_nums = []

    for filename in tqdm(files, total=len(files), desc="Processing files..."):
        if not filename.endswith(".txt"):
            continue

        doc_data_json_path = os.path.join(doc_data_path, filename.replace(".txt", ".json"))
        if not os.path.exists(doc_data_json_path):
            doc_data_json_path = os.path.join(alternate_doc_data_path, filename.replace(".txt", ".json"))

        doc_data = json.load(open(doc_data_json_path, encoding="UTF-8"))
        token_nums.append(doc_data["tokenNum"])

    stats["max_token_num"] = max(token_nums)
    stats["min_token_num"] = min(token_nums)
    stats["mean_token_num"] = round(st.mean(token_nums))
    stats["median_token_num"] = round(st.median(token_nums))
    stats["mode_token_num"] = round(st.mean(st.multimode(token_nums)))
    stats["stdev_token_num"] = round(st.stdev(token_nums))

    return stats


def corpus_analysis_full(path: str, doc_data_path: str, alternate_doc_data_path: str = "") -> Dict[
    str, Union[int, float]]:
    """
    Wrapper function combining the simple and the token based corpus analysis. Writes an Excel of the stats (stats.xlsx)
    in the corpus' folder.

    :param path: The corpus' folder.
    :param doc_data_path: The folder that has all the DocData jsons related to the corpus' folder.
    :param alternate_doc_data_path: A folder with extra DocDatas, if needed. Optional.
    :return: The analysis as a dictionary.
    """

    analysis_simple = corpus_analysis_simple(path)
    analyis_tokens = corpus_analysis_tokens(path, doc_data_path, alternate_doc_data_path=alternate_doc_data_path)
    analysis = analysis_simple | analyis_tokens
    pd.Series(analysis).to_excel(os.path.join(path, "stats.xlsx"), engine="xlsxwriter")
    return analysis


def create_doc_datas_for_whole_corpus(corpus_path: str, output_path: str, model: spacy.Language, benepar_parser: Parser,
                                      token_freq_dict: Dict[str, int], h_spell: Hunspell):
    """
    Create a doc_data for every text in the corpus.

    :param corpus_path: The path of the corpus. A dictionary containing txt files.
    :param output_path: The output path for the docdatas. This dictionary will contain json files, and each filename
    will match the corresponding txt file's name (it will be same, except for the extension).
    :param model: A Spacy language model, required for the doc_data analysis.
    :param benepar_parser: A benepar model, required for the doc_data analyis.
    :param token_freq_dict: A dictionary containing token frequencies, required for the doc_data analyis.
    :param h_spell: A Spelling model, required for the doc_data analyis.
    """
    print("Base corpus is: ", corpus_path)
    print("Output in: ", output_path)

    if len(os.listdir(output_path)) > 0:
        choice = input("OUTPUT PATH IS NOT EMPTY, SURE YOU WANNA CONTINUE? Y/N\n")
        if choice != "Y":
            print("Terminating")
            return

    files = os.listdir(corpus_path)
    choice = input("Would you like to reverse the list? Y/N\n")
    if choice == "Y":
        files.reverse()
        print("The list has been reversed!")
    else:
        print("The list has NOT been changed!")

    for filename in tqdm(files, total=len(files), desc="Creating DocDatas..."):
        # SKIP FILE IF ALREADY DONE
        if os.path.isfile(os.path.join(output_path, filename.replace(".txt", ".json"))):
            continue

        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(corpus_path, filename), encoding="UTF-8") as f:
            text = f.read()
        raw_data = RawDocInfo(text=text, wordlists=WordLists())
        data = create_doc_data(raw_data, model=model, benepar_parser=benepar_parser, token_freq_dict=token_freq_dict,
                               h_spell=h_spell)
        with open(os.path.join(output_path, filename.replace(".txt", ".json")), "w", encoding="UTF-8") as f:
            f.write(data.json())


def features_from_corpus_1_to_4(path: str) -> pd.DataFrame:
    '''
    Create a dataframe in a way that for every author we choose one true pair (two texts with the same author)
    and four false pairs (the other four texts are from other authors). The pairs are distincts from each other.
    
    :param path: the path to the files we are going to build the dataframe from
    '''

    files = os.listdir(path)
    corpus_name = os.path.basename(os.path.normpath(path))
    print("Corpus name: {}".format(corpus_name))
    data = []
    for index, filename in tqdm(enumerate(files), total=len(files), desc="Creating df"):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(path, filename), encoding="UTF-8") as f:
            text1 = f.read()
        author = filename.split("_")[0]
        record = {
            "text": text1,
            "author": author,
            "docname": filename.replace(".txt", ".json"),  # storing the docdata's filename will be more useful
        }
        data.append(record)
    data_df = pd.DataFrame(data)
    data_df = data_df.sample(frac=1.0, random_state=42)  # randomize order of text pairs

    data_set = set()
    used_pair_indexes = dict()
    for key in data_df.index.values:
        used_pair_indexes[key] = set()
    for data in data_df["author"].unique():
        # Same author
        author = data_df[data_df["author"] == data]
        index1 = author.index.values[0]
        available_part_of_df = author.drop(index=index1)
        if len(author) < 1:
            print("Something went wrong while making true pairs")
            break
        index2 = available_part_of_df.index.values[0]
        while index2 in used_pair_indexes[index1] or index1 in used_pair_indexes[index2]:
            available_part_of_df = author.drop(index=index2)
            if len(available_part_of_df) < 1:
                print("Something went wrong while making positive pairs")
                break
            index2 = available_part_of_df.index.values[0]
        data_set.add(frozenset({index1, index2}))
        used_pair_indexes[index1].add(index2)
        used_pair_indexes[index2].add(index1)
        # 4 different authors
        sub_dataframe = data_df[data_df["author"] != data]
        available_part_of_df = sub_dataframe
        if len(available_part_of_df) < 4:
            max_num = len(available_part_of_df)
            print("THERE ARE NO 4 RECORDS IN THE DF")
        else:
            max_num = 4
        for i in range(max_num):
            index1 = author.index.values[0]
            index2 = available_part_of_df.index.values[0]
            while index2 in used_pair_indexes[index1] or index1 in used_pair_indexes[index2]:
                available_part_of_df = available_part_of_df.drop(index=index2)
                if len(available_part_of_df) < 1:
                    print("Something went wrong while making negative pairs")
                    break
                index1 = author.index.values[0]
                index2 = available_part_of_df.index.values[0]
            data_set.add(frozenset({index1, index2}))
            used_pair_indexes[index1].add(index2)
            used_pair_indexes[index2].add(index1)

    data_list = []
    for data in data_set:
        list_current_data = list(data)
        record = {
            "text1": data_df["text"][list_current_data[0]],
            "text2": data_df["text"][list_current_data[1]],
            "author1": data_df["author"][list_current_data[0]],
            "author2": data_df["author"][list_current_data[1]],
            "docname1": data_df["docname"][list_current_data[0]],  # storing the docdata's filename will be more useful
            "docname2": data_df["docname"][list_current_data[1]],
            "label": int(data_df["author"][list_current_data[0]] == data_df["author"][list_current_data[1]])
        }
        data_list.append(record)
    df_final = pd.DataFrame(data_list)
    df_final.to_csv(os.path.join(path, f"{corpus_name}_processed.csv"), index=False)

    print("Record count is okay: ", len(df_final) == len(data_set) and len(data_set) == 5 * data_df["author"].nunique())
    print("True to all label ratio: ", len(df_final[df_final["label"] == 1]) / len(df_final))
    print("Number of unique texts:", len(set(df_final["text1"]).union(set(df_final["text2"]))))
    print("Number of unique authors:", len(set(df_final["author1"]).union(set(df_final["author2"]))))

    return df_final


def df_analysis(path: str):
    """
    Get different metrics and information about the dataframe such as the containg author and text number etc.

    :param df_path: The path of the dataframe.
    """

    df = pd.read_csv(path)
    for index in df.index:
        if [df["author1"][index], df["author2"][index]] != sorted([df["author1"][index], df["author2"][index]]):
            df["author1"][index], df["author2"][index] = df["author2"][index], df["author1"][index]
            df["text1"][index], df["text2"][index] = df["text2"][index], df["text1"][index]
    df_dropped = df.drop_duplicates(subset=["author1", "author2", "text1", "text2"])
    print("\nThere are duplicates:", len(df) != len(df_dropped), "len original:", len(df), "len dropped",
          len(df_dropped))

    print("Record count: ", len(df))
    print("True to all label ratio: ", len(df[df["label"] == 1]) / len(df))
    print("Number of unique texts:", len(set(df["text1"]).union(set(df["text2"]))))
    print("Number of unique authors:", len(set(df["author1"]).union(set(df["author2"]))))
    print("Number of true labels: ", len(df[df["label"] == 1]))
    print("Number of false labels: ", len(df[df["label"] == 0]), "\n")

    lista0 = [abs(len(x) - len(y)) for x, y in zip(df[df["label"] == 0]["text1"], df[df["label"] == 0]["text2"])]
    print("0 label min:", min(lista0))
    print("0 label max:", max(lista0))
    print("0 label mean:", st.mean(lista0))
    print("0 label median:", st.median(lista0))
    print("0 label mode:", st.stdev(lista0), "\n")

    lista1 = [abs(len(x) - len(y)) for x, y in zip(df[df["label"] == 1]["text1"], df[df["label"] == 1]["text2"])]
    print("1 label min:", min(lista1))
    print("1 label max:", max(lista1))
    print("1 label mean:", st.mean(lista1))
    print("1 label median:", st.median(lista1))
    print("1 label mode:", st.stdev(lista1))


def directory_of_doc_data_jsons_to_single_json(path: str, skip_word_data=False):
    """
    Writes the data of every doc_data.json found on path (a directory) into a single json, where the key is the original
    json's filename, the value is the original json's data.

    :param path: The path of the directory containing the doc_data jsons.
    :param skip_word_data: If True, WordData info will not be saved in the large JSON, conserving significant diskspace.
    """

    data = dict()
    files = os.listdir(path)
    for filename in tqdm(files, desc="Reading JSON files...", total=len(files)):
        if not filename.endswith(".json") or filename == "all_jsons.json":
            continue
        small_json_data = json.load(open(os.path.join(path, filename), encoding="UTF-8"))

        if skip_word_data:
            del small_json_data["totalword"]

        data[filename] = small_json_data
    json.dump(data, open(os.path.join(path, "all_jsons.json"), "w", encoding="UTF-8"))


def train_test_val_split(path: str, ratio: Tuple[float, float, float], train_path: str, test_path: str, val_path: str):
    """
    Split the corpus into train-test-val.

    :param path: The corpus to split. A path to a directory of texts.
    :param ratio: The ratio of the train-test-val sizes. For example: 0.8, 0.1, 0.1. Their sum neeeds to be 1.
    :param train_path: The directory to save the training corpus in.
    :param test_path: The directory to save the test corpus in.
    :param val_path: The directory to save the validation corpus in.
    """

    if not (0.99999999 < (ratio[0] + ratio[1] + ratio[2]) < 1.00000001):
        raise ValueError(f"The sum of the ratios should be 1, but was {ratio[0] + ratio[1] + ratio[2]}")

    all_texts = sorted([filename for filename in os.listdir(path) if filename.endswith(".txt")])
    corpus_size = len(all_texts)
    train_size = int(corpus_size * ratio[0])
    test_size = int(corpus_size * ratio[1])

    prev_author = ""
    subcorpus_size = 0

    do_train = True
    do_test = False
    do_val = False

    for filename in tqdm(all_texts, desc="Splitting corpus...", total=corpus_size):
        if not filename.endswith(".txt"):
            continue
        subcorpus_size += 1
        full_path = os.path.join(path, filename)
        author = filename.split("_")[0]
        if do_train and subcorpus_size >= train_size and prev_author != "" and author != prev_author:
            do_train = False
            do_test = True
            subcorpus_size = 0
        elif do_test and subcorpus_size >= test_size and prev_author != "" and author != prev_author:
            do_test = False
            do_val = True
            subcorpus_size = 0

        if do_train:
            shutil.copyfile(full_path, os.path.join(train_path, filename))
        elif do_test:
            shutil.copyfile(full_path, os.path.join(test_path, filename))
        elif do_val:
            shutil.copyfile(full_path, os.path.join(val_path, filename))
        else:
            raise ValueError("do_train, do_test, do_val are all False, which should not happen.")

        prev_author = author


def token_freq_histogram(path: str, doc_data_path: str, fig_file_name: str, alternate_doc_data_path: str = "") -> Dict[
    str, Union[int, float]]:
    print("Performing token-related corpus analysis for histogram...")

    files = os.listdir(path)
    token_nums = []

    for filename in tqdm(files, total=len(files), desc="Processing files..."):
        if not filename.endswith(".txt"):
            continue

        doc_data_json_path = os.path.join(doc_data_path, filename.replace(".txt", ".json"))
        if not os.path.exists(doc_data_json_path):
            doc_data_json_path = os.path.join(alternate_doc_data_path, filename.replace(".txt", ".json"))

        doc_data = json.load(open(doc_data_json_path, encoding="UTF-8"))
        if doc_data["tokenNum"] <= 1000:
            token_nums.append(doc_data["tokenNum"])

    # Define the bin edges
    bin_edges = np.arange(0, max(token_nums) + 51, 50)

    # Create the histogram
    plt.hist(token_nums, bins=bin_edges, edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Token Numbers')

    # Add brackets and labels for each column
    for i in range(len(bin_edges) - 1):
        x = bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2  # x-coordinate for the label
        y = plt.hist(token_nums, bins=bin_edges, edgecolor='black')[0][i]  # height of the column
        plt.text(x, y, str(int(y)), ha='center', va='bottom')  # display the value above the column

    # Save the figure as a png
    plt.savefig(fig_file_name)


def merge_index_gpt_gemini_data(index_data_path: str, gpt_data_path: str, gemini_data_path: str) -> pd.DataFrame:
    """
    Combine the index data with the GPT and Gemini data.

    :param index_data_path: Path of the index corpus.
    :param gpt_data_path: Path of the GPT csv.
    :return: The merged DataFrame.
    """
    random.seed(42)
    gpt_texts = pd.read_csv(gpt_data_path)["answer"].tolist()
    gemini_texts = pd.read_csv(gemini_data_path)["text"].tolist()

    all_gpt_token_nums = [len(text.split(" ")) for text in gpt_texts]
    all_gemini_token_nums = [len(text.split(" ")) for text in gemini_texts]

    average_ml_token_num = int(np.mean([np.mean(all_gpt_token_nums), np.mean(all_gemini_token_nums)]))
    average_ml_token_std = int(np.mean([np.std(all_gpt_token_nums), np.std(all_gemini_token_nums)]))

    index_min_len = average_ml_token_num - average_ml_token_std
    index_max_len = average_ml_token_num + average_ml_token_std

    number_of_average_len_index_texts_needed = int((len(gpt_texts) * 0.85))
    number_of_average_len_index_texts = 0

    index_files = sorted(os.listdir(index_data_path))
    number_of_authors_in_index = len(set([name[:4] for name in index_files]))
    number_of_same_author_index_texts_needed = max(0, len(gpt_texts) - number_of_authors_in_index)
    files_already_read = set()
    prev_author = ""
    index_texts = []
    print("Processing...")
    for filename in index_files:
        if len(files_already_read) == len(gpt_texts):
            break


        author = filename.split("_")[0]
        if author != prev_author:
            with open(os.path.join(index_data_path, filename), encoding="UTF-8") as f:
                text = f.read()
                text = re.sub(r" ([.?!:;,)\]]+)", r"\1", text)
                text = re.sub(r"([(\[]) ", r"\1", text)

                token_num = len(text.split(" "))
                if number_of_average_len_index_texts < number_of_average_len_index_texts_needed:
                    if token_num >= index_min_len and token_num <= index_max_len:
                        number_of_average_len_index_texts += 1
                    else:
                        continue
                index_texts.append(text)
                files_already_read.add(text)
        prev_author = author

    if number_of_same_author_index_texts_needed > 0:
        print(f"{number_of_same_author_index_texts_needed} extra texts read.")
        extra_filenames = random.sample(set(index_files).difference(files_already_read),
                                        number_of_same_author_index_texts_needed)
        for filename in extra_filenames:
            with open(os.path.join(index_data_path, filename), encoding="UTF-8") as f:
                text = f.read()
                index_texts.append(text)

    # preprocess, remove whitespaces before punctuations
    index_texts = [re.sub(r" ([.?!:;,)\]]+)", r"\1", text) for text in index_texts]
    # preprocess, remove whitespaces after ( and [
    index_texts = [re.sub(r"([(\[]) ", r"\1", text) for text in index_texts]

    all_texts = index_texts + gpt_texts + gemini_texts
    labels = len(index_texts) * [0] + len(gpt_texts) * [1] + len(gemini_texts) * [
        2]  # 0 is human, 1 is GPT, 2 is Gemini

    # the DocDatas will be named according to the text names
    text_names = [f"index{num}" for num in range(len(index_texts))] + [f"gpt{num}" for num in range(len(gpt_texts))] + [
        f"gemini{num}" for num in range(len(gemini_texts))]

    index_with_gpt_and_gemini_corpus = pd.DataFrame({"text": all_texts, "text_name": text_names, "label": labels})
    index_with_gpt_and_gemini_corpus = index_with_gpt_and_gemini_corpus.sample(frac=1.0, random_state=42)
    return index_with_gpt_and_gemini_corpus


def create_docdatas_for_tdk(data: pd.DataFrame, output_path: str, model: spacy.Language, benepar_parser: Parser,
                            token_freq_dict: Dict[str, int], h_spell: Hunspell):
    """
    Create doc_data.json files for all the TDK data.

    :param data: The TDK data. Needs a text and a text_name field (for naming the doc_data).
    :param output_path: The output path for the docdatas. This dictionary will contain json files, and each filename
    will match the corresponding text_name.
    :param model: A Spacy language model, required for the doc_data analysis.
    :param benepar_parser: A benepar model, required for the doc_data analyis.
    :param token_freq_dict: A dictionary containing token frequencies, required for the doc_data analyis.
    :param h_spell: A Spelling model, required for the doc_data analyis.
    """

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing..."):
        text = row["text"]
        doc_data_name = row["text_name"] + ".json"
        full_path = os.path.join(output_path, doc_data_name)
        if os.path.exists(full_path):
            continue

        raw_data = RawDocInfo(text=text, wordlists=WordLists())
        data = create_doc_data(raw_data, model=model, benepar_parser=benepar_parser, token_freq_dict=token_freq_dict,
                               h_spell=h_spell)
        data = data.__dict__
        del data["totalword"]
        json.dump(data, open(full_path, "w", encoding="UTF-8"), ensure_ascii=False)


def split_tdk_corpus_to_train_val_test(corpus_path: str, ratio: Tuple[float, float, float]) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the corpus into train-val-test.

    :param corpus_path: Path of the full corpus.
    :param ratio: A tuple of 3 floats between 0 and 1. Their sum must be 1. First number is the train ratio,
    second is the validation ratio, third is the test ratio.
    :return: A tuple of 3 DataFrames: train, val, test.
    """
    if sum(ratio) < 1:
        raise ValueError("Wrong ratio.")

    data = pd.read_csv(corpus_path)
    total_size = len(data)

    train_size = int(total_size * ratio[0])
    val_size = int(total_size * ratio[1])
    test_size = int(total_size * ratio[2])

    data = data.sample(frac=1.0, random_state=42)  # random shuffle data

    train_data = data.iloc[0:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    return train_data, val_data, test_data

def random_misspell_text(text: str) -> Tuple[str, int]:
    new_text = []
    k = random.choice([90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

    misspell_dict = {
        "q": ["w"],
        "w": ["q", "e"],
        "e": ["w", "r"],
        "r": ["e", "t"],
        "t": ["r", "z"],
        "z": ["t", "u"],
        "u": ["z", "i"],
        "i": ["u", "o"],
        "o": ["i", "p"],
        "p": ["o", "ő"],
        "ő": ["p", "ú"],
        "ö": ["9", "ü"],
        "ü": ["ö", "ó"],
        "ó": ["ü"],
        "ú": ["ő"],
        "a": ["s"],
        "s": ["a", "d"],
        "d": ["s", "f"],
        "f": ["d", "g"],
        "g": ["f", "h"],
        "h": ["g", "j"],
        "j": ["h", "k"],
        "k": ["j", "l"],
        "l": ["k", "é"],
        "é": ["l", "á"],
        "á": ["é", "ű"],
        "ű": ["á"],
        "í": ["y"],
        "y": ["í", "x"],
        "x": ["y", "c"],
        "c": ["x", "v"],
        "v": ["c", "b"],
        "b": ["v", "n"],
        "n": ["b", "m"],
        "m": ["n"],
        ",": ["m", "."],
        "?": [",", "M", ":"],
        ".": [",", "-"],
        "-": ["."],
        ":": ["?", ".", "_"]
    }

    misspelled_indexes = random.sample(range(0, len(text)), k=k)
    number_of_misspells = 0

    for index, char in enumerate(text):
        if index in misspelled_indexes:
            if char in misspell_dict:
                number_of_misspells += 1

            char = misspell_dict.get(char, char)
            if type(char) == list:
                char = random.choice(char)
        new_text.append(char)

    new_text = "".join(new_text)
    return new_text, number_of_misspells

def random_extra_punctuations(text):
    new_text = []
    number_of_extra_multiple_puncts = 0
    for char in text:
        rng = random.choice(list(range(0, 11)))
        if char == ".":
            if rng == 10:
                char = "..."
                number_of_extra_multiple_puncts += 1
        elif char == "?":
            if rng == 10:
                char = "?!"
                number_of_extra_multiple_puncts += 1
            elif rng == 9:
                char = "???"
                number_of_extra_multiple_puncts += 1
            elif rng == 8 or rng == 7:
                char = "??"
                number_of_extra_multiple_puncts += 1
        elif char == "!":
            if rng > 7:
                char = "!!"
                number_of_extra_multiple_puncts += 1
            elif rng == 7:
                char = "!!!"
                number_of_extra_multiple_puncts += 1

        new_text.append(char)
    new_text = "".join(new_text)
    return new_text, number_of_extra_multiple_puncts