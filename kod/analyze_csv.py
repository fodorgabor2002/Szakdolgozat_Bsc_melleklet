import os
import random

import pandas as pd
import re
from typing import List
import json
import numpy as np
import matplotlib.pyplot as plt




def split_multiple_comments_in_one_text(text: str) -> List[str]:
    texts = re.split(r"\d\.", text)
    texts = [text.strip() for text in texts if len(text.strip()) > 0]
    return texts

def txt_files_to_gpt_output_dataframe(folder_path: str, text_name_base: str) -> pd.DataFrame:
    files = os.listdir(folder_path)
    data = []
    for i, filename in enumerate(files):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(folder_path, filename), encoding="UTF-8") as f:
            text = f.read()
        data.append({
            "text": text,
            "text_name": f"{text_name_base}{i}"
        })
    dataframe = pd.DataFrame(data)
    return dataframe

def combine_cms(path: str, output_path: str):
    bottom_row_sum = [0, 0, 0]
    middle_row_sum = [0, 0, 0]
    top_row_sum = [0, 0, 0]

    for filename in os.listdir(path):
        if not filename.startswith("fold") or not filename.endswith(".json"):
            continue
        cm = json.load(open(os.path.join(path, filename), encoding="UTF-8"))[0]

        bottom_row = cm["z"][0]
        middle_row = cm["z"][1]
        top_row = cm["z"][2]

        bottom_row_sum = [sum(nums) for nums in zip(bottom_row_sum, bottom_row)]
        middle_row_sum = [sum(nums) for nums in zip(middle_row_sum, middle_row)]
        top_row_sum = [sum(nums) for nums in zip(top_row_sum, top_row)]

    matrix = np.array([top_row_sum, middle_row_sum, bottom_row_sum])

    # Define labels for the axes
    labels = ['human', 'GPT', 'gemini']

    # Create the heatmap plot
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')

    # Add labels to the axes
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels[::-1])

    # Add value annotations to the center of each square
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(matrix[i, j]), ha='center', va='center', color='red')

    # Add a colorbar
    plt.colorbar()

    # Save the plot to an image file
    plt.savefig(output_path)
    return matrix, top_row_sum, middle_row_sum, bottom_row_sum

if __name__ == "__main__":
    path = "resources/data/all_data_0421/index_gpt_gemini_0421.csv"
    data = pd.read_csv(path)

    data.drop_duplicates(subset="text", inplace=True)  # in case there is duplicated data
    data = data[(data["text"].str.len() > 1200) | (data["label"] != 2)]  # drop some trash Gemini data
    data = data[(data["text"].str.len() > 100) | (data["label"] != 1)]  # drop trash GPT data

    dummy_labels = [0] * 1000 + [1] * 999 + [2] * 486
    random.shuffle(dummy_labels)

    def dummy_baseline(text):
        return len(text) < 1000 or len(text) > 4400 or len(text.split(" ")) < 150 or len(text.split(" ")) > 700

    data["pred_label"] = dummy_labels

    correct_data = data[data["pred_label"] == data["label"]]
    wrong_data = data[data["pred_label"] != data["label"]]
