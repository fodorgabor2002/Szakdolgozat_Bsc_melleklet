import os.path
from typing import List, Set, Dict, Union, Tuple, Literal
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def tdk_get_doc_data_features_for_dataframe(base_dataframe: pd.DataFrame, doc_data_path: str) -> pd.DataFrame:
    """
    Get DocData features for a dataframe of texts.

    :param base_dataframe: The DataFrame of texts. Should contain (at the minimum) the following columns:
    text, text_name. The text_name is the DocData's name associated with the text.
    :param doc_data_path: The path of the single, large json file containing all the small doc_data json data.
    :return: The original DataFrame expanded with the DocData features. The original DataFrame does not change.
    """

    base_dataframe = base_dataframe.copy()  # ensuring the original DataFrame does not change
    doc_frame_data = []

    if not os.path.isfile(doc_data_path):
        raise ValueError(f"The large DocData json {doc_data_path} does not exist.")

    all_doc_data = json.load(open(doc_data_path, encoding="UTF-8"))

    for _, row in tqdm(base_dataframe.iterrows(), desc="Reading DocDatas...", total=len(base_dataframe)):
        try:
            data = all_doc_data[row["text_name"] + ".json"]
        except KeyError:
            raise KeyError(f"No DocData found for {row['text_name']}")

        data = {feature: (value if type(value) != bool else int(value)) for feature, value in data.items()}

        doc_frame_data.append(data)

    doc_frame = pd.DataFrame(doc_frame_data)

    final_frame = pd.concat([base_dataframe, doc_frame], axis=1)

    return final_frame

def normalize_numerical_data(data: pd.DataFrame, numerical_columns: List[str], scaler: MinMaxScaler = None) -> Tuple[pd.DataFrame, Union[MinMaxScaler, Literal[None]]]:
    """
    Normalize numerical columns in the DataFrame using MinMaxScaling.

    :param data: The original dataframe.
    :param numerical_columns: The columns to normalize.
    :param scaler: The scaler to use for MinMaxScaling. None by default, this is for scaling with a scaler that was
    already fit on previous data.
    :return: The normalized DataFrame and the scaler used to normalize it.
    """

    if len(numerical_columns) == 0:
        print("Nothing to normalize (no numerical data). Returning original DataFrame.")
        return data, None

    data = data.copy()

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data[numerical_columns])

    data[numerical_columns] = scaler.transform(data[numerical_columns])
    # Make sure every value is between 0 and 1
    data[numerical_columns] = data[numerical_columns].clip(lower=0, upper=1)

    return data, scaler
