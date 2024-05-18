import pandas as pd
import os

import joblib
import pandas as pd
import torch
from multimodal_transformers.data import load_data
from multimodal_transformers.model import BertWithTabular
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer
from tqdm import tqdm
import math
import random
import numpy as np
import json

import process_data


def infer_for_text_pair_multiple(model, tokenizer, data: pd.DataFrame, doc_data_path: str,
                                 scaler: MinMaxScaler = None) -> pd.DataFrame:
    """
    Infer labels (binary label, whether text1 and text2 has the same author for a whole test DataFrame.

    :param model: The model to use for the inference.
    :param tokenizer: The tokenizer used for the inference.
    :param data: The DataFrame to perform inference on. Should be similar in format to the one used in training.
    :param doc_data_path: Base path for the DocDatas for the data.
    :param scaler: The MinMaxScaler that was used on the model's training data. None by default.
    :return: A labeled DataFrame.
    """

    org_data = data.copy()
    text_columns = model.config.text_cols
    numerical_columns = model.config.numerical_cols
    categorical_columns = model.config.categorical_cols

    data.reset_index(inplace=True, drop=True)

    data = process_data.tdk_get_doc_data_features_for_dataframe(base_dataframe=data, doc_data_path=doc_data_path)

    data, _ = process_data.normalize_numerical_data(data=data, numerical_columns=numerical_columns, scaler=scaler)

    data_for_infer = load_data(data, text_cols=text_columns, tokenizer=tokenizer, label_col="label",
                              label_list=["human", "gpt"], categorical_cols=categorical_columns,
                              numerical_cols=numerical_columns,
                              sep_text_token_str="[SEP]", categorical_encode_type=None, max_token_length=512)

    BATCH_SIZE = 32
    pred_labels = []

    model.eval()

    for threshold in tqdm(range(0, len(data_for_infer), BATCH_SIZE), desc="Inferring...",
                          total=math.ceil(len(data_for_infer)/BATCH_SIZE)):
        batch = data_for_infer[threshold:threshold+BATCH_SIZE]
        if len(batch["cat_feats"]) == 0:
            with torch.no_grad():
                _, logits, _ = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    numerical_feats=batch["numerical_feats"],
                )
        else:
            with torch.no_grad():
                _, logits, _ = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    numerical_feats=batch["numerical_feats"],
                    cat_feats=batch["cat_feats"]
                )


        pred_labels.extend([label for label in np.argmax(logits, axis=1).numpy()])

    org_data["pred_label"] = pred_labels
    return org_data


if __name__ == "__main__":
    model_name = "Index GPT Gemini 1713850486"
    model = BertWithTabular.from_pretrained(os.path.join("..", "resources", model_name))
    tokenizer = BertTokenizer.from_pretrained(os.path.join("..", "resources", "tokenizer"))

    scaler_path = f"{os.path.join('..', 'resources', model_name, model_name + '_scaler.save')}"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    data_path = "data/ParlaMint-HU/pol_texts.csv"
    doc_data_path = "data/ParlaMint-HU/osszes_docdata/all_jsons.json"
    data = pd.read_csv(data_path)

    doc_data_dict = json.load(open(doc_data_path, encoding="UTF-8"))
    doc_data_keys = {key.replace(".json", "") for key in doc_data_dict.keys()}
    data = data[data["text_name"].isin(doc_data_keys)]
    data = data.sample(frac=1.0, random_state=42)

    human_pack1_path = "data/human_packs/andris_human_pack1.csv"
    human_pack2_path = "data/human_packs/andris_human_pack2.csv"
    human_pack3_path = "data/human_packs/andris_human_pack3.csv"
    human_pack1_docdata_path = "data/human_packs/human_pack1_docdata/all_jsons.json"
    human_pack2_docdata_path = "data/human_packs/human_pack2_docdata/all_jsons.json"
    human_pack3_docdata_path = "data/human_packs/human_pack3_docdata/all_jsons.json"

    human_pack1_frame = pd.read_csv(human_pack1_path)
    human_pack1_frame["label"] = [1] * len(human_pack1_frame)

    human_pack2_frame = pd.read_csv(human_pack2_path)
    human_pack2_frame["label"] = [1] * len(human_pack2_frame)

    human_pack3_frame = pd.read_csv(human_pack3_path)
    human_pack3_frame["label"] = [1] * len(human_pack3_frame)

    eszter_data_path = "data/Eszter_ChatGPT-szövegek/eszter_gpt_output.csv"
    eszter_docdata_path = "data/Eszter_ChatGPT-szövegek/osszes_docdata/all_jsons.json"
    eszter_data = pd.read_csv(eszter_data_path)
    eszter_data["label"] = [1] * len(eszter_data)