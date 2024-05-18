import os
import json
import shutil
import uuid
from typing import Dict, Union, List

import pandas
import clearml
import pandas as pd
import process_data
from clearml import Task, Logger
from multimodal_transformers.data import load_data
from multimodal_transformers.model import BertWithTabular
from multimodal_transformers.model import TabularConfig
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertConfig, IntervalStrategy
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import time
import torch
import copy
import joblib
from sklearn.metrics import confusion_matrix
from data_preparation import random_misspell_text

tokenizer = BertTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
pretrained_model_name = 'SZTAKI-HLT/hubert-base-cc'

CONFIG_PATH = "tdk_misspelled_config.json"

base_config = json.load(open(CONFIG_PATH, encoding="UTF-8"))

#DATA_PATH = "data/all_data_0421/index_gpt_gemini_0421.csv"
#DOC_DATA_PATH = "data/all_data_0421/osszes_docdata/all_jsons.json"

DATA_PATH = "tdk_data/all_data_0421/index_gpt_gemini_0421.csv"
DOC_DATA_PATH = "tdk_data/all_data_0421/osszes_docdata/all_jsons.json"

all_data = pd.read_csv(DATA_PATH)
all_data.drop_duplicates(subset="text", inplace=True)  # in case there is duplicated data
all_data = all_data[(all_data["text"].str.len() > 1200) | (all_data["label"] != 2)]  # drop some trash Gemini data
all_data = all_data[(all_data["text"].str.len() > 100) | (all_data["label"] != 1)]  # drop trash GPT data
all_data = all_data.sample(frac=1, random_state=42)  # random shuffle again, for cross-validation
total_size = len(all_data)
NUMBER_OF_FOLDS = 10
fraction_size = total_size // NUMBER_OF_FOLDS

label_names = ["human", "GPT", "gemini"]

other_numerical_columns = {
    "sentencesPerParagraph", "fleschReadingEaseScore", "gunningFogIndex", "fleschKincaidGradeLevel",
    "colemanLiauIndex",
    "smogIndex", "automatedReadabilityIndex", "linsearWriteIndex"
}

combine_feat_method_options = ["attention_on_cat_and_numerical_feats", "gating_on_cat_and_num_feats_then_sum",
                               "weighted_feature_sum_on_transformer_cat_and_numerical_feats"]

drop_col_options = [None]

rates_by_categories = {
    "all": {"lemmaRate", "allUpperRate", "firstUpperRate", "declarSentRate", "imperSentRate", "questionRate",
            "nounRate", "verbRate", "adjRate", "infRate", "partPastRate", "partPresRate", "partFutRate", "transRate",
            "xRate", "advRate", "properNounRate", "numRate", "conjRate", "punctRate", "pronRate", "relPronRate",
            "demPronRate",
            "adposRate", "multiplePunctRate", "partsDensityRate", "pastTenseRate", "presentTenseRate", "condRate",
            "impRate", "causRate", "modalRate", "sg1VerbRate", "sg2VerbRate", "sg3VerbRate", "pl1VerbRate",
            "pl2VerbRate", "pl3VerbRate", "superlatRate", "comparatRate", "pluralNounRate", "subjRate", "objRate",
            "attRate", "subordRate", "adverbRate", "coordRate", "partsRate", "clauseRate", "simpleRate", "complexRate",
            "complexity1Rate", "complexity2Rate", "complexity3Rate", "complexClauseRate", "sentencesPerParagraph",
            "wordsLengthRate", "negationRate", "functionRate", "contentRate", "postscriptRate", "quoteRate",
            "dashRate", "smileyRate", "dateRate", "hesitationRate", "restartRate", "repeatRate",
            "wordBeforeHesitationRate", "wordAfterHesitationRate", "easyWordRate", "hardWordRate",
            "locationNameRate", "personNameRate", "organizationNameRate", "misspelledRate", "missingAccentRate"},

    # statistical features
    "statistical": {"lemmaRate", "allUpperRate", "firstUpperRate", "declarSentRate", "imperSentRate", "questionRate"},

    # morphological features - POS
    "morphological": {"nounRate", "verbRate", "adjRate", "infRate", "partPastRate", "partPresRate", "partFutRate",
                      "transRate", "xRate",
                      "advRate", "properNounRate", "numRate", "conjRate", "punctRate", "pronRate", "relPronRate",
                      "demPronRate",
                      "adposRate", "multiplePunctRate", "partsDensityRate"},

    # morphological features - deep morph
    "morphological_deep": {"pastTenseRate", "presentTenseRate", "condRate", "impRate",
                           "causRate", "modalRate", "sg1VerbRate", "sg2VerbRate", "sg3VerbRate", "pl1VerbRate",
                           "pl2VerbRate", "pl3VerbRate",
                           "superlatRate", "comparatRate", "pluralNounRate"},

    # syntactic features
    "syntactic": {"subjRate", "objRate", "attRate", "subordRate", "adverbRate",
                  "coordRate", "partsRate", "clauseRate", "simpleRate", "complexRate", "complexity1Rate",
                  "complexity2Rate",
                  "complexity3Rate", "complexClauseRate", "sentencesPerParagraph", "wordsLengthRate"},

    # semantic features
    "semantic": {"negationRate", "functionRate", "contentRate", "postscriptRate"},

    # pragmatic features
    "pragmatic": {"quoteRate", "dashRate", "smileyRate", "dateRate", "hesitationRate", "restartRate",
                  "repeatRate", "wordBeforeHesitationRate", "wordAfterHesitationRate"},

    # phonetic features
    "phonetic": {"easyWordRate", "hardWordRate"},

    # NER
    "NER": {"locationNameRate", "personNameRate", "organizationNameRate"},

    # Spelling
    "spelling": {"misspelledRate", "missingAccentRate"}
}


def compute_metrics(eval_pred):
    global top_macro_f1_for_fold, current_epoch, label_names, total_epoch_number, macro_f1_values_over_folds

    m1 = evaluate.load('accuracy')
    m2 = evaluate.load('f1')
    m3 = evaluate.load('precision')
    m4 = evaluate.load('recall')

    logits, labels = eval_pred
    logits = logits[0]
    predictions = np.argmax(logits, axis=1)

    results = dict()

    cm = confusion_matrix(labels, predictions)

    Logger.current_logger().report_confusion_matrix(
        f"Validation Set Confusion Matrix - after epoch {current_epoch}",
        "ignored",
        matrix=cm,
        xaxis="Predicted",
        xlabels=label_names,
        yaxis="Real",
        ylabels=label_names
    )

    acc = m1.compute(predictions=predictions, references=labels)["accuracy"]

    results["accuracy"] = acc

    macro_f1 = m2.compute(predictions=predictions, references=labels, average="macro")["f1"]
    macro_precision = m3.compute(predictions=predictions, references=labels, average="macro")["precision"]
    macro_recall = m4.compute(predictions=predictions, references=labels, average="macro")["recall"]

    results["macro_f1"] = macro_f1
    results["macro_precision"] = macro_precision
    results["macro_recall"] = macro_recall

    micro_f1 = m2.compute(predictions=predictions, references=labels, average="micro")["f1"]
    micro_precision = m3.compute(predictions=predictions, references=labels, average="micro")["precision"]
    micro_recall = m4.compute(predictions=predictions, references=labels, average="micro")["recall"]

    results["micro_f1"] = micro_f1
    results["micro_precision"] = micro_precision
    results["micro_recall"] = micro_recall

    weighted_f1 = m2.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    weighted_precision = m3.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    weighted_recall = m4.compute(predictions=predictions, references=labels, average="weighted")["recall"]

    results["weighted_f1"] = weighted_f1
    results["weighted_precision"] = weighted_precision
    results["weighted_recall"] = weighted_recall

    if macro_f1 > top_macro_f1_for_fold:
        top_macro_f1_for_fold = macro_f1

    # "human" - 0
    # "GPT" - 1
    # "gemini" - 2

    human_preds = [1 if pred == 0 else 0 for pred in predictions]
    human_truths = [1 if truth == 0 else 0 for truth in labels]

    gpt_preds = [1 if pred == 1 else 0 for pred in predictions]
    gpt_truths = [1 if truth == 1 else 0 for truth in labels]

    gemini_preds = [1 if pred == 2 else 0 for pred in predictions]
    gemini_truths = [1 if truth == 2 else 0 for truth in labels]

    human_f1 = m2.compute(predictions=human_preds, references=human_truths, average="binary")[
        "f1"]
    human_precision = \
        m3.compute(predictions=human_preds, references=human_truths, average="binary")[
            "precision"]
    human_recall = m4.compute(predictions=human_preds, references=human_truths, average="binary")[
        "recall"]

    gpt_f1 = \
        m2.compute(predictions=gpt_preds, references=gpt_truths, average="binary")[
            "f1"]
    gpt_precision = \
        m3.compute(predictions=gpt_preds, references=gpt_truths, average="binary")["precision"]
    gpt_recall = \
        m4.compute(predictions=gpt_preds, references=gpt_truths, average="binary")["recall"]

    gemini_f1 = \
        m2.compute(predictions=gemini_preds, references=gemini_truths, average="binary")[
            "f1"]
    gemini_precision = \
        m3.compute(predictions=gemini_preds, references=gemini_truths, average="binary")["precision"]
    gemini_recall = \
        m4.compute(predictions=gemini_preds, references=gemini_truths, average="binary")["recall"]

    results["human_f1"] = human_f1
    results["human_precision"] = human_precision
    results["human_recall"] = human_recall
    results["gpt_f1"] = gpt_f1
    results["gpt_precision"] = gpt_precision
    results["gpt_recall"] = gpt_recall
    results["gemini_f1"] = gemini_f1
    results["gemini_precision"] = gemini_precision
    results["gemini_recall"] = gemini_recall

    current_epoch += 1

    if current_epoch == total_epoch_number:
        macro_f1_values_over_folds.append(top_macro_f1_for_fold)
        macro_f1_average_over_folds = sum(macro_f1_values_over_folds) / len(macro_f1_values_over_folds)

        tensorboard_writer.add_scalar("mf1_avg", macro_f1_average_over_folds)
    return results


def experiment(config: Dict[str, Union[int, float, List[str]]], train_data: pd.DataFrame, val_data: pd.DataFrame,
               extra_clearml_tags: List[str] = None):
    global tensorboard_writer, current_epoch, label_names, macro_f1_values_over_folds, total_epoch_number

    print("Beginning experiment...")
    current_epoch = 0

    # to avoid modifying the "global" copy
    train_data = train_data.copy()
    val_data = val_data.copy()

    text_columns = ["text"]
    categorical_columns = set()

    numerical_columns = {col for col in train_data.columns if col.endswith("Rate")
                         or col in other_numerical_columns or col in other_numerical_columns}

    # ignore features that are not in config["used_features"]
    filtered_numerical_columns = []
    filtered_categorical_columns = []
    for feature in config["used_features"]:
        feature_is_numerical = False
        feature_is_categorical = False

        if feature in numerical_columns:
            filtered_numerical_columns.append(feature)
            feature_is_numerical = True

        if not feature_is_numerical:
            if feature in categorical_columns:
                filtered_categorical_columns.append(feature)
                feature_is_categorical = True

            if not feature_is_categorical:
                raise ValueError(f"{feature} is in used_features but was not calculated.")

    categorical_columns = filtered_categorical_columns
    numerical_columns = filtered_numerical_columns

    print("Used numerical features: ", numerical_columns)
    print("Used categorical features: ", categorical_columns)

    if len(train_data) % config["batch_size"] == 1:
        train_data = train_data.iloc[:-1]
        print("Dropped final train record to avoid Batch Normalization error")
    if len(val_data) % config["batch_size"] == 1:
        val_data = val_data.iloc[:-1]
        print("Dropped final validation record to avoid Batch Normalization error")

    train_data, scaler = process_data.normalize_numerical_data(data=train_data, numerical_columns=numerical_columns)
    val_data, _ = process_data.normalize_numerical_data(data=val_data, numerical_columns=numerical_columns,
                                                        scaler=scaler)

    numeric_dimensions = len(numerical_columns)
    num_classes = len(train_data["label"].value_counts())

    cat_dimensions = len(categorical_columns)

    print("Loading train_dataset...")
    train_load_time = time.time()
    train_dataset = load_data(train_data, text_cols=text_columns, tokenizer=tokenizer, label_col="label",
                              label_list=label_names, categorical_cols=categorical_columns,
                              numerical_cols=numerical_columns,
                              sep_text_token_str="[SEP]", categorical_encode_type=None, max_token_length=512)

    print("Time it took to load train_dataset: ", time.time() - train_load_time)

    print("Loading val_dataset...")
    val_load_time = time.time()
    val_dataset = load_data(val_data, text_cols=text_columns, tokenizer=tokenizer, label_col="label",
                            label_list=label_names, categorical_cols=categorical_columns,
                            numerical_cols=numerical_columns,
                            sep_text_token_str="[SEP]", categorical_encode_type=None, max_token_length=512)

    print("Time it took to load val_dataset: ", time.time() - val_load_time)

    task_name_id = int(time.time())  # current unix time, use as task ID

    tags = [f"FOLD_{fold_index}", "gemini", "token_num_balanced", "misspelled", "0421_fixed"] + extra_clearml_tags if extra_clearml_tags is not None else [
        f"FOLD_{fold_index}", "gemini", "token_num_balanced", "misspelled", "0421_fixed"]
    task_name = f'Index GPT Gemini {task_name_id}'

    task = Task.init(project_name='GPT/GPT_Training', task_name=task_name,
                     task_type=Task.TaskTypes.training, reuse_last_task_id=False,
                     tags=tags)

    model_name = task_name

    best_models_dir = "best_models_0425_TDK_3_classes"
    if not os.path.isdir(best_models_dir):
        os.mkdir(best_models_dir)

    config["model_name"] = model_name

    task.connect(config, name="Hyperparameters")
    tensorboard_writer = SummaryWriter('./tensorboard_logs')

    print(num_classes)
    train_label_value_counts = train_data["label"].value_counts()
    val_label_value_counts = val_data["label"].value_counts()

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    tensorboard_writer.add_scalar("Number of index texts in train", train_label_value_counts[0])
    tensorboard_writer.add_scalar("Number of GPT texts in train", train_label_value_counts[1])
    tensorboard_writer.add_scalar("Number of Gemini texts in train", train_label_value_counts[2])
    tensorboard_writer.add_scalar("Number of index texts in validation", val_label_value_counts[0])
    tensorboard_writer.add_scalar("Number of GPT texts in validation", val_label_value_counts[1])
    tensorboard_writer.add_scalar("Number of Gemini texts in validation", val_label_value_counts[2])

    tensorboard_writer.add_scalar("Train Size", train_size)
    tensorboard_writer.add_scalar("Validation Size", val_size)

    bert_config = BertConfig.from_pretrained(pretrained_model_name)

    tabular_config = TabularConfig(
        combine_feat_method=config["combine_feat_method"],
        # change this to specify the method of combining tabular data
        cat_feat_dim=cat_dimensions,  # need to specify this
        numerical_feat_dim=numeric_dimensions,  # need to specify this
        num_labels=num_classes,  # need to specify this
        numerical_bn=(config["batch_size"] > 1)  # whether to batch_normalize numerical features
    )
    bert_config.tabular_config = tabular_config

    model = BertWithTabular.from_pretrained(pretrained_model_name, config=bert_config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"misspelled_model_checkpoints {task_name_id}",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        save_total_limit=5,
        logging_steps=10,
        metric_for_best_model='macro_f1',
        load_best_model_at_end=True
    )

    task.upload_artifact(name="Train Data", artifact_object=train_dataset)
    task.upload_artifact(name="Validation Data", artifact_object=val_dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.mkdir(os.path.join(best_models_dir, model_name))
    trainer.save_model(os.path.join(best_models_dir, model_name))
    model_config = model.config.to_dict()
    model_config["text_cols"] = text_columns
    model_config["numerical_cols"] = numerical_columns
    model_config["categorical_cols"] = categorical_columns
    json.dump(model_config, open(os.path.join(best_models_dir, model_name, "config.json"), "w", encoding="UTF-8"),
              ensure_ascii=False)

    # Save the data scaler that was used on the training (and the validation) data (if one had been used)
    if scaler is not None:
        scaler_filename = f"{os.path.join(best_models_dir, model_name, model_name + '_scaler.save')}"
        joblib.dump(scaler, scaler_filename)

    shutil.rmtree(f"misspelled_model_checkpoints {task_name_id}", ignore_errors=False, onerror=None)

    task.close()
    tensorboard_writer.close()


text_only_already_done = False

# Run an experiment for every single combine_feat_method - drop_col combo, for example:
for combine_feat_method in combine_feat_method_options:
    for category_name, category in rates_by_categories.items():
        for drop_col in drop_col_options:
            macro_f1_values_over_folds = []
            prev_index = 0
            validation_texts_seen_so_far = set()

            if combine_feat_method == "text_only":
                if text_only_already_done:
                    continue
                text_only_already_done = True

            for fold_index in range(NUMBER_OF_FOLDS):
                top_macro_f1_for_fold = 0
                next_index = prev_index + fraction_size
                val_data = all_data.iloc[prev_index:next_index]
                train_data = all_data.iloc[~all_data.index.isin(val_data.index)]

                prev_index = next_index

                # Sanity checks
                val_train_overlaps = len(set(train_data.text).intersection(set(val_data.text))) > 0
                if val_train_overlaps:
                    raise Exception("Train and validation overlap!")
                val_fold_overlaps = len(set(val_data.text).intersection(validation_texts_seen_so_far)) > 0
                if val_fold_overlaps:
                    raise Exception("Some of the validation data in current fold was also in a previous fold!")
                validation_texts_seen_so_far = validation_texts_seen_so_far.union(set(val_data.text))

                train_data.reset_index(inplace=True, drop=True)
                val_data.reset_index(inplace=True, drop=True)

                train_data = process_data.tdk_get_doc_data_features_for_dataframe(base_dataframe=train_data,
                                                                                  doc_data_path=DOC_DATA_PATH)
                val_data = process_data.tdk_get_doc_data_features_for_dataframe(base_dataframe=val_data,
                                                                                doc_data_path=DOC_DATA_PATH)

                new_texts_col = []
                new_misspelled_num_col = []
                new_misspelled_rate_col = []
                for _, row in val_data.iterrows():
                    new_text, extra_misspell_num = random_misspell_text(row["text"])
                    new_misspelled_num = row["misspelledNum"] + extra_misspell_num
                    new_misspelled_rate = new_misspelled_num / row["tokenNum"]
                    new_texts_col.append(new_text)
                    new_misspelled_num_col.append(new_misspelled_num)
                    new_misspelled_rate_col.append(new_misspelled_rate)

                val_data["text"] = new_texts_col
                val_data["misspelledNum"] = new_misspelled_num_col
                val_data["misspelledRate"] = new_misspelled_rate_col

                config = copy.deepcopy(base_config)

                # Rate-k után ez nem kell
                config["used_features"].extend(list(category))

                config["combine_feat_method"] = combine_feat_method
                if drop_col is not None:
                    config["used_features"].remove(drop_col)
                config["dropped_col"] = drop_col
                total_epoch_number = config["num_train_epochs"]
                print("Running experiment with this config: ", str(config))
                print("Dropped column is: ", drop_col)
                experiment(config, train_data=train_data, val_data=val_data, extra_clearml_tags=[category_name])
