import json
import os
import re
import pandas as pd
import math

import numpy as np

config = None


def __init():
    load_config()


def load_config(file_path="config.json"):
    global config
    with open(file_path, "r") as f:
        config = json.load(f)
    return config


def read(*, filename, filter_samples: bool = False, only_rules: bool) -> pd.DataFrame:
    df = pd.read_excel(
        os.path.join(config["DATA_FOLDER"], filename),
        sheet_name=config["SHEET_NAME"],
        names=config["COL_NAMES"],
        usecols=config["COL_INTERVAL"],
        na_values=["NA"],
    )
    df = df[df.remarks.str.lower() != "invalid_sentence"]  # drop invalid sentences
    df = df[
        df.iloc[:, 4].isin(["y", "Y", "n", "N"])
    ]  # this is to only filter annotated sentences (to exclude ones after the end)
    if filter_samples:
        df = df[
            df.is_const.str.lower() == ("y" if only_rules else "n")
        ]  # filter by only_rules option
    return df


def extract_attributes(hmap):
    sum_of_attributes = {}
    for keyword, attributes in hmap.items():
        sum_of_attributes[keyword] = sum(
            int(value) for key, value in attributes.items()
        )
    return sum_of_attributes


def filter_hmap(hmap, sum_of_attributes, threshold):
    hmap_filtered = {
        keyword: attributes
        for keyword, attributes in hmap.items()
        if sum_of_attributes[keyword] != 0
        and attributes["not_included"] / sum_of_attributes[keyword] < threshold
    }

    hmap_filtered = {
        key: value
        for key, value in sorted(
            hmap_filtered.items(), key=lambda item: sum(item[1].values()), reverse=True
        )
    }

    return hmap_filtered


def extract_keywords(df: pd.DataFrame) -> list[list[str]]:
    extracted_keywords = []
    for col in config["ENTITY_COLS"]:
        keywords = (
            df[col]
            .dropna()
            .astype(str)
            .str.split(",")
            .dropna()
            .explode()
            .str.strip()
            .str.lower()
            .unique()
        )
        keywords = keywords[keywords != ""]
        keywords = keywords.tolist()
        for phrase in config["PHRASES"]:
            if all(subphrase in keywords for subphrase in phrase):
                # remove subparts from keywords
                for subphrase in phrase:
                    keywords.remove(subphrase)

                # add the phrase
                keywords.append((" ").join(phrase))

        extracted_keywords.append(keywords)

    return list(set(item for sublist in extracted_keywords for item in sublist))


def compute_frequency_of_keywords(df: pd.DataFrame, keywords: list[str]) -> dict:
    phrases = [" ".join(phrase) for phrase in config["PHRASES"]]
    hmap = {
        keyword: {
            "const_keyword": 0,
            "relation_keyword": 0,
            "runtime_only": 0,
            "not_included": 0,
        }
        for keyword in keywords
    }

    for _, row in df.iterrows():
        sentence = row["sentences"]
        for keyword in keywords:
            if keyword in phrases and re.search(r"\s", keyword):
                pattern = r"\b{}\b".format(r"\s+\w+\s+".join(keyword.split(" ")))
            else:
                pattern = r"\b{}\b".format(re.escape(keyword.lower()))
            if re.search(pattern, str(sentence).lower()):
                flagged = False
                for col in hmap[keyword]:
                    if col == "not_included":
                        continue
                    if not pd.isna(row[col]) and re.search(
                        pattern, str(row[col]).lower()
                    ):
                        hmap[keyword][col] += 1
                        flagged = True
                if not flagged:
                    hmap[keyword]["not_included"] += 1

    return hmap


def compute_similarity(hmap1, hmap2):
    dotprods = []
    ## cosine sim (formula used below): https://miro.medium.com/v2/resize:fit:720/format:webp/1*LfW66-WsYkFqWc4XYJbEJg.png
    wf1 = {}
    wf2 = {}
    for word, freq in hmap1.items():
        wf1[word] = sum(freq.values())

    for word, freq in hmap2.items():
        wf2[word] = sum(freq.values())

    terms = set(wf1).union(wf2)
    for k in terms:
        dotprods.append(
            (k, wf1.get(k, 0) * wf2.get(k, 0))
        )  # contribute towards score only if both specs have it
    mag1 = math.sqrt(sum(wf1.get(k, 0) ** 2 for k in terms))
    mag2 = math.sqrt(sum(wf2.get(k, 0) ** 2 for k in terms))
    dotprods.sort(key=lambda x: x[1], reverse=True)
    cosine_sim = round(sum(dotprod for term, dotprod in dotprods) / (mag1 * mag2), 3)
    return [
        term for term, dotprod in dotprods[:3]
    ], cosine_sim  # normalize by product of sizes


def combine_all_dataframes(
    *, all_specs: dict[str, str], filter_samples: bool, only_rules: bool
) -> list[tuple]:
    dataframes = []
    for spec, spec_file in all_specs.items():
        if spec != "All":
            dataframes.append(
                (
                    spec,
                    read(
                        filename=spec_file,
                        filter_samples=filter_samples,
                        only_rules=only_rules,
                    ),
                )
            )
    return dataframes


def read_entities(df_train, size=None):
    if size == None:
        size = len(df_train)
    sentencesID = df_train[config["Constants"]["Columns"]["SentenceID"]][:size]
    sentences = df_train[config["Constants"]["Columns"]["Sentence"]][:size]
    exact_keywords = make_array(
        df_train[config["Constants"]["Columns"]["InformationModel"]]
    )[:size]
    relational_keywords = make_array(
        df_train[config["Constants"]["Columns"]["Relation"]]
    )[:size]
    constraint_keywords = make_array(
        df_train[config["Constants"]["Columns"]["Constraint"]]
    )[:size]
    numbers = make_array(df_train[config["Constants"]["Columns"]["Numbers"]])[:size]
    quotes = make_array(df_train[config["Constants"]["Columns"]["Quotes"]])[:size]
    runtime_only = make_array(df_train[config["Constants"]["Columns"]["RuntimeOnly"]])[:size]

    return (
        sentencesID,
        sentences,
        exact_keywords,
        relational_keywords,
        constraint_keywords,
        quotes,
        numbers,
        runtime_only,
    )


def make_array(col):
    g_arr = []
    for item in col:
        tmp1 = (
            str(item)
            .replace("'", "")
            .replace('"', "")
            .replace("]", "")
            .replace("[", "")
            .replace("{", "")
            .replace("}", "")
            .split(",")
        )
        arr = []
        if tmp1 != [""] and tmp1 != ["nan"]:
            for i in tmp1:
                arr.append(i.strip())
                arr = list(np.unique(arr))

        g_arr.append(arr)

    return g_arr


def read_file_detailed(df_train):
    __init()
    if "Remarks" in df_train.columns:
        for i in df_train.index:
            if (
                df_train["Remarks"][i] == "Invalid_sentence"
                or df_train["Remarks"][i] == "invalid_sentence"
            ):
                df_train = df_train.drop([i])
    if config["Constants"]["Columns"]["BinaryCls"] in df_train:
        if "end" in df_train[config["Constants"]["Columns"]["BinaryCls"]].values:
            end_idx = df_train.index[
                df_train[config["Constants"]["Columns"]["BinaryCls"]] == "end"
            ].tolist()
            df_train = df_train.loc[: end_idx[0]]
            size = len(df_train)

        else:
            size = len(df_train)
    else:
        size = len(df_train)
    if config["Constants"]["Columns"]["BinaryCls"] in df_train:
        labels = [
            1 if x == "y" else 0
            for x in df_train[config["Constants"]["Columns"]["BinaryCls"]]
        ][:size]

    else:
        labels = []

    (
        sentencesID,
        sentences,
        exact_keywords,
        relational_keywords,
        constraint_keywords,
        quotes,
        numbers,
        runtime_only,
    ) = read_entities(df_train, size)
    return (
        sentencesID,
        sentences,
        exact_keywords,
        relational_keywords,
        constraint_keywords,
        quotes,
        numbers,
        runtime_only,
        labels,
        df_train,
    )


def get_features(
    sentences, exact_keywords, relational_keywords, constraint_keywords, quotes, numbers, runtime_only
):
    features = []
    for i in range(len(sentences)):
        tmp = [
            len(exact_keywords[i]),
            len(relational_keywords[i]),
            len(constraint_keywords[i]),
            len(quotes[i]),
            len(numbers[i]),
            len(runtime_only[i])
        ]
        features.append(tmp)
    return features
