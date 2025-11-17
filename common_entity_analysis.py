import argparse
import os
from collections import Counter
from typing import List, Dict
from matplotlib.pyplot import close
import util
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


headers = ["im_keywords", "const_keyword", "relation_keyword", "runtime_only"]


def save_heatmap(
    heatmap: np.ndarray, labels: List[str], feature_name: str, args_: argparse.Namespace
):
    filename = generate_filename(feature_name, args_)
    mask = np.triu(np.ones_like(heatmap))
    cosine_plot = sns.heatmap(
        heatmap,
        cmap="YlGnBu",
        annot=True,
        mask=mask,
        xticklabels=labels,
        yticklabels=labels,
    )
    fig = cosine_plot.get_figure()
    fig.savefig(filename, bbox_inches="tight")
    close(fig)
    print(f"Image saved to {filename} successfully!")


def compute_cosine_maps(
    df_list: List[pd.DataFrame], corpus_mapping: List[Dict]
) -> np.ndarray:
    cosine_map = np.empty((len(df_list), len(df_list), 4), dtype=np.float32)
    for i, df1 in enumerate(df_list):
        for j, df2 in enumerate(df_list):
            if j <= i:  # compute only the lower diagonal matrix
                for k in [0, 1, 2, 3]:
                    s1 = flatten_series(df1.iloc[:, k])
                    s2 = flatten_series(df2.iloc[:, k])
                    cosine_map[i, j, k] = cal_cosine_sim(s1, s2, corpus_mapping[k])
    return cosine_map


def flatten_series(series, filter_none=True):
    fl = []
    for data in series:
        for d in str(data).split(","):
            fl.append(d.lower().replace(" ", ""))
    if filter_none:
        fl = list(filter(None, fl))
    return fl


def generate_corpus_mappings(df_list: List[pd.DataFrame]) -> List[Dict]:
    corpus_mappings = []
    im_words = []
    constraint_words = []
    relation_words = []
    runtime_words = []
    for df in df_list:
        im_words += flatten_series(df.iloc[:, 0])
        constraint_words += flatten_series(df.iloc[:, 1])
        relation_words += flatten_series(df.iloc[:, 2])
        runtime_words += flatten_series(df.iloc[:, 3])
    im_words = sorted(list(set(im_words)))
    constraint_words = sorted(list(set(constraint_words)))
    relation_words = sorted(list(set(relation_words)))
    runtime_words = sorted(list(set(runtime_words)))
    corpus_mappings.append({k: v for v, k in enumerate(im_words)})
    corpus_mappings.append({k: v for v, k in enumerate(constraint_words)})
    corpus_mappings.append({k: v for v, k in enumerate(relation_words)})
    corpus_mappings.append({k: v for v, k in enumerate(runtime_words)})
    return corpus_mappings


def cal_cosine_sim(
    list1: List[str],
    list2: List[str],
    corpus_mapping: Dict,
    round_op=True,
    round_decimal=2,
):
    vec1 = np.zeros(len(corpus_mapping))
    vec2 = np.zeros(len(corpus_mapping))
    count1 = Counter(list1)
    count2 = Counter(list2)
    for (k1, v1), (k2, v2) in zip(count1.items(), count2.items()):
        vec1[corpus_mapping.get(k1)] = v1
        vec2[corpus_mapping.get(k2)] = v2
    vec1 = vec1 / (np.linalg.norm(vec1) + 1e-16)
    vec2 = vec2 / (np.linalg.norm(vec2) + 1e-16)
    cos_sim = np.dot(vec1, vec2)
    if round_op:
        return np.round(cos_sim, round_decimal)
    else:
        return cos_sim


def generate_filename(feature_name, args_):
    filename = "common_entity_analysis"
    if args_.filter_samples:
        filename += "_filtered"
        if args_.only_rules is not None:
            if args_.only_rules:
                filename += "_rules"
            else:
                filename += "_non_rules"
    else:
        filename += "_unfiltered"

    filename += f"_{feature_name}.png"

    return os.path.join(args_.output_path, "common_entity_analysis/", filename)


def run(args_: argparse.Namespace):
    config = util.load_config()
    all_spec_names = config["FILENAMES"]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    target = os.path.join(args.output_path, "common_entity_analysis/")
    if not os.path.exists(target):
        os.makedirs(target)

    dataframes = util.combine_all_dataframes(
        all_specs=all_spec_names,
        filter_samples=args_.filter_samples,
        only_rules=args_.only_rules,
    )

    # drop unnecessary columns and prepare df list
    data_frame_list = []
    comp_spec_list = []
    for spec, df in dataframes:
        df.drop(df.columns.difference(headers), 1, inplace=True)
        data_frame_list.append(df)
        comp_spec_list.append(spec)

    # actual heatmap computation logic
    corpus_mappings = generate_corpus_mappings(data_frame_list)
    heat_maps = compute_cosine_maps(data_frame_list, corpus_mappings)

    # splitting the heatmaps across headers and saving logic
    for idx, header in enumerate(headers):
        map_ = heat_maps[:, :, idx]
        save_heatmap(map_, comp_spec_list, header, args_)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser("")

    _parser.add_argument(
        "-f",
        "--filter_samples",
        help="flag indicating if filtering is active",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    sentence_filter_group = _parser.add_mutually_exclusive_group()

    sentence_filter_group.add_argument(
        "--only_rule_sentences",
        action=argparse.BooleanOptionalAction,
        help="filter by only rule sentences",
        default=False,
    )

    sentence_filter_group.add_argument(
        "--only_non_rule_sentences",
        action=argparse.BooleanOptionalAction,
        help="filter by only non-rule sentences",
        default=False,
    )

    _parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="output directory where file will be saved in",
        default="output/",
    )

    args = _parser.parse_args()

    if args.filter_samples and not (
        args.only_rule_sentences or args.only_non_rule_sentences
    ):
        _parser.error(
            "When using -f, either --only_rule_sentences or --only_non_rule_sentences is required."
        )

    # bool
    args.only_rules = args.filter_samples and args.only_rule_sentences

    run(args)
