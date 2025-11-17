import argparse
import os
import util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_comparison_matrix(filtered_hashmaps: list[dict]) -> list:
    # comparison matrix
    comparison_matrix = [
        [0 for _ in range(len(filtered_hashmaps))]
        for _ in range(len(filtered_hashmaps))
    ]

    # one willing to see what the most contributing words towards
    # the similarity can simply uncomment the following lines and line 35
    # then return it and use the dataframe.

    # most_contributing_words = [
    #     [0 for _ in range(len(filtered_hashmaps))]
    #     for _ in range(len(filtered_hashmaps))
    # ]
    for i in range(len(filtered_hashmaps)):
        spec_i, hashmap_i = filtered_hashmaps[i]
        for ii in range(i + 1, len(filtered_hashmaps)):
            spec_ii, hashmap_ii = filtered_hashmaps[ii]

            local_most_contributing_words, similarity = util.compute_similarity(
                hmap1=hashmap_i, hmap2=hashmap_ii
            )
            comparison_matrix[i][ii] = similarity

            # most_contributing_words[i][j] = local_most_contributing_words

    return comparison_matrix


def save_comparison_heatmap(comparison_df, file_path):
    mask = np.triu(np.ones(comparison_df.T.shape), k=1)
    plt.rcParams["figure.figsize"] = (10, 9)
    fig = plt.gcf()
    ax = sns.heatmap(
        comparison_df.T,
        xticklabels=comparison_df.columns.values,
        yticklabels=comparison_df.columns.values,
        square=True,
        annot=True,
        mask=mask,
    )

    plt.xticks(rotation=90)

    plt.savefig(file_path)
    print(f"Image saved to {file_path} successfully!")


def generate_filename(filename, args):
    filename = "specification_analysis"
    if args.filter_samples:
        filename += "_filtered"
        if args.only_rules is not None:
            if args.only_rules:
                filename += "_rules"
            else:
                filename += "_non_rules"
    else:
        filename += "_unfiltered"

    filename += f"_threshold_{args.threshold}.png"

    return filename


def run(args: argparse.Namespace):
    config = util.load_config()
    all_spec_names = config["FILENAMES"]

    directory = os.path.dirname(args.output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = os.path.join(args.output_path, "specification_analysis/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    dataframes = util.combine_all_dataframes(
        all_specs=all_spec_names,
        filter_samples=args.filter_samples,
        only_rules=args.only_rules,
    )

    filtered_hashmaps = []

    for df_index in range(len(dataframes)):
        spec, df = dataframes[df_index]
        keywords = util.extract_keywords(df=df)
        hashmap = util.compute_frequency_of_keywords(df=df, keywords=keywords)
        attributes = util.extract_attributes(hmap=hashmap)
        filtered_hashmap = util.filter_hmap(
            hmap=hashmap, sum_of_attributes=attributes, threshold=args.threshold
        )
        filtered_hashmaps.append((spec, filtered_hashmap))

    comparison_matrix = generate_comparison_matrix(filtered_hashmaps)

    comparison_df = pd.DataFrame(
        comparison_matrix,
        columns=[name for name in all_spec_names.keys() if name != "All"],
        dtype=float,
    )

    filename = generate_filename("specification_analysis", args)

    target = os.path.join(args.output_path, "specification_analysis/", filename)

    save_comparison_heatmap(comparison_df=comparison_df, file_path=target)


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
        "-t", "--threshold", type=float, help="inclusion threshold", default="0.9"
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
