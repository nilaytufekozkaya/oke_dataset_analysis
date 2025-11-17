import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import util


config = util.load_config()
plt.rcParams["figure.figsize"] = (14, 13)


class SentenceGroup:
    def __init__(self) -> None:
        self.IM_kws = []
        self.rel_kws = []
        self.const_kws = []
        self.quot_kws = []
        self.num_kws = []
        self.runtime_only_kws = []

    def get_features_each(self, r_sentences):
        IM_kws = []
        rel_kws = []
        const_kws = []
        quot_kws = []
        num_kws = []
        runtime_only_kws = []
        for item in r_sentences:
            IM_kws.append(item[0])
            rel_kws.append(item[1])
            const_kws.append(item[2])
            quot_kws.append(item[3])
            num_kws.append(item[4])
            runtime_only_kws.append(item[5])
        self.IM_kws = IM_kws
        self.rel_kws = rel_kws
        self.const_kws = const_kws
        self.quot_kws = quot_kws
        self.num_kws = num_kws
        self.runtime_only_kws = runtime_only_kws


def get_file_names(input_files):
    # get p and n sentences:
    file_names_ = [x.split("/")[-1] for x in input_files]
    file_names = [x.split(".")[0] for x in file_names_]
    return file_names

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_dist(sp, sn, input_file, output_dir):
    dent = False
    input_file_name = os.path.splitext(os.path.basename(input_file))[0]
    formatted_filename = get_key_of_dict(input_file_name)

    # 3 satır x 2 sütun
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), dpi=100)
    fig.suptitle(formatted_filename)

    labels = ["rule sentence", "non-rule sentence"]
    colors = ["red", "tan"]
    bins = np.arange(-1, 10) + 0.5  # -0.5..9.5 → tam sayılara merkezli

    # (veri, başlık) çiftleri
    panels = [
        ([sp.IM_kws,           sn.IM_kws],           "IM keywords distribution"),
        ([sp.rel_kws,          sn.rel_kws],          "Relation keywords distribution"),
        ([sp.const_kws,        sn.const_kws],        "Constraint keywords distribution"),
        ([sp.quot_kws,         sn.quot_kws],         "Quote keywords distribution"),
        ([sp.num_kws,          sn.num_kws],          "Numbers keywords distribution"),
        ([sp.runtime_only_kws, sn.runtime_only_kws], "Runtime-only keywords distribution"),
    ]

    # Her paneli çiz
    for ax, (data, title) in zip(axes.flatten(), panels):
        ax.hist(
            data,
            density=dent,
            histtype="bar",
            color=colors,
            label=labels,
            bins=bins,
        )
        ax.set_title(title)
        ax.legend(prop={"size": 9})
        ax.locator_params(integer=True)

    # Kenar boşlukları
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92, wspace=0.25, hspace=0.35)

    # Kaydet
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{formatted_filename}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Image saved to {out_path} successfully!")



def plot_dist_old(sp, sn, input_file, output_dir):
    dent = False
    input_file_name = input_file.split("/")[-1].split(".")[0]
    fig, ((ax0, ax1, ax2, ax3, ax4, ax5)) = plt.subplots(
        nrows=6,
        ncols=1,
        figsize=(8, 12),
        dpi=80,
    )
    formatted_filename = get_key_of_dict(input_file_name)
    fig.suptitle(formatted_filename)
    labels = ["rule sentence", "non-rule sentence"]
    colors = ["red", "tan"]
    ax0.hist(
        [sp.IM_kws, sn.IM_kws],
        density=dent,
        histtype="bar",
        color=colors,
        label=labels,
        bins=np.arange(-1, 10) + 0.5,
    )
    ax0.legend(prop={"size": 10})
    ax0.set_title("IM keywords distribution")
    ax0.locator_params(integer=True)

    ax1.hist(
        [sp.rel_kws, sn.rel_kws],
        density=dent,
        histtype="bar",
        color=colors,
        label=labels,
        bins=np.arange(-1, 10) + 0.5,
    )
    ax1.legend(prop={"size": 10})
    ax1.set_title("Relation keywords distribution")
    ax1.locator_params(integer=True)

    ax2.hist(
        [sp.const_kws, sn.const_kws],
        density=dent,
        histtype="bar",
        color=colors,
        label=labels,
        bins=np.arange(-1, 10) + 0.5,
    )
    ax2.legend(prop={"size": 10})
    ax2.set_title("Contraint keywords distribution")
    ax2.locator_params(integer=True)

    ax3.hist(
        [sp.quot_kws, sn.quot_kws],
        density=dent,
        histtype="bar",
        color=colors,
        label=labels,
        bins=np.arange(-1, 10) + 0.5,
    )
    ax3.legend(prop={"size": 10})
    ax3.set_title("Quote keywords distribution")
    ax3.locator_params(integer=True)

    ax4.hist(
        [sp.num_kws, sn.num_kws],
        density=dent,
        histtype="bar",
        color=colors,
        label=labels,
        bins=np.arange(-1, 10) + 0.5,
    )
    ax4.legend(prop={"size": 10})
    ax4.set_title("Numbers keywords distribution")
    ax4.locator_params(integer=True)

    ax5.hist(
        [sp.runtime_only_kws, sn.runtime_only_kws],
        density=dent,
        histtype="bar",
        color=colors,
        label=labels,
        bins=np.arange(-1, 10) + 0.5,
    )
    ax5.legend(prop={"size": 10})
    ax5.set_title("Numbers keywords distribution")
    ax5.locator_params(integer=True)

    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    plt.savefig(f"{output_dir}{formatted_filename}.png")
    plt.close()


def find_dists(input_file):
    df = pd.read_excel(input_file, config["SHEET_NAME"])

    (
        _,
        sentences,
        exact_keywords,
        relational_keywords,
        constraint_keywords,
        quotes,
        numbers,
        runtime_only,
        labels_bn,
        df,
    ) = util.read_file_detailed(df)

    features = util.get_features(
        sentences,
        exact_keywords,
        relational_keywords,
        constraint_keywords,
        quotes,
        numbers,
        runtime_only
    )

    r_sentences = []
    nr_sentences = []
    for i in range(len(labels_bn)):
        if labels_bn[i] == 1:
            r_sentences.append(features[i])
        else:
            nr_sentences.append(features[i])

    sp = SentenceGroup()
    sp.get_features_each(r_sentences)

    sn = SentenceGroup()
    sn.get_features_each(nr_sentences)
    return sp, sn


def run_plot(input_file, output_dir):
    sp, sn = find_dists(input_file)
    plot_dist(sp, sn, input_file, output_dir)


def get_heap_elements(dict, file_names):
    ims = [0] * len(file_names)
    consts = [0] * len(file_names)
    relations = [0] * len(file_names)
    quots = [0] * len(file_names)
    nums = [0] * len(file_names)
    runtime_onlys = [0] * len(file_names)
    cnt = 0
    for file_name1 in file_names:
        im = dict[file_name1].IM_kws
        i = np.histogram(im, density=True, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8])[0]
        ims[cnt] = i

        rel = dict[file_name1].rel_kws
        r = np.histogram(rel, density=True, bins=[0, 1, 2, 3, 4, 5])[0]
        relations[cnt] = r

        cons = dict[file_name1].const_kws
        c = np.histogram(cons, density=True, bins=[0, 1, 2, 3, 4, 5, 6])[0]
        consts[cnt] = c

        quo = dict[file_name1].quot_kws
        q = np.histogram(quo, density=True, bins=[0, 1, 2, 3])[0]
        quots[cnt] = q

        num = dict[file_name1].num_kws
        n = np.histogram(num, density=True, bins=[0, 1, 2, 3])[0]
        nums[cnt] = n

        runtime_only = dict[file_name1].runtime_only_kws
        n = np.histogram(runtime_only, density=True, bins=[0, 1, 2, 3])[0]
        runtime_onlys[cnt] = n

        cnt += 1

    return ims, relations, consts, quots, nums, runtime_onlys


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def draw_heatmap(input_files, output_dir, dict_entity, entity_name, font_scale=1):
    # --- veriyi hazırla ---
    dict_p, dict_n = {}, {}
    file_names = get_file_names(input_files)
    for i, input_file in enumerate(input_files):
        sp, sn = find_dists(input_file)
        dict_p[file_names[i]] = sp
        dict_n[file_names[i]] = sn

    formatted_filenames = [get_key_of_dict(fn) for fn in file_names]

    # --- çizim stili: fontları 2x büyüt ---
    prev = sns.axes_style()
    prev_ctx = sns.plotting_context()
    sns.set_theme(style="white")
    sns.set_context("talk", font_scale=font_scale)  # tüm metinler ~2×

    try:
        # ---------- RULE ----------
        res_arr = get_heap_elements(dict_p, file_names)
        ent = res_arr[dict_entity[entity_name]]
        sim = cosine_similarity(ent)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=0)  # üst üçgen + diyagonal gizle
        sns.heatmap(
            sim,
            ax=ax,
            mask=mask,
            annot=True,
            fmt=".2f",
            xticklabels=formatted_filenames,
            yticklabels=formatted_filenames,
            square=True,
            cbar_kws={"shrink": 0.85},
            annot_kws={"size": int(8 * font_scale)},  # hücre yazıları 2×
        )
        ax.set_title("Rule Sentences", pad=12)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{entity_name}_rule_sentence_heatmap.png"),
                    bbox_inches="tight", dpi=300)
        plt.close(fig)

        # ---------- NON-RULE ----------
        res_arr = get_heap_elements(dict_n, file_names)
        ent = res_arr[dict_entity[entity_name]]
        sim = cosine_similarity(ent)

        fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=150)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=0)
        sns.heatmap(
            sim,
            ax=ax2,
            mask=mask,
            annot=True,
            fmt=".2f",
            xticklabels=formatted_filenames,
            yticklabels=formatted_filenames,
            square=True,
            cbar_kws={"shrink": 0.85},
            annot_kws={"size": int(8 * font_scale)},
        )
        ax2.set_title("Non-rule Sentences", pad=12)
        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, f"{entity_name}_non_rule_sentence_heatmap.png"),
                     bbox_inches="tight", dpi=300)
        plt.close(fig2)

    finally:
        # Stil ve context'i eski haline getir
        sns.set_style(prev)
        sns.set_context(rc=prev_ctx)


def draw_heatmap_old(input_files, output_dir, dict_entity, entity_name):
    dict_p = {}
    dict_n = {}
    i = 0
    file_names = get_file_names(input_files)
    for input_file in input_files:
        sp, sn = find_dists(input_file)
        dict_p[file_names[i]] = sp
        dict_n[file_names[i]] = sn
        i += 1

    formatted_filenames = [get_key_of_dict(filename) for filename in file_names]
    res_arr = get_heap_elements(dict_p, file_names)
    ent = res_arr[dict_entity[entity_name]]
    ax = plt.axes()
    mask = np.triu(np.ones_like(cosine_similarity(ent)))
    cosine_plot = sns.heatmap(
        cosine_similarity(ent),
        annot=True,
        mask=mask,
        xticklabels=formatted_filenames,
        yticklabels=formatted_filenames,
        ax=ax,
        fmt=".2f",
    )
    ax.set_title("Rule Sentences")
    fig = cosine_plot.get_figure()
    fig.savefig(output_dir + entity_name + "_rule_sentence_heatmap.png", dpi=fig.dpi)
    fig.clear()

    res_arr = get_heap_elements(dict_n, file_names)
    ent = res_arr[dict_entity[entity_name]]
    ax = plt.axes()
    mask = np.triu(np.ones_like(cosine_similarity(ent)))
    cosine_plot = sns.heatmap(
        cosine_similarity(ent),
        annot=True,
        mask=mask,
        xticklabels=formatted_filenames,
        yticklabels=formatted_filenames,
        ax=ax,
        fmt=".2f",
    )
    ax.set_title("Non-rule Sentences")
    fig2 = cosine_plot.get_figure()
    fig2.savefig(
        output_dir + entity_name + "_non_rule_sentence_heatmap.png", dpi=fig2.dpi
    )
    fig2.clear()


def get_key_of_dict(value: str) -> str:
    filenames = config.get("FILENAMES", {})
    # -5 to cut '.xlsx'
    return next(
        (k for k, v in filenames.items() if v[:-5].lower() == value.lower()), ""
    )


def run(args: argparse.Namespace):
    input_dir = args.input_path
    output_dir = args.output_path
    entity_name = args.entity_name

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    input_files = [
        os.path.join(config["DATA_FOLDER"], v.lower())
        for k, v in config["FILENAMES"].items()
        if k != "All"  ## all of the specifications but the merged one with key 'All'
    ]

    # run bar chart all
    if entity_name == "all":
        target = os.path.join(output_dir, "distribution_barchart/")
        if not os.path.exists(target):
            os.makedirs(target)
        for input_file in input_files:
            run_plot(input_file, target)

    dict_entity = {
        "information_model": 0,
        "relation": 1,
        "constraint": 2,
        "quotes": 3,
        "number": 4,
        "runtime_only": 5
    }

    target = os.path.join(output_dir, "distribution_heatmap/")
    if not os.path.exists(target):
        os.makedirs(target)

    if entity_name == "all":
        for entity in dict_entity:
            draw_heatmap(
                input_files,
                target,
                dict_entity,
                entity,
            )
    else:
        draw_heatmap(
            input_files,
            target,
            dict_entity,
            entity_name,
        )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser("")

    _parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="input directory where the excel files will be read from",
        default="oke_dataset/excel/",
    )

    _parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="output directory where file will be saved in",
        default="output/entity_distribution_analysis/",
    )

    _parser.add_argument(
        "-e",
        "--entity_name",
        type=str,
        help="the name of the entity to be analyzed",
        default="all",
    )

    args = _parser.parse_args()
    run(args)
