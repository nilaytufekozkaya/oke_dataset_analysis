# OKE Dataset Analysis
This repository provides four different analyses for the OKE Dataset introduced in Tufek Ozkaya et al. (2023). 

# Motivation
The motivation of the analyses described below is to:
1. Detect inconsistencies among annotators of the dataset and highlight consistency of the keywords in [consistency analysis](#1-keyword-analysis)
2. See the distributions of the annotated entities among the companion specifications of OPC UA and extract correlation between these specifications in [entity distribution analysis](#2-entity-distribution-analysis)
3. Just by using the annotated entities, and their frequencies, generating heatmap(s) to compute similarity score between companion specifications in [common entity analysis](#3-common-entity-analysis)


# Dataset
The OPC UA Knowledge Extraction (OKE) Dataset is a dataset specifically created for sentence classification, named entity recognition and disambiguation (NERD), and entity linking. To learn more about the dataset and download it, please visit this [link](https://zenodo.org/records/10284578) and get the latest version of it. After downloading the dataset, please make sure to place the files in the `./oke_dataset/` directory before running the analysis scripts.

The files should appear in the following hierarchy:

```.
└── oke-dataset-analysis/
    ├── oke_dataset/
    │   ├── csv/
    │   │   ├── AutoId.csv
    │   │   └── ...
    │   └── excel/
    │       ├── AutoId.xlsx
    │       └── ...
    ├── output
    └── ...
```

## Prerequisites

Make sure that you have `python >= 3.9` and install the `requirements.txt`.

# Analysis

## 1. Keyword Analysis

This analysis will produce 4 different files of charts.
* most used keywords
* most used keywords, filtered by inclusion*
* keywords assigned to more than 2 categories
* most conflicting keyword**

*inclusion: Given a keyword, assume that it is seen `n` times in the specification. if it is annotated only a few times, and not annotated for the remaining. it will be discarded. this margin is determined by the `threshold` parameter.

**Here, most conflicting means that if keyword is assinged to more than 2 categories and those categories' rate is close to each other. for example, for a keyword annotated 10 times, if it is annotated 5 times as `runtime-only`, and 5 times as `information model keyword`. this is one of the most conflicting situations.

- IMPORTANT: Before running the analysis, please run the helper script as: 
    ```bash 
    python merge_all_specs.py
    ```
which will combine all the samples in all the files into one file. Afterwards one is free to run any experiment described below.
#### Running commands:
- Running all sentences from all Excel sheets as follows. This will take some time (up to 5 mins depending on the hardware).
   
    ```bash 
    python keyword_analysis.py
    ```

* Select a custom specification (default = all): 
    ```bash 
    python keyword_analysis.py -s autoid
    ```
    specification options: "all", "autoid", "iolink", "isa95", "machinetools", "mv1ccm", "mv2amcm", "packml", "padim", "profinet", "robotics", "uafx", "weihenstephan"


* Select the sentence type:
    * for only rule sentences: 
        ```bash 
        python keyword_analysis.py -s autoid -f --only_rule_sentences
        ```

    * for only non-rule sentences:
        ```bash  
        python keyword_analysis.py -s autoid -f --only_non_rule_sentences
        ```
    the default is all sentences.

* Overriding the inclusion threshold for keywords (default=0.9)*: 
    ```bash 
    python keyword_analysis.py -s autoid -f --only_non_rule_sentences --threshold 0.8
    ```


* Setting output path:
    ```bash 
    python keyword_analysis.py -s autoid -o test_output/
    ```

    *decreasing threshold will likely to lead to less keywords to be used during computation.

The output will be generated under ./output/keyword_analysis/ folder.

## 2. Entity Distribution Analysis
In order to run entity distribution analysis, the following commands can be called. 

#### Computation Steps:
1. Calculate the entity distribution histogram for each companion specification.
2. Normalize these histogram vectors to have unit length.
3. Take cosine similarity (dot product) between these vectors.
4. Collect the pair-wise cosine similarities in a heatmap.
5. Do it for rule and non-rule sentences seperately

#### Running commands:
* for all entity categories: 
    ```bash 
    python entity_distribution_analysis.py --entity_name all
    ```

* for a specfict entity category call to generate heatmap: 
    ```bash 
    python entity_distribution_analysis.py --entity_name information_model
    ```

#### Output:
The output will be generated under ./output/entity_distribution_analysis/ folder. There will be two sub folders generated for this purpose: 
1.  ./output/entity_distribution_analysis/distribution_barchart/
this shows the distribution charts of each entity of each datasheet
1. ./output/entity_distribution_analysis/distribution_heatmap/
this involves the heatmaps of the distribution correlation of each datasheet based on different entity categories. 

## 3. Common Entity Analysis
This analysis will produce four files, each of which contains a heatmap indicating the similarities of specifications based on each of the four following columns in the dataset. The columns are: "Information Model Keywords", "Constraint Keywords", "Relation Keywords" and "Runtime only".
#### Computation Steps:
1. Collect all the keywords from specified column across all the companion specs.
2. For each comp spec, calculate a keyword frequency histogram over the set obtained in step #1.
3. Normalize these histogram vectors to have unit length.
4. Take cosine similarity (dot product) between these vectors.
5. Collect the pair-wise cosine similarities in a heatmap.
#### Running commands:
* all sentences (unfiltered): 
    ```bash 
    python common_entity_analysis.py
    ```

* only rule sentences: 
    ```bash 
    python common_entity_analysis.py -f --only_rule_sentences
    ```

* only non-rule sentences: 
    ```bash 
    python common_entity_analysis.py -f --only_non_rule_sentences
    ```

#### Output:
The output will be generated under ./output/common_entity_analysis/ folder.

## 4. General Specification Analysis
This analysis will produce a single file that contains a heatmap indicating the similarities of specifications based on some columns in the dataset. 

#### Running commands:
* all sentences (unfiltered): 
    ```bash 
    python spec_analysis.py
    ```

* only rule sentences: 
    ```bash 
    python spec_analysis.py -f --only_rule_sentences
    ```
* only non-rule sentences: 
    ```bash 
    python spec_analysis.py -f --only_non_rule_sentences
    ```

* overriding inclusion threshold for keywords (default=0.9)*:
    ```bash 
    python spec_analysis.py -f --only_rule_sentences --threshold 0.8
    ```

*decreasing threshold will likely to lead to less keywords to be used during computation

#### Output:
The output will be generated under ./output/specification_analysis/ folder.

# Dataset Citation

    @dataset{tufek_ozkaya_2023_10284578,
        author       = {Tufek Ozkaya, Nilay},
        title        = {OPC UA Knowledge Extraction (OKE) Dataset},
        month        = dec,
        year         = 2023,
        publisher    = {Zenodo},
        doi          = {10.5281/zenodo.10284577},
        url          = {https://doi.org/10.5281/zenodo.10284577}
    }

# Authors
Nilay Tüfek Özkaya (nilay.tuefek-oezkaya@siemens.com)
Valentin Philipp (valentin.just@tuwien.ac.at)
Berkay Ugur (berkaysenocak@gmail.com)
Tathagata Bandyopadhyay (tathagata.bandyopadhyay@siemens.com)
