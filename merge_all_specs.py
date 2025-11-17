import os
import pandas as pd
from util import load_config

config = load_config()


def merge_all_specs(data_folder) -> None:
    def read_excel(
        data_folder,
        filename,
        sheet_name=config["SHEET_NAME"],
        col_names=config["COL_NAMES"],
    ) -> pd.DataFrame:
        df = pd.read_excel(
            os.path.join(data_folder, filename),
            sheet_name=sheet_name,
            names=col_names,
            usecols="A:L",
            na_values=["NA"],
        )
        df["title"] = filename
        return df

    all_specs = []
    with os.scandir(data_folder) as it:
        for entry in it:
            if entry.name.endswith(".xlsx"):
                all_specs.append(
                    read_excel(data_folder=data_folder, filename=entry.name)
                )

    all_specs = pd.concat(all_specs, ignore_index=True)
    all_specs = all_specs[all_specs.iloc[:, 4].isin(["y", "Y", "n", "N"])]
    all_specs["id"] = range(1, len(all_specs) + 1)
    all_specs.to_excel(
        f"{data_folder}/{config['FILENAME']}",
        index=False,
        sheet_name=config["SHEET_NAME"],
    )
    print(f"Generated {data_folder}{config['FILENAME']}. Done!")


if __name__ == "__main__":
    merge_all_specs(data_folder=config["DATA_FOLDER"])
