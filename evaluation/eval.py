import json
import os
import pandas as pd
from evaluation_functions import jec_ac, jec_kd, cjft, ydlj, ftcs, jdzy, jetq, ljp_accusation, ljp_article, ljp_imprison, wbfl, xxcq, flzx, wsjd, yqzy, lblj, zxfl, sjjc
import sys
import argparse
import pandas as pd
import numpy as np

def read_json(input_file):
    # load the json file
    with open(input_file, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    dict_size = len(data_dict)
    new_data_dict = []
    for i in range(dict_size):
        example = data_dict[str(i)]
        new_data_dict.append(example)

    return new_data_dict

def get_score(matrix_df, model_name, task_name):
    try: 
        if np.isnan(matrix_df.loc[model_name, task_name]): return None
        return matrix_df.loc[model_name, task_name]
    except KeyError:
        return None

def set_score(matrix_df, model_name, task_name, score):
    if matrix_df.empty and len(matrix_df.columns) == 0:
        matrix_df[task_name] = pd.Series(dtype='float64')
    if model_name not in matrix_df.index:
        matrix_df.loc[model_name] = np.nan
    if task_name not in matrix_df.columns:
        matrix_df[task_name] = np.nan
    matrix_df.loc[model_name, task_name] = score

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="zero_shot", help="evaluation type")
    args = parser.parse_args(argv)
    funct_dict = {"3-6": jec_ac.compute_jec_ac,
                  "3-2": cjft.compute_cjft,
                  "3-8": flzx.compute_flzx,
                  "3-7": jetq.compute_jetq,
                  "3-3": ljp_accusation.compute_ljp_accusation,
                  "3-1": ljp_article.compute_ljp_article,
                  "3-4": ljp_imprison.compute_ljp_imprison,
                  "3-5": ljp_imprison.compute_ljp_imprison
    }
    
    # Load existing score and abstention data
    try:
        score_df = pd.read_csv(f"../{args.type}_score.csv", index_col=0)
    except:
        score_df = pd.DataFrame(
            index=pd.Index([], name='model_name'),
            columns=pd.Index(["3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8"], name='task'),
            dtype=float
        )
    try:
        abstention_df = pd.read_csv(f"../{args.type}_abstention.csv", index_col=0)
    except:
        abstention_df = pd.DataFrame(
            index=pd.Index([], name='model_name'),
            columns=pd.Index(["3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8"], name='task'),
            dtype=float
        )
    
    model_folders = os.listdir(f"../predictions/{args.type}/")
    for model_name in model_folders:
        if model_name.startswith("."): continue
        model_folder_dir = os.path.join(f"../predictions/{args.type}", model_name)
        if not os.path.isdir(model_folder_dir): continue
        print(f"*** Evaluating System: {model_folder_dir} ***")
        dataset_files = os.listdir(model_folder_dir)
        for dataset_file in dataset_files:
            datafile_name = dataset_file.split(".")[0]
            if get_score(score_df, model_name, datafile_name) is not None: continue
            input_file = os.path.join(model_folder_dir, dataset_file)
            if datafile_name not in funct_dict: continue
            print(f"Processing {datafile_name}:")
            data_dict = read_json(input_file)
            score_function = funct_dict[datafile_name]
            score = score_function(data_dict)
            print(f"Score of {datafile_name}: {score}")
            abstention_rate = score["abstention_rate"] if "abstention_rate" in score else 0
            set_score(score_df, model_name, datafile_name, score["score"])
            set_score(abstention_df, model_name, datafile_name, abstention_rate)

            score_df.to_csv(f"../{args.type}_score.csv", index=True)
            abstention_df.to_csv(f"../{args.type}_abstention.csv", index=True)

if __name__ == '__main__':
    main(sys.argv[1:])
