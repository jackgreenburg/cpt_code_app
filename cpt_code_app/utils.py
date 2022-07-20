from typing import List
import pickle
import os
import pandas as pd

def load_dataset(data_location: str="/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/cpt_code_app_data/data/paper_official_dict.pkl"):
    with open(data_location,'rb') as f:
        return pickle.load(f)
    
def load_pickles(path: str) -> List:
    """
    Load all pickles in directory outer_dir\inner_dir
    So make sure you want all pickles in outer_dir\inner_dir
    """
    print(f"Opening {path} -- ", end="")
    
    names = os.listdir(path)
    results = []
    for n in names:
        if n[-4:] == ".pkl":
            with open(os.path.join(path, n),'rb') as infile:
                 results.append(pickle.load(infile))
            print("|", end="") 
    print(" --> done")
    return results

def report_to_str(path_text, markdown=False):
    # note about implementation here, should just always pass pandas data series...
    # fetch the text of the pathology report, formatting if necessary
    if isinstance(path_text, str):
        return path_text
    elif isinstance(path_text, pd.core.series.Series):
        text = ""
        for val in path_text.index:
            # check if section empty --> if empty do not include header
            if not pd.isna(path_text[val]):
                if markdown:
                    text += f"{val}\n\n"  # newline
                    text += f"{path_text[val]}\n\n"  # add newline at end
                else:
                    text += f"!{val.replace(' ', '_')}! "  # can't use a space, have to split with something else
                    text += f"{path_text[val]} "  # add space at end
        return text
    else:
        raise Exception(f"Not supported with reports of type: {type(path_text)}")
