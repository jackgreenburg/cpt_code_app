from typing import List
import pickle
import os

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
