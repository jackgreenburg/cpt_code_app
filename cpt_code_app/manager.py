import os
import pickle
import pysnooper

from .utils import load_dataset, load_pickles

class DataManager:
    """
    Required info
    Each set of models needs to have their own lists for count_mat, words, X, and y
    - count_mat: training data
    - words: list of all words, corresponds to training data
    - X: all report texts
    - y: matrix indicating all assigned CPT codes per report

    Each individual model must have the model and lists for preds, probas, and
    - model: model
    - pred: all predictions
    - proba: all probabilities

    Remove all later references to `dataset["y"]` or `dataset['y']`
    - `dataset["y"]` is made unecessary by `dataset["dx"]["y"]` and `dataset["total]["y]`
    - Or vice versa, but that remains to be seen

    Goal is .pkl containing:
    - X: All reports
    - y: original assignments
    - count_mat: sparse matrix representening feature appearance
    - words: all words to combine with features
    - best_model(s): 1 or 38 model(s) corresponding to primary code
    - preds/proba (optional): can be generated on running
    - splits/best fold (**optional**): to show if in training or test set
    """
    def __init__(self,):
        self.data = {}
        self.current = None
        self.results = None
#         self.labeledSparseMatrix = None
        self.explainerDict = None
        self.codes = None
        self.allData = None

    @pysnooper.snoop()
    def set(self, name, dataset=None, dx_total=None, path=None):
        if self.current == name:
            return True

        if name in self.data:
            pass
        else:  # data does not already exist, so it must be loaded
            # create entry
            self.data[name] = {}

            # initialize dict to store explainer and shap values so they do not have to be generated again
            self.data[name]["explainerDict"] = {}

            # load pickled models
            self.data[name]["results"] = load_pickles(path)

            if  self.data[name]["results"][0]["best_model"].objective == "multi:softprob":
                self.data[name]["codes"] = ["88302", "88304", "88305", "88307", "88309"]

                # get boolean of if each report is in report
                remove_bool=(dataset['y'][["88302 ","88304 ","88305 ","88307 ","88309 "]].sum(1)==1).values

                # set dataset
                self.data[name]["allData"] = {"words": dataset[dx_total]["words"],
                                              "X": dataset[dx_total]["X"][remove_bool],
                                              "count_mat": dataset[dx_total]["count_mat"][remove_bool],
                                              "y": dataset["y"][["88302 ","88304 ","88305 ","88307 ","88309 "]][remove_bool]}

            elif self.data[name]["results"][0]["best_model"].objective == "binary:logistic":
#                 self.data[name]["codes"] = [code[:-1] for code in dataset["y"].columns.values]
                self.data[name]["codes"] = [result["code"] for result in self.data[name]["results"]]

                # set dataset
                self.data[name]["allData"] = dataset[dx_total]
                self.data[name]["allData"]["y"] = dataset["y"]
            else:
                raise Exception("Unhandled model edgecase.")

        # now point to values in dict
        self.current = name
        self.results = self.data[name]["results"]
#         self.labeledSparseMatrix = self.data[name]["labeledSparseMatrix"]
        self.explainerDict = self.data[name]["explainerDict"]
        self.codes = self.data[name]["codes"]
        self.allData = self.data[name]["allData"]


if __name__ == "__main__":
    print("manager.py main.")
