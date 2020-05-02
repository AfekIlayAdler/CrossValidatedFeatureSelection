from pathlib import Path

N_EXPERIMENTS = 15
N_PROCESS = 1

# dataset
EXP_NAME = F"house_grants"
Y_COL_NAME = "Grant.Status"
PROPORTION_NAN_COL_REMOVE = 0.7
COLUMNS_TO_REMOVE = ["Grant.Application.ID","Start.date"]


DEBUG = True
MODELS = {
    'sklearn': ['one_hot', 'mean_imputing'],
    'ours': ['Kfold', 'CartVanilla']}

DATA_PATH = Path(r"C:\Users\afeki\OneDrive\Desktop\code\Final\datasets\melbourne_grants_classification"
                 r"\unimelb_training.csv")


