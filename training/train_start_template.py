import importlib
import os
MODEL_NAME = "STUB_MODEL_NAME"
DATASET_NAME = "STUB_DATASET_NAME"
training_app = importlib.import_module("training.experiments.{}_training".format(MODEL_NAME))
if "STUB" in DATASET_NAME:
    config_file = MODEL_NAME + '.yaml'
else:
    config_file = MODEL_NAME + '_' + DATASET_NAME + '.yaml'
if __name__ == "__main__":
    training_app.run(os.path.join('../configs', config_file))
