###############################################################################
import numpy as np
import random as rn

#DO NOT CHANGE THIS
from utils import select_validation_set

np.random.seed(2577)
rn.seed(3581)
###################

from utils import load_datasets_filenames, load_experiment_configuration
from utils import load_dataset, save_predictions
from utils import scale_data

from sklearn.model_selection import StratifiedKFold, train_test_split



if __name__ == "__main__":

    print("Step 1 - Loading configurations")

    datasets_filenames = load_datasets_filenames()
    config = load_experiment_configuration()
    predictions = {}
    exp = 1

    print("Step 2 - Starting experiment")

    for dataset_name in datasets_filenames:
        print('Dataset: ', dataset_name)
        instances, gold_labels = load_dataset(dataset_name)
        skfold = StratifiedKFold(n_splits = config["num_folds"],
                                 shuffle = True)
        predictions[dataset_name] = {}

        for fold, division in enumerate(skfold.split(X=instances, y=gold_labels), 1):
            train_idxs = division[0]
            test_idxs = division[1]
            train_instances = instances.iloc[train_idxs].values
            train_gold_labels = gold_labels.iloc[train_idxs].values.ravel()
            test_instances = instances.iloc[test_idxs].values
            test_gold_labels = gold_labels.iloc[test_idxs].values.ravel()

            predictions[dataset_name][fold] = {}
            predictions[dataset_name][fold]["gold_labels"] = test_gold_labels.tolist()

            # separacao dos dados em dificuldade das instancias com kDN
            for hardness_type, filter_func in config["validation_hardnesses"]:
                validation_instances, validation_gold_labels = select_validation_set(
                    train_instances, train_gold_labels, filter_func, config["kdn"])

                predictions[dataset_name][fold][hardness_type] = {}

                base_clf = config["base_classifier"]()
                clf_pool = config["generation_strategy"](base_clf, config["pool_size"])
                clf_pool.fit(train_instances, train_gold_labels)

                print("Experiment " + str(exp))
                exp+=1

    print("Step 2 - Finished experiment")

    print("Step 3 - Storing predictions")
    save_predictions(predictions)

    print(predictions)