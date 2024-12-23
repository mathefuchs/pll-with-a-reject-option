""" Script to run all experiments. """

import io
import random
import string
from glob import glob
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from partial_label_learning.config import (AUG_TYPE, DATASET_KIND,
                                           SELECTED_DATASETS)
from partial_label_learning.data import Experiment
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.relatedwork.cc_2020 import CC
from partial_label_learning.relatedwork.crosel_2024 import CroSel
from partial_label_learning.relatedwork.dst_pll_2024 import DstPll
from partial_label_learning.relatedwork.ipal_2015 import Ipal
from partial_label_learning.relatedwork.pl_ecoc_2017 import PlEcoc
from partial_label_learning.relatedwork.pl_knn_2005 import PlKnn
from partial_label_learning.relatedwork.pl_svm_2008 import PlSvm
from partial_label_learning.relatedwork.pop_2023 import Pop
from partial_label_learning.relatedwork.proden_2020 import Proden
from partial_label_learning.relatedwork.valen_2021 import Valen
from partial_label_learning.result import Result

DEBUG = False
ALGOS: Dict[str, Tuple[int, Type[PllBaseClassifier]]] = {
    # ML methods
    "pl-knn-2005": (0, PlKnn),
    "pl-svm-2008": (1, PlSvm),
    "ipal-2015": (2, Ipal),
    "pl-ecoc-2017": (3, PlEcoc),
    # Deep Learning methods
    "proden-2020": (4, Proden),
    "cc-2020": (5, CC),
    "valen-2021": (6, Valen),
    "pop-2023": (7, Pop),
    "crosel-2024": (8, CroSel),
    # Our method
    "dst-pll-2024": (9, DstPll),
}


def fts(number: float, max_digits: int = 6) -> str:
    """ Float to string. """

    return f"{float(number):.{max_digits}f}".rstrip("0").rstrip(".")


def get_header() -> str:
    """ Builds the header. """

    return "dataset,datasetkind,algo,seed,augmenttype," + \
        "split,truelabel,predlabel,correct,guess,reject,maxprob\n"


def append_output(
    output: List[str], algo_name: str, exp: Experiment,
    result: Result, is_train: bool,
) -> None:
    """ Create output from result. """

    true_label_list = exp.datasplit.y_true_train \
        if is_train else exp.datasplit.y_true_test
    res = result.train_result if is_train else result.test_result
    for true_label, pred, reject, probs, is_guess in zip(
        true_label_list, res.pred, res.reject, res.conf_probs, res.is_guessing,
    ):
        output.append(f"{int(SELECTED_DATASETS[exp.dataset_name][0])}")
        output.append(f",{int(DATASET_KIND[exp.dataset_kind])}")
        output.append(f",{int(ALGOS[algo_name][0])},{int(exp.seed)}")
        output.append(f",{int(AUG_TYPE[exp.augment_type])}")
        output.append(f",{0 if is_train else 1}")
        output.append(f",{int(true_label)},{int(pred)},")
        output.append(f"{int(true_label == pred)},{int(is_guess)},")
        output.append(f"{fts(reject)},{fts(np.max(probs))}\n")


def print_debug_msg(
    algo_name: str, exp: Experiment, result: Result,
) -> None:
    """ Print debug message. """

    train_acc = accuracy_score(
        exp.datasplit.y_true_train, result.train_result.pred)
    test_acc = accuracy_score(
        exp.datasplit.y_true_test, result.test_result.pred)
    if algo_name == "dst-pll-2024":
        train_acc_no_reject = accuracy_score(
            exp.datasplit.y_true_train[result.train_result.reject > 0],
            result.train_result.pred[result.train_result.reject > 0],
        ) if np.count_nonzero(result.train_result.reject > 0) >= 1 else 0.0
        test_acc_no_reject = accuracy_score(
            exp.datasplit.y_true_test[result.test_result.reject > 0],
            result.test_result.pred[result.test_result.reject > 0],
        ) if np.count_nonzero(result.test_result.reject > 0) >= 1 else 0.0
    else:
        train_acc_no_reject, test_acc_no_reject = train_acc, test_acc
    print(", ".join([
        f"{exp.dataset_name: >20}", f"{algo_name: >15}",
        f"{exp.augment_type: >15}", f"{exp.seed}",
        f"{train_acc:.3f}", f"{test_acc:.3f}",
        f"{result.train_result.frac_guessing():.3f}",
        f"{result.test_result.frac_guessing():.3f}",
        f"{result.train_result.frac_no_reject():.3f}",
        f"{result.test_result.frac_no_reject():.3f}",
        f"{train_acc_no_reject:.3f}",
        f"{test_acc_no_reject:.3f}",
    ]))


def run_experiment(fname: str, algo_name: str, algo_type: Type[PllBaseClassifier]) -> None:
    """ Runs the given experiment. """

    # Run experiment
    exp: Experiment = torch.load(fname)
    rng = np.random.Generator(np.random.PCG64(exp.seed))
    algo = algo_type(
        rng, DEBUG, dataset_kind=exp.dataset_kind,
        dataset_name=exp.dataset_name,
    )
    result = Result(
        train_result=algo.fit(
            exp.datasplit.x_train,
            exp.datasplit.y_train,
        ),
        test_result=algo.predict(exp.datasplit.x_test),
    )
    output = [get_header()]
    append_output(output, algo_name, exp, result, is_train=True)
    append_output(output, algo_name, exp, result, is_train=False)

    if DEBUG:
        # Print debug message
        print_debug_msg(algo_name, exp, result)
        csv_df = pd.read_csv(io.StringIO("".join(output)))
        csv_df.to_parquet(
            f"results/result_{algo_name}_{exp.seed}.parquet.gz",
            compression="gzip",
        )
    else:
        # Store predictions
        fname = "".join([
            random.choice(string.ascii_lowercase) for _ in range(12)])
        csv_df = pd.read_csv(io.StringIO("".join(output)))
        csv_df.to_parquet(f"results/{fname}.parquet.gz", compression="gzip")


if __name__ == "__main__":
    if not DEBUG:
        # Run all experimental settings
        Parallel(n_jobs=12)(
            delayed(run_experiment)(fname, algo_name, algo_type)
            for fname in tqdm(list(sorted(glob("experiments/*.pt"))))
            for algo_name, (_, algo_type) in ALGOS.items()
        )
    else:
        # Run single experiments for debugging
        print(62 * " " + "acc           guess         no-reject     no-reject acc")
        for s in range(5):
            run_experiment(f"experiments/exp{s}.pt", "pl-knn-2005", PlKnn)
            run_experiment(f"experiments/exp{s}.pt", "dst-pll-2024", DstPll)
