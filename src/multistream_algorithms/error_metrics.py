from multistream_algorithms.e_detectors import (
    ProblemType,
    DetectorType,
    MultiStreamAlgorithm,
    get_e_detector,
    get_declarations,
)

from typing import List
from enum import Enum
import numpy as np
from tqdm import tqdm

ErrorType = Enum("ErrorType", zip(["fdr", "fwer", "pfer"], range(4)))


def get_error_rate(
    error_type: ErrorType,
    problem_type: ProblemType,
    detector_type: DetectorType,
    algorithm: MultiStreamAlgorithm,
    n_sims=100,
    timeseries_length=1000,
    changepoint: int | List[int] = 200,
    n_streams=50,
    alpha: float | List[float] = 0.001,
    signal_strength=1,
):
    if isinstance(changepoint, int) or changepoint is None:
        changepoint = [changepoint] * n_streams
    match error_type:
        case ErrorType.fdr:
            fdr = np.zeros((n_sims, timeseries_length))
            for n in tqdm(range(n_sims)):
                e_detector = get_e_detector(
                    problem_type,
                    detector_type,
                    timeseries_length,
                    changepoint,
                    n_streams,
                    signal_strength,
                )
                declarations = get_declarations(e_detector, algorithm, alpha)
                for t in range(timeseries_length):
                    false_declared = sum(
                        changepoint[k] is None or changepoint[k] > t
                        for k in declarations[t]
                    )
                    fdr[n, t] = false_declared / max(1, len(declarations[t]))

            return np.mean(fdr, axis=0)

        case ErrorType.fwer:
            fwer = np.zeros((n_sims, timeseries_length))

            for n in tqdm(range(n_sims)):
                e_detector = get_e_detector(
                    problem_type,
                    detector_type,
                    timeseries_length,
                    changepoint,
                    n_streams,
                    signal_strength,
                )
                declarations = get_declarations(e_detector, algorithm, alpha)
                for t in range(timeseries_length):
                    fwer[n, t] = any(
                        changepoint[k] is None or changepoint[k] > t
                        for k in declarations[t]
                    )

            return np.mean(fwer, axis=0)

        case ErrorType.pfer:
            pfer = np.zeros((n_sims, timeseries_length))

            for n in tqdm(range(n_sims)):
                e_detector = get_e_detector(
                    problem_type,
                    detector_type,
                    timeseries_length,
                    changepoint,
                    n_streams,
                )
                declarations = get_declarations(e_detector, algorithm, alpha)
                for t in range(timeseries_length):
                    pfer[n, t] = sum(
                        changepoint[k] is None or changepoint[k] > t
                        for k in declarations[t]
                    )

            return np.mean(pfer, axis=0)


def get_mean_detection_curve(
    problem_type: ProblemType,
    detector_type: DetectorType,
    algorithm: MultiStreamAlgorithm,
    timeseries_length=1000,
    changepoint: int | List[int] = 200,
    n_streams=50,
    alpha: float | List[float] = 0.001,
    n_sims=100,
    signal_strength=1,
):
    detection_curve = np.zeros((n_sims, timeseries_length))
    for n in tqdm(range(n_sims)):
        e_detector = get_e_detector(
            problem_type,
            detector_type,
            timeseries_length,
            changepoint,
            n_streams,
            signal_strength,
        )
        declarations = get_declarations(e_detector, algorithm, alpha)
        detection_curve[n, :] = np.array(
            [len(declarations[i]) for i in range(timeseries_length)]
        )

    return np.mean(detection_curve, axis=0)
