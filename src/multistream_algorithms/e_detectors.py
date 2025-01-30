from multistream_algorithms.sequential_independence_testing.testing import SeqIndTester

from enum import Enum
from typing import List
from collections import defaultdict
import numpy as np
from tqdm import tqdm

ProblemType = Enum(
    "ProblemType",
    zip(["independence", "symmetry", "mean_change", "conformal"], range(4)),
)

DetectorType = Enum(
    "DetectorType",
    zip(["cusum", "sr"], range(2)),
)

MultiStreamAlgorithm = Enum(
    "MultiStreamAlgorithm", zip(["naive", "edbh", "edbonf", "edholm"], range(4))
)


def get_e_detector(
    problem_type: ProblemType,
    detector_type: DetectorType = DetectorType.cusum,
    timeseries_length=1000,
    changepoint: int | List[int] = 200,
    n_streams=50,
    signal_strength=1,
):
    if isinstance(changepoint, int) or changepoint is None:
        changepoint = [changepoint] * n_streams

    match problem_type:
        case ProblemType.independence:
            x = np.random.normal(size=(n_streams, timeseries_length, 1))

            eps = np.random.normal(size=(n_streams, timeseries_length, 1))
            mask = np.zeros((n_streams, timeseries_length, 1))
            for k in range(n_streams):
                if changepoint[k] is not None:
                    mask[k, changepoint[k] :, :] = 0.5

            y = mask * x + (1 - mask) * eps

            e_detector_temp = np.ones((n_streams, timeseries_length // 2))

            for k in range(n_streams):
                tester = SeqIndTester()
                tester.significance_level = 0.001
                tester.payoff_style = "hsic"
                tester.lmbd_type = "ONS"
                tester.truncation_level = 0.5

                for cur_pair in range(1, timeseries_length // 2):
                    tester.process_pair(
                        x[k, 2 * cur_pair : 2 * (cur_pair + 1)],
                        y[k, 2 * cur_pair : 2 * (cur_pair + 1)],
                        x[k, : 2 * cur_pair],
                        y[k, : 2 * cur_pair],
                    )

                    e_detector_temp[k, cur_pair] = tester.wealth

            e_detector = np.zeros((n_streams, timeseries_length))
            e_detector[:, : 2 * e_detector_temp.shape[1]] = np.repeat(
                e_detector_temp, 2, axis=1
            )

            e_detector[:, -1] = e_detector_temp[:, -1]

            return e_detector

        case ProblemType.symmetry:
            x = np.random.normal(size=[n_streams, timeseries_length])
            for k in range(n_streams):
                if changepoint[k] is not None:
                    x[k, changepoint[k] :] = x[k, changepoint[k] :] + 1
            e_processes = np.zeros((n_streams, timeseries_length))
            e_detector = np.zeros((n_streams, timeseries_length))

            for t in range(timeseries_length):
                for k in range(n_streams):
                    curr_obs = x[k, t]
                    e_processes[k, t] = 1
                    e_processes[k, :] += np.sign(curr_obs) * (e_processes[k, :] > 0)

                    if detector_type == DetectorType.cusum:
                        e_detector[k, t] = np.max(e_processes[k, :])
                    elif detector_type == DetectorType.sr:
                        e_detector[k, t] = np.sum(e_processes[k, :])

            return e_detector

        case ProblemType.mean_change:
            e_detector = np.ones((n_streams, timeseries_length))

            x = np.random.normal(size=(n_streams, timeseries_length)) - signal_strength
            for k in range(n_streams):
                if changepoint[k] is not None:
                    x[k, changepoint[k] :] += 2 * signal_strength

            for t in range(timeseries_length - 1):
                likelihood_ratio = np.exp(
                    -0.5
                    * (
                        (x[:, t] - signal_strength) ** 2
                        - (x[:, t] + signal_strength) ** 2
                    )
                )
                if detector_type == DetectorType.cusum:
                    e_detector[:, t + 1] = likelihood_ratio * np.maximum(
                        e_detector[:, t], 1
                    )
                elif detector_type == DetectorType.sr:
                    e_detector[:, t + 1] = likelihood_ratio * (e_detector[:, t] + 1)

            return e_detector

        case ProblemType.conformal:
            x = np.random.normal(size=(n_streams, timeseries_length))
            for k in range(n_streams):
                if changepoint[k] is not None:
                    x[k, changepoint[k] :] += 1

            betting_function = lambda z: 1 / (2 * np.sqrt(z))

            e_detector = np.zeros((n_streams, timeseries_length))
            nonconformity_scores = np.zeros((n_streams, timeseries_length))

            e_detector[:, 0] = np.random.uniform(0, 1, n_streams)

            for t in range(1, timeseries_length):
                for k in range(n_streams):
                    nonconformity_scores[k, t] = x[k, t] - np.mean(x[k, : (t + 1)])
                    e_detector[k, t] = (e_detector[k, t - 1] + 1) * betting_function(
                        (
                            np.sum(
                                nonconformity_scores[k, :t] > nonconformity_scores[k, t]
                            )
                            + np.random.uniform(0, 1)
                            * np.sum(
                                nonconformity_scores[k, : (t + 1)]
                                == nonconformity_scores[k, t]
                            )
                        )
                        / (t + 1)
                    )

            return e_detector


def get_declarations(
    e_detector, algorithm: MultiStreamAlgorithm, alpha: float | List[float] = 0.001
):
    n_streams, timeseries_length = e_detector.shape
    if isinstance(alpha, float):
        alpha = [alpha] * timeseries_length

    declared_streams = defaultdict(list)
    match algorithm:
        case MultiStreamAlgorithm.naive:
            for k in range(n_streams):
                for t in range(1, timeseries_length):
                    if e_detector[k, t] >= 1 / alpha[t]:
                        declared_streams[t].append(k)

        case MultiStreamAlgorithm.edbh:
            for t in range(timeseries_length):
                order_statistics = sorted(
                    zip(e_detector[:, t], range(n_streams)), key=lambda x: -x[0]
                )

                k_star = max(
                    (
                        k + 1
                        for k in range(n_streams)
                        if (k + 1) * order_statistics[k][0] / n_streams >= 1 / alpha[t]
                    ),
                    default=0,
                )

                for i in range(k_star):
                    declared_streams[t].append(order_statistics[i][1])

        case MultiStreamAlgorithm.edholm:
            for t in range(timeseries_length):
                order_statistics = sorted(
                    zip(e_detector[:, t], range(n_streams)), key=lambda x: -x[0]
                )

                k_star = max(
                    (
                        k
                        for k in range(n_streams + 1)
                        if all(
                            order_statistics[i][0] / (n_streams - i) >= 1 / alpha[t]
                            for i in range(k)
                        )
                    ),
                    default=0,
                )

                declared_streams[t] = [order_statistics[i][1] for i in range(k_star)]

        case MultiStreamAlgorithm.edbonf:
            for t in range(timeseries_length):
                declared_streams[t] = [
                    k
                    for k in range(n_streams)
                    if e_detector[k, t] >= n_streams / alpha[t]
                ]

    return declared_streams
