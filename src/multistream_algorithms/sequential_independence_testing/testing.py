import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from .payoff_fns import COCOWitness, HSICWitness, KCCWitness
from sklearn.metrics import pairwise_distances

TYPES_KERNEL = ["rbf", "laplace"]


def compute_hyperparam(data: np.ndarray, kernel_type="rbf", style="median") -> float:
    """
    Median heuristic for computing kernel hyperparameters
    """
    if kernel_type == "rbf":
        if data.ndim == 1:
            dist = pairwise_distances(data.reshape(-1, 1)) ** 2
        else:
            dist = pairwise_distances(data) ** 2
    elif kernel_type == "laplace":
        if data.ndim == 1:
            dist = pairwise_distances(data.reshape(-1, 1), metric="l1")
        else:
            dist = pairwise_distances(data, metric="l1")
    else:
        raise ValueError("Unknown kernel type")
    mask = np.ones_like(dist, dtype=bool)
    np.fill_diagonal(mask, 0)
    if style == "median":
        return 1 / (2 * np.median(dist[mask]))
    elif style == "mean":
        return 1 / (2 * np.mean(dist[mask]))


def compute_rkhs_norm(
    data_x: np.ndarray,
    data_y: np.ndarray,
    gamma_x: float = 1,
    gamma_y: float = 1,
    kernel_type="rbf",
) -> float:
    """
    Compute RKHS norm of the difference between
        empirical means when RBF kernel is used

    Parameters
    ----------
    data: array_like
        first column are x', second are y's
        this data is used to approximate the feature means of empirical distributions

    gamma: float
        scale for the RBF kernel
    Returns
    -------
        RKHS-norm of the difference between mean embeddings/sqrt of hsic test statistic
    """

    n_pts = data_x.shape[0]

    if data_x.ndim == 1:
        if kernel_type == "rbf":
            k_mat = rbf_kernel(data_x.reshape(-1, 1), gamma=gamma_x)
        elif kernel_type == "laplace":
            k_mat = laplacian_kernel(data_x.reshape(-1, 1), gamma=gamma_x)
    else:
        if kernel_type == "rbf":
            k_mat = rbf_kernel(data_x, gamma=gamma_x)
        elif kernel_type == "laplace":
            k_mat = laplacian_kernel(data_x, gamma=gamma_x)
    if data_y.ndim == 1:
        if kernel_type == "rbf":
            l_mat = rbf_kernel(data_y.reshape(-1, 1), gamma=gamma_y)
        elif kernel_type == "laplace":
            l_mat = laplacian_kernel(data_y.reshape(-1, 1), gamma=gamma_y)
    else:
        if kernel_type == "rbf":
            l_mat = rbf_kernel(data_y, gamma=gamma_y)
        elif kernel_type == "laplace":
            l_mat = laplacian_kernel(data_y, gamma=gamma_y)
    h_mat = np.eye(n_pts) - np.ones(shape=[n_pts, n_pts]) / n_pts
    hsic = np.sqrt(np.trace(k_mat @ h_mat @ l_mat @ h_mat)) / n_pts
    return hsic


def batch_hsic(
    data_x: np.ndarray,
    data_y: np.ndarray,
    n_permutations: int = 20,
    gamma_x: float = 1,
    gamma_y: float = 1,
    kernel_type: np.ndarray = "rbf",
) -> float:
    """
    Function that implements batch-HSIC for independence testing
    Permutation p-value is used for testing.

    Returns
    -------
        P: float
            permutation p-value
    """
    n_pts = data_x.shape[0]
    # compute kernel matrices
    if data_x.ndim == 1:
        if kernel_type == "rbf":
            k_mat = rbf_kernel(data_x.reshape(-1, 1), gamma=gamma_x)
        elif kernel_type == "laplace":
            k_mat = laplacian_kernel(data_x.reshape(-1, 1), gamma=gamma_x)
    else:
        if kernel_type == "rbf":
            k_mat = rbf_kernel(data_x, gamma=gamma_x)
        elif kernel_type == "laplace":
            k_mat = laplacian_kernel(data_x, gamma=gamma_x)
    if data_y.ndim == 1:
        if kernel_type == "rbf":
            l_mat = rbf_kernel(data_y.reshape(-1, 1), gamma=gamma_y)
        elif kernel_type == "laplace":
            l_mat = laplacian_kernel(data_y.reshape(-1, 1), gamma=gamma_y)
    else:
        if kernel_type == "rbf":
            l_mat = rbf_kernel(data_y, gamma=gamma_y)
        elif kernel_type == "laplace":
            l_mat = laplacian_kernel(data_y, gamma=gamma_y)

    # center one of two matrices (suffices)
    k_mat -= k_mat.mean(axis=0)
    l_mat -= l_mat.mean(axis=0)

    # compute test statistic
    hsic_orig = (k_mat * l_mat.T).sum() / n_pts**2
    perm_comparison = [1]

    for _ in range(n_permutations):
        cur_perm = np.random.permutation(n_pts)
        perm_comparison += [
            (k_mat[cur_perm, :][:, cur_perm] * l_mat.T).sum() / n_pts**2 >= hsic_orig
        ]

    P = np.mean(perm_comparison)
    return P


class SeqIndTester(object):
    """
    SKIT
    """

    def __init__(self):
        # specify the payoff function style, default: hsic
        self.payoff_style = "hsic"
        self.bet_type = "mean"
        # specify type of a kernel, default: RBF-kernel with scale parameter 1
        self.kernel_type = "rbf"
        self.kernel_param_x = 1
        self.kernel_param_y = 1
        # lmbd params
        # choose fixed or mixture method
        self.lmbd_type = "mixing"
        self.lmbd_value = 0.5
        self.lmbd_grid_size = 19
        self.grid_of_lmbd = None
        # wealth process vals
        self.wealth = 1
        self.wealth_flag = False
        # store intermediate vals for linear updates
        self.wf = None
        self.recompute_sample_size = 2
        self.num_proc_pairs = 1
        self.mixed_wealth = None
        self.payoff_hist = list()
        # for testing
        self.significance_level = 0.05
        self.null_rejected = False
        self.run_mean = 0
        self.run_second_moment = 0
        self.norm_constant_search = list()
        self.opt_lmbd = 0
        self.grad_sq_sum = 1
        self.truncation_level = 0.5
        self.odd_fun = "tanh"
        self.hist_grad_sq = list()
        self.num_of_grads = 1
        self.denom = 0

    def initialize_mixture_method(self):
        """
        Initialize mixture method
        """
        # self.grid_of_lmbd = np.linspace(0.05, 0.95, self.lmbd_grid_size)
        self.grid_of_lmbd = np.linspace(0.05, 0.95, self.lmbd_grid_size)
        # wealth now is tracked for each value of lmbd, resulting wealth is average for uniform prior
        self.wealth = [1 for _ in range(self.lmbd_grid_size)]
        self.wealth_flag = [False for _ in range(self.lmbd_grid_size)]

    def compute_hsic_payoff(self, next_pair_x, next_pair_y, prev_data_x, prev_data_y):
        """
        Function that computes HSIC payoff
        """
        if self.wf is None:
            self.wf = HSICWitness()
            self.wf.kernel_type = self.kernel_type
            self.wf.kernel_param_x = self.kernel_param_x
            self.wf.kernel_param_y = self.kernel_param_y
            self.wf.bet_type = self.bet_type
            # self.wf.weighted_wf = self.weighted_wf
            if self.bet_type == "mean":
                # initialize norm constant
                self.wf.initialize_norm_const(prev_data_x, prev_data_y)
            if self.lmbd_type == "mixing":
                self.initialize_mixture_method()

        # evaluate wf for points from joint dist and from product and evaluate wf
        w1 = self.wf.evaluate_wf(
            next_pair_x[1:2], next_pair_y[1:2], prev_data_x, prev_data_y
        )
        w2 = self.wf.evaluate_wf(
            next_pair_x[0:1], next_pair_y[0:1], prev_data_x, prev_data_y
        )
        w3 = self.wf.evaluate_wf(
            next_pair_x[0:1], next_pair_y[1:2], prev_data_x, prev_data_y
        )
        w4 = self.wf.evaluate_wf(
            next_pair_x[1:2], next_pair_y[0:1], prev_data_x, prev_data_y
        )

        payoff_fn = 1 / 2 * (w1 + w2 - w3 - w4)
        # update normalization constant
        self.wf.update_norm_const(next_pair_x, next_pair_y, prev_data_x, prev_data_y)

        return payoff_fn

    def compute_kcc_payoff(self, next_pair_x, next_pair_y, prev_data_x, prev_data_y):
        """
        Function that computes KCC payoff
        """
        if self.wf is None:
            self.wf = KCCWitness()
            self.wf.kernel_type = self.kernel_type
            self.wf.kernel_param_x = self.kernel_param_x
            self.wf.kernel_param_y = self.kernel_param_y
            if self.lmbd_type == "mixing":
                self.initialize_mixture_method()
        n_pts = prev_data_x.shape[0]
        if n_pts == self.recompute_sample_size:
            # use historical data to estimate witness functions
            if self.recompute_sample_size == 2:
                # case of 2x2 matrix close to matrix of ones
                # perturb if needed, won't violate assumptions
                if np.linalg.norm(prev_data_x[0] - prev_data_x[1]) < 1e-1:
                    prev_data_x[0] += (
                        np.random.uniform() * np.ones_like(prev_data_x[0]) * 1e-2
                    )
                    prev_data_x[1] += (
                        np.random.uniform() * np.ones_like(prev_data_x[1]) * 1e-2
                    )
                if np.linalg.norm(prev_data_y[0] - prev_data_y[1]) < 1e-1:
                    prev_data_y[0] += (
                        np.random.uniform() * np.ones_like(prev_data_y[0]) * 1e-2
                    )
                    prev_data_y[1] += (
                        np.random.uniform() * np.ones_like(prev_data_y[1]) * 1e-2
                    )
            self.wf.estimate_coefs(prev_data_x, prev_data_y)
            self.recompute_sample_size += 4 * self.num_proc_pairs + 2
            self.num_proc_pairs += 1
            # evaluate witness fns on the next pair
        wf_x = self.wf.evaluate_wf_x(next_pair_x)
        wf_y = self.wf.evaluate_wf_y(next_pair_y)
        # compute payoff function
        payoff_fn = 1 / 4 * (wf_y[1] - wf_y[0]) * (wf_x[1] - wf_x[0])
        return payoff_fn

    def compute_coco_payoff(self, next_pair_x, next_pair_y, prev_data_x, prev_data_y):
        """
        Function that computes COCO payoff
        """
        if self.wf is None:
            self.wf = COCOWitness()
            self.wf.kernel_type = self.kernel_type
            self.wf.kernel_param_x = self.kernel_param_x
            self.wf.kernel_param_y = self.kernel_param_y
            if self.lmbd_type == "mixing":
                self.initialize_mixture_method()
        n_pts = prev_data_x.shape[0]
        if n_pts == self.recompute_sample_size:
            # use historical data to estimate witness functions
            if self.recompute_sample_size == 2:
                # case of 2x2 matrix close to matrix of ones
                # perturb if needed, won't violate assumptions
                if np.linalg.norm(prev_data_x[0] - prev_data_x[1]) < 1e-1:
                    prev_data_x[0] += (
                        np.random.uniform() * np.ones_like(prev_data_x[0]) * 1e-2
                    )
                    prev_data_x[1] += (
                        np.random.uniform() * np.ones_like(prev_data_x[1]) * 1e-2
                    )
                if np.linalg.norm(prev_data_y[0] - prev_data_y[1]) < 1e-1:
                    prev_data_y[0] += (
                        np.random.uniform() * np.ones_like(prev_data_y[0]) * 1e-2
                    )
                    prev_data_y[1] += (
                        np.random.uniform() * np.ones_like(prev_data_y[1]) * 1e-2
                    )
            self.wf.estimate_coefs(prev_data_x, prev_data_y)
            self.recompute_sample_size += 4 * self.num_proc_pairs + 2
            self.num_proc_pairs += 1
            # evaluate witness fns on the next pair
        wf_x = self.wf.evaluate_wf_x(next_pair_x)
        wf_y = self.wf.evaluate_wf_y(next_pair_y)
        # compute payoff function
        payoff_fn = 1 / 4 * (wf_y[1] - wf_y[0]) * (wf_x[1] - wf_x[0])
        return payoff_fn

    def process_pair(self, next_pair_x, next_pair_y, prev_data_x, prev_data_y):
        """
        Function to call to process next pair of datapoints:
        """
        # perform pairing to obtain points from the product
        # form points from joint dist and from product
        if self.payoff_style == "hsic":
            payoff_fn = self.compute_hsic_payoff(
                next_pair_x, next_pair_y, prev_data_x, prev_data_y
            )
        elif self.payoff_style == "coco":
            payoff_fn = self.compute_coco_payoff(
                next_pair_x, next_pair_y, prev_data_x, prev_data_y
            )
        elif self.payoff_style == "kcc":
            payoff_fn = self.compute_kcc_payoff(
                next_pair_x, next_pair_y, prev_data_x, prev_data_y
            )
        else:
            raise ValueError(
                "Unknown version of payoff function: use hsic, coco or kcc"
            )

        cand_payoff = payoff_fn
        # compute payoff function
        if self.bet_type == "mean":
            # cand_payoff = 1/2 * (w1+w2-w3-w4)
            self.payoff_hist += [cand_payoff]
            if self.lmbd_type == "aGRAPA":
                if self.num_proc_pairs == 1:
                    self.run_mean = [1e-3]
                    self.run_second_moment = [1]
                    self.opt_lmbd = min(
                        max(
                            np.mean(self.run_mean) / np.mean(self.run_second_moment), 0
                        ),
                        self.truncation_level,
                    )
                    payoff_fn = self.opt_lmbd * cand_payoff
                    self.run_mean += [cand_payoff]
                    self.run_second_moment += [cand_payoff**2]
                    self.lmbd_hist = [self.opt_lmbd]
                else:
                    self.opt_lmbd = min(
                        max(
                            np.mean(self.run_mean) / np.mean(self.run_second_moment), 0
                        ),
                        self.truncation_level,
                    )
                    payoff_fn = self.opt_lmbd * cand_payoff
                    self.run_mean += [cand_payoff]
                    self.run_second_moment += [cand_payoff**2]
                    self.lmbd_hist = [self.opt_lmbd]
            elif self.lmbd_type == "ONS":
                if self.num_proc_pairs == 1:

                    payoff_fn = self.opt_lmbd * cand_payoff
                    self.run_mean = np.copy(cand_payoff)
                    self.lmbd_hist = [self.opt_lmbd]
                else:
                    grad = self.run_mean / (1 + self.run_mean * self.opt_lmbd)
                    self.grad_sq_sum += grad**2
                    self.opt_lmbd = max(
                        0,
                        min(
                            self.truncation_level,
                            self.opt_lmbd
                            + 2 / (2 - np.log(3)) * grad / self.grad_sq_sum,
                        ),
                    )
                    payoff_fn = self.opt_lmbd * cand_payoff
                    self.run_mean = np.copy(cand_payoff)
                    self.lmbd_hist = [self.opt_lmbd]
            elif self.lmbd_type == "mixing":
                payoff_fn = cand_payoff
            self.num_proc_pairs += 1

        if self.lmbd_type == "ONS" or self.lmbd_type == "aGRAPA":
            cand_wealth = self.wealth * (1 + payoff_fn)
            if cand_wealth >= 0 and self.wealth_flag is False:
                self.wealth = cand_wealth
                if self.wealth >= 1 / self.significance_level:
                    self.null_rejected = True
            else:
                self.wealth_flag = True

        elif self.lmbd_type == "mixing":
            # update wealth for each value of lmbd
            cand_wealth = [
                self.wealth[cur_ind] * (1 + cur_lmbd * payoff_fn)
                for cur_ind, cur_lmbd in enumerate(self.grid_of_lmbd)
            ]
            for cur_ind in range(self.lmbd_grid_size):
                if cand_wealth[cur_ind] >= 0 and self.wealth_flag[cur_ind] is False:
                    self.wealth[cur_ind] = cand_wealth[cur_ind]
                    # update mixed wealth
                    self.mixed_wealth = np.mean(cand_wealth)
                    # update whether null is rejected
                    if self.mixed_wealth >= 1 / self.significance_level:
                        self.null_rejected = True
                else:
                    self.wealth_flag[cur_ind] = True
                    self.wealth[cur_ind] = 0
