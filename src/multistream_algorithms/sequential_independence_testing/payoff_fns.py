import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, linear_kernel
from scipy.sparse.linalg import eigsh
from scipy.sparse import bmat
from .decomp import incomplete_cholesky

TYPES_KERNEL = ["rbf", "laplace"]


def rescale_eigenvals(lmbds: np.ndarray, kappa: float, sample_size: int) -> np.ndarray:
    """
    Function that performs rescaling of eigenvalues for KCCA method

    Parameters
    ----------
        lmbds: array_like
            singular values of low-rank approximation of kernel matrix
        kappa: float
            regularization parameter
        sample_size: int
            sample size / size of the original kernel matrix

    Returns
    -------
        scaled_eigs:
    """
    scaled_eigs = 1 / np.sqrt(lmbds**2 / sample_size + kappa)
    return scaled_eigs


def evaluate_wf(
    new_data: np.ndarray,
    hist_data: np.ndarray,
    perm_vec: np.ndarray,
    coefs: np.ndarray,
    kernel_type,
    kernel_param: float,
) -> np.ndarray:
    """
    Evaluate witness functions (for COCO/KCC) at a set of new points

    Parameters
    ----------
        new_data: array_like
            points at which witness function has to be evaluated

        hist_data: array_like
            historical data used to compute COCO/KCC solution

        perm_vec: array_like
            permutations applied to historical data
                when incomplete Cholesky was computed

        coefs: array_like
            argmax of the COCO/KCC objective function

        kernel_type: ['rbf', 'laplace']
            kernel type used

        kernel_param: float
            bandwidth of a kernel

    Returns
    -------
        wf: array_like
            array of witness function evaluations

    """
    if new_data.ndim == 1:
        if kernel_type == "rbf":
            eval_k_mat = rbf_kernel(
                new_data.reshape(-1, 1), hist_data[perm_vec], gamma=kernel_param
            )
        elif kernel_type == "laplace":
            eval_k_mat = laplacian_kernel(
                new_data.reshape(-1, 1), hist_data[perm_vec], gamma=kernel_param
            )
    else:
        if kernel_type == "rbf":
            eval_k_mat = rbf_kernel(new_data, hist_data[perm_vec], gamma=kernel_param)
        elif kernel_type == "laplace":
            eval_k_mat = laplacian_kernel(
                new_data, hist_data[perm_vec], gamma=kernel_param
            )
    mean_alpha = np.mean(coefs)
    wf = eval_k_mat @ (coefs - mean_alpha)

    return wf


class COCOWitness(object):
    """
    COCO witness functions
    """

    def __init__(self):
        self.kernel_param_x = 1
        self.kernel_param_y = 1
        self.kernel_type = "rbf"
        # store data for wf evaluations
        self.data_x = None
        self.data_y = None

        # permutation vectors for cholesky
        self.perm_vec_k = None
        self.perm_vec_l = None
        # cholesky error parameter
        self.inc_chol_tol = 1e-3

        # estimator of coco
        self.coco_est = None
        # estimated coefficients
        self.alpha = None
        self.beta = None

    def estimate_coefs(self, data_x, data_y):
        """
        Estimation of witness function
        """

        # compute the size of the dataset
        n_pts = data_x.shape[0]

        # kernel evaluations are performed only if needed!
        # store data for wf evaluations
        if data_x.ndim == 1:
            self.data_x = np.copy(data_x.reshape(-1, 1))
        else:
            self.data_x = np.copy(data_x)
        if data_y.ndim == 1:
            self.data_y = np.copy(data_y.reshape(-1, 1))
        else:
            self.data_y = np.copy(data_y)

        # compute decompositions
        self.perm_vec_k, decomp_k, error_k = incomplete_cholesky(
            self.data_x,
            n_pts * self.inc_chol_tol,
            kernel_type=self.kernel_type,
            kernel_param=self.kernel_param_x,
        )
        self.perm_vec_l, decomp_l, error_l = incomplete_cholesky(
            self.data_y,
            n_pts * self.inc_chol_tol,
            kernel_type=self.kernel_type,
            kernel_param=self.kernel_param_y,
        )
        # center matrices
        decomp_k -= decomp_k.mean(axis=0)
        decomp_l -= decomp_l.mean(axis=0)
        # compute decompositions
        u_k, s_k, _ = np.linalg.svd(decomp_k, full_matrices=False)
        u_l, s_l, _ = np.linalg.svd(decomp_l, full_matrices=False)

        # ignore very small eigenvalues
        u_k = u_k[:, s_k >= 1e-8]
        u_l = u_l[:, s_l >= 1e-8]
        s_k = s_k[s_k >= 1e-8]
        s_l = s_l[s_l >= 1e-8]

        prod_k = u_k * s_k
        prod_l = u_l * s_l
        rank_k = len(s_k)

        prod_mat_kl = prod_k.T @ prod_l / n_pts
        prod_mat_lk = prod_l.T @ prod_k / n_pts
        eig_mat = bmat([[None, prod_mat_kl], [prod_mat_lk, None]])
        self.coco_est, leading_eigen = eigsh(eig_mat, k=1, which="LA")

        self.alpha = (u_k / s_k) @ leading_eigen[:rank_k, 0]
        self.beta = (u_l / s_l) @ leading_eigen[rank_k:, 0]

        # renormalize
        norm_1 = np.linalg.norm(self.alpha - self.alpha.mean())
        norm_2 = np.linalg.norm(self.beta - self.beta.mean())
        norm_3 = np.linalg.norm(leading_eigen[:rank_k, 0])
        norm_4 = np.linalg.norm(leading_eigen[rank_k:, 0])

        self.alpha /= np.sqrt(norm_1**2 * error_k + norm_3**2)
        self.beta /= np.sqrt(norm_2**2 * error_l + norm_4**2)

    def evaluate_wf_x(self, new_data):
        """
        Evaluate witness functions (for X: g(X)) at a set of new points
        """
        return evaluate_wf(
            new_data,
            self.data_x,
            self.perm_vec_k,
            self.alpha,
            self.kernel_type,
            self.kernel_param_x,
        )

    def evaluate_wf_y(self, new_data):
        """
        Evaluate witness functions (for Y: h(Y)) at a set of new points
        """
        return evaluate_wf(
            new_data,
            self.data_y,
            self.perm_vec_l,
            self.beta,
            self.kernel_type,
            self.kernel_param_y,
        )


class KCCWitness(object):
    """
    KCC witness functions
    """

    def __init__(self):
        self.kernel_param_x = 1
        self.kernel_param_y = 1
        self.kernel_type = "rbf"
        self.data_x = None
        self.data_y = None

        # permutation vectors for cholesky
        self.perm_vec_k = None
        self.perm_vec_l = None
        # param for incomplete cholesky
        self.inc_chol_tol = 1e-3
        # regularization parameters for kcca
        self.kappa1 = 1e-1
        self.kappa2 = 1e-1

        # estimated canonical correlation
        self.kcca_est = None
        # estimated coefficients
        self.alpha = None
        self.beta = None

    def estimate_coefs(self, data_x, data_y):
        """
        Estimation of witness function
        """

        # compute the size of the dataset
        n_pts = data_x.shape[0]

        # kernel evaluations are performed only if needed!
        # store data for wf evaluations
        if data_x.ndim == 1:
            self.data_x = np.copy(data_x.reshape(-1, 1))
        else:
            self.data_x = np.copy(data_x)
        if data_y.ndim == 1:
            self.data_y = np.copy(data_y.reshape(-1, 1))
        else:
            self.data_y = np.copy(data_y)

        # compute decompositions
        self.perm_vec_k, decomp_k, error_k = incomplete_cholesky(
            self.data_x,
            n_pts * self.inc_chol_tol,
            kernel_type=self.kernel_type,
            kernel_param=self.kernel_param_x,
        )
        self.perm_vec_l, decomp_l, error_l = incomplete_cholesky(
            self.data_y,
            n_pts * self.inc_chol_tol,
            kernel_type=self.kernel_type,
            kernel_param=self.kernel_param_y,
        )
        # center matrices
        decomp_k -= decomp_k.mean(axis=0)
        decomp_l -= decomp_l.mean(axis=0)
        # compute decompositions
        u_k, s_k, _ = np.linalg.svd(decomp_k, full_matrices=False)
        u_l, s_l, _ = np.linalg.svd(decomp_l, full_matrices=False)

        # ignore very small eigenvalues
        u_k = u_k[:, s_k >= 1e-8]
        u_l = u_l[:, s_l >= 1e-8]
        s_k = s_k[s_k >= 1e-8]
        s_l = s_l[s_l >= 1e-8]

        scaling_factor_k = rescale_eigenvals(s_k, self.kappa1, n_pts)
        scaling_factor_l = rescale_eigenvals(s_l, self.kappa2, n_pts)

        prod_k = u_k * (s_k * scaling_factor_k)
        prod_l = u_l * (s_l * scaling_factor_l)
        rank_k = len(s_k)

        prod_mat_kl = prod_k.T @ prod_l / n_pts
        prod_mat_lk = prod_l.T @ prod_k / n_pts

        eig_mat = bmat([[None, prod_mat_kl], [prod_mat_lk, None]])

        self.kcca_est, leading_eigen = eigsh(eig_mat, k=1, which="LA")

        self.alpha = (u_k * (scaling_factor_k / s_k)) @ leading_eigen[:rank_k, 0]
        self.beta = (u_l * (scaling_factor_l / s_l)) @ leading_eigen[rank_k:, 0]

        # renormalize
        norm_1 = np.linalg.norm(self.alpha - self.alpha.mean())
        norm_2 = np.linalg.norm(self.beta - self.beta.mean())
        # part below is different from coco
        norm_3 = np.linalg.norm(scaling_factor_k * leading_eigen[:rank_k, 0])
        norm_4 = np.linalg.norm(scaling_factor_l * leading_eigen[rank_k:, 0])

        self.alpha /= np.sqrt(norm_1**2 * error_k + norm_3**2)
        self.beta /= np.sqrt(norm_2**2 * error_l + norm_4**2)

    def evaluate_wf_x(self, new_data):
        """
        Evaluate witness functions (for X: g(X)) at a set of new points
        """
        return evaluate_wf(
            new_data,
            self.data_x,
            self.perm_vec_k,
            self.alpha,
            self.kernel_type,
            self.kernel_param_x,
        )

    def evaluate_wf_y(self, new_data):
        """
        Evaluate witness functions (for Y: h(Y)) at a set of new points
        """
        return evaluate_wf(
            new_data,
            self.data_y,
            self.perm_vec_l,
            self.beta,
            self.kernel_type,
            self.kernel_param_y,
        )


class HSICWitness(object):
    def __init__(self):
        # specify type of a kernel, default: RBF-kernel with scale parameter 1
        self.kernel_type = "rbf"
        self.kernel_param_x = 1
        self.kernel_param_y = 1
        # number of processed pairs
        self.num_processed_pairs = 0
        # store normalization constant
        self.norm_constant = 1e-6
        # store intermediate vals for linear updates
        # K^t 1
        self.k_vec_of_ones_product = None
        # L^t 1
        self.l_vec_of_ones_product = None
        # tr(K^t L^t)
        self.trace_prod = None
        self.bet_type = "mean"
        self.weighted_wf = False

    def initialize_norm_const(self, first_pair_x, first_pair_y):
        """
        Function used to initialize the normalization
            constant using the first data pair
        """
        if self.kernel_type == "rbf":
            if first_pair_x.ndim == 1:
                k_mat = rbf_kernel(
                    first_pair_x.reshape(-1, 1), gamma=self.kernel_param_x
                )
            else:
                k_mat = rbf_kernel(first_pair_x, gamma=self.kernel_param_x)
            if first_pair_y.ndim == 1:
                l_mat = rbf_kernel(
                    first_pair_y.reshape(-1, 1), gamma=self.kernel_param_y
                )
            else:
                l_mat = rbf_kernel(first_pair_y, gamma=self.kernel_param_y)
        elif self.kernel_type == "laplace":
            if first_pair_x.ndim == 1:
                k_mat = laplacian_kernel(
                    first_pair_x.reshape(-1, 1), gamma=self.kernel_param_x
                )
            else:
                k_mat = laplacian_kernel(first_pair_x, gamma=self.kernel_param_x)
            if first_pair_y.ndim == 1:
                l_mat = laplacian_kernel(
                    first_pair_y.reshape(-1, 1), gamma=self.kernel_param_y
                )
            else:
                l_mat = laplacian_kernel(first_pair_y, gamma=self.kernel_param_y)

        # compute delta_1 and cache it
        delta_1 = np.trace(k_mat @ l_mat)
        self.trace_prod = delta_1
        # compute K^t 1 and delta_2, store K^t 1
        # self.k_vec_of_ones_product = k_mat @ np.ones(2)
        self.k_vec_of_ones_product = k_mat.sum(axis=1)
        delta_2 = self.k_vec_of_ones_product.sum()
        # same for L^t 1 and delta_3
        self.l_vec_of_ones_product = l_mat.sum(axis=1)
        delta_3 = self.l_vec_of_ones_product.sum()
        # compute delta_4
        delta_4 = self.k_vec_of_ones_product @ self.l_vec_of_ones_product

        # update number of processed pairs
        self.num_processed_pairs += 1

        # compute the first normalization constant
        self.norm_constant += np.sqrt(
            delta_1 / 4 + delta_2 * delta_3 / 16 - delta_4 / 4
        )

    def update_norm_const(self, next_pair_x, next_pair_y, prev_data_x, prev_data_y):
        """
        Function that updates the value of the normalization constant in linear time
            using cached values
        """
        # compute kernel matrices for new points
        if self.kernel_type == "rbf":
            # kernel evaluations with previous pts: x's
            # if one feature, add reshaping
            if next_pair_x.ndim == 1:
                k_mat_old = rbf_kernel(
                    prev_data_x.reshape(-1, 1),
                    next_pair_x.reshape(-1, 1),
                    gamma=self.kernel_param_x,
                )
                # kernel evaluations for a new pair: x's
                k_mat_new = rbf_kernel(
                    next_pair_x.reshape(-1, 1), gamma=self.kernel_param_x
                )
            else:
                k_mat_old = rbf_kernel(
                    prev_data_x, next_pair_x, gamma=self.kernel_param_x
                )
                # kernel evaluations for a new pair: x's
                k_mat_new = rbf_kernel(next_pair_x, gamma=self.kernel_param_x)
            # kernel evaluations with previous pts: y's
            if next_pair_y.ndim == 1:
                l_mat_old = rbf_kernel(
                    prev_data_y.reshape(-1, 1),
                    next_pair_y.reshape(-1, 1),
                    gamma=self.kernel_param_y,
                )
                # kernel evaluations for a new pair: y's
                l_mat_new = rbf_kernel(
                    next_pair_y.reshape(-1, 1), gamma=self.kernel_param_y
                )
            else:
                l_mat_old = rbf_kernel(
                    prev_data_y, next_pair_y, gamma=self.kernel_param_y
                )
                # kernel evaluations for a new pair: y's
                l_mat_new = rbf_kernel(next_pair_y, gamma=self.kernel_param_y)
        elif self.kernel_type == "laplace":
            # kernel evaluations with previous pts: x's
            # if one feature, add reshaping
            if next_pair_x.ndim == 1:
                k_mat_old = laplacian_kernel(
                    prev_data_x.reshape(-1, 1),
                    next_pair_x.reshape(-1, 1),
                    gamma=self.kernel_param_x,
                )
                # kernel evaluations for a new pair: x's
                k_mat_new = laplacian_kernel(
                    next_pair_x.reshape(-1, 1), gamma=self.kernel_param_x
                )
            else:
                k_mat_old = laplacian_kernel(
                    prev_data_x, next_pair_x, gamma=self.kernel_param_x
                )
                # kernel evaluations for a new pair: x's
                k_mat_new = laplacian_kernel(next_pair_x, gamma=self.kernel_param_x)
            # kernel evaluations with previous pts: y's
            if next_pair_y.ndim == 1:
                l_mat_old = laplacian_kernel(
                    prev_data_y.reshape(-1, 1),
                    next_pair_y.reshape(-1, 1),
                    gamma=self.kernel_param_y,
                )
                # kernel evaluations for a new pair: y's
                l_mat_new = laplacian_kernel(
                    next_pair_y.reshape(-1, 1), gamma=self.kernel_param_y
                )
            else:
                l_mat_old = laplacian_kernel(
                    prev_data_y, next_pair_y, gamma=self.kernel_param_y
                )
                # kernel evaluations for a new pair: y's
                l_mat_new = laplacian_kernel(next_pair_y, gamma=self.kernel_param_y)
        # update the value of tr(K^t L^t)
        self.trace_prod += 2 * (k_mat_old * l_mat_old).sum() + np.sum(
            k_mat_new * l_mat_new
        )
        # update the value of K^t 1
        term_1 = self.k_vec_of_ones_product + k_mat_old.sum(axis=1)
        term_2 = k_mat_old.T.sum(axis=1) + k_mat_new.sum(axis=1)
        self.k_vec_of_ones_product = np.hstack([term_1, term_2])
        # update the value of L^t 1
        term_1 = self.l_vec_of_ones_product + l_mat_old.sum(axis=1)
        term_2 = l_mat_old.T.sum(axis=1) + l_mat_new.sum(axis=1)
        self.l_vec_of_ones_product = np.hstack([term_1, term_2])

        # update number of processed pairs
        self.num_processed_pairs += 1

        # compute delta_1
        delta_1 = self.trace_prod
        # compute K^t 1 and delta_2
        delta_2 = self.k_vec_of_ones_product.sum()
        # same for L^t 1 and delta_3
        delta_3 = self.l_vec_of_ones_product.sum()
        # compute delta_4
        delta_4 = self.k_vec_of_ones_product @ self.l_vec_of_ones_product

        # compute normalization constant
        self.norm_constant = np.sqrt(
            delta_1 / ((2 * self.num_processed_pairs) ** 2)
            + delta_2 * delta_3 / ((2 * self.num_processed_pairs) ** 4)
            - 2 * delta_4 / ((2 * self.num_processed_pairs) ** 3)
        )

    def evaluate_wf(self, new_pt_x, new_pt_y, prev_data_x, prev_data_y):
        """
        Witness function evaluation
        """
        # if there is a single feature, reshape the array
        if self.kernel_type == "rbf":
            if new_pt_x.ndim == 1:
                k_mat = rbf_kernel(
                    new_pt_x.reshape(-1, 1),
                    prev_data_x.reshape(-1, 1),
                    gamma=self.kernel_param_x,
                )
            else:
                k_mat = rbf_kernel(new_pt_x, prev_data_x, gamma=self.kernel_param_x)
            if new_pt_y.ndim == 1:
                l_mat = rbf_kernel(
                    new_pt_y.reshape(-1, 1),
                    prev_data_y.reshape(-1, 1),
                    gamma=self.kernel_param_y,
                )
            else:
                l_mat = rbf_kernel(new_pt_y, prev_data_y, gamma=self.kernel_param_y)
        elif self.kernel_type == "laplace":
            if new_pt_x.ndim == 1:
                k_mat = laplacian_kernel(
                    new_pt_x.reshape(-1, 1),
                    prev_data_x.reshape(-1, 1),
                    gamma=self.kernel_param_x,
                )
            else:
                k_mat = laplacian_kernel(
                    new_pt_x, prev_data_x, gamma=self.kernel_param_x
                )
            if new_pt_y.ndim == 1:
                l_mat = laplacian_kernel(
                    new_pt_y.reshape(-1, 1),
                    prev_data_y.reshape(-1, 1),
                    gamma=self.kernel_param_y,
                )
            else:
                l_mat = laplacian_kernel(
                    new_pt_y, prev_data_y, gamma=self.kernel_param_y
                )
        elif self.kernel_type == "linear":
            if new_pt_x.ndim == 1:
                k_mat = linear_kernel(
                    new_pt_x.reshape(-1, 1), prev_data_x.reshape(-1, 1)
                )
            else:
                k_mat = linear_kernel(new_pt_x, prev_data_x)
            if new_pt_y.ndim == 1:
                l_mat = linear_kernel(
                    new_pt_y.reshape(-1, 1), prev_data_y.reshape(-1, 1)
                )
            else:
                l_mat = linear_kernel(new_pt_y, prev_data_y)
        if self.bet_type == "mean":
            mu_joint = np.mean(l_mat * k_mat)
            mu_product = np.mean(l_mat, axis=1) @ np.mean(k_mat, axis=1)
            res = (mu_joint - mu_product) / self.norm_constant
        elif self.bet_type == "symmetry":
            if self.weighted_wf is False:
                mu_joint = np.mean(l_mat * k_mat)
                mu_product = np.mean(l_mat, axis=1) @ np.mean(k_mat, axis=1)
                res = mu_joint - mu_product
            else:
                # assign weights
                cur_len = k_mat.shape[1] // 2
                weights = [(0.95) ** j for j in range(cur_len)][::-1]
                weights = np.repeat(weights, 2).reshape(1, -1)
                sum_weights = np.sum(weights)
                weights /= 2 * sum_weights
                mu_joint = np.sum(weights * l_mat * k_mat)
                mu_product = np.sum(weights * l_mat, axis=1) @ np.sum(
                    weights * k_mat, axis=1
                )
                res = mu_joint - mu_product
        return res
