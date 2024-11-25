import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from typing import Tuple, Literal

TYPES_KERNEL = ["rbf", "laplace"]


def incomplete_cholesky(
    mat: np.ndarray, tol_level: float = 1e-6, kernel_type="rbf", kernel_param: float = 1
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Function that computes incomplete Cholesky decomposition
    (adopted from F. Bach's implementation:
    https://www.di.ens.fr/~fbach/kernel-ica/index.htm)
    Parameters
    ----------
        mat: array_like
            Raw data X (used as input as not all kernel matrix evaluations are needed)
            Function returns for incomplete Cholesky decomposition
                for the coresponding kernel matrix
            (we assume that Gaussian or Laplacian kernel is used)

        tol_level: float
            stopping criterion

        kernel_type: ['rbf', 'laplace']
            Type of a kernel to be used

        kernel_param: float
            bandwidth of a kernel

    Returns
    -------
        p_vec: array_like
            permutation matrix stored as an array
            (to obtain permuted kernel matrix, call mat[p_vec,:][:,p_vec])

        G: array_like
            lower triangular matrix obtained via Cholesky decomposition:
                if K is a kernel matrix, then
            |K[p_vec,:][:,p_vec] - G @ G.T|<= tol_level
            see paper for more details

        cur_error: float
            nuclear norm of the unexplained error
    """
    n_pts = mat.shape[0]
    # initial permutation is identity
    p_vec = np.arange(n_pts, dtype="int")
    # initial sum of diagonal entries (all are ones)
    cur_ind = 0
    diagG = np.ones(n_pts)
    G = np.empty(shape=[n_pts, 0])
    perm_mat = np.copy(mat)
    cur_error = n_pts
    while cur_error > tol_level:
        G = np.c_[G, np.zeros(n_pts)]
        if cur_ind == 0:
            # process first column
            j_st = 0
        else:
            # find the best column
            j_st = cur_ind + np.argmax(diagG[cur_ind:])
            # update permutation matrix
            p_vec[[cur_ind, j_st]] = p_vec[[j_st, cur_ind]]
            # update elemets of matrix G
            G[[cur_ind, j_st]] = G[[j_st, cur_ind]]
            perm_mat[[cur_ind, j_st]] = perm_mat[[j_st, cur_ind]]

        # cholesky update / set the diagonal element to 1 (for gaussian and laplace kernels)
        new_el = np.sqrt(diagG[j_st])
        G[cur_ind, cur_ind] = new_el

        if cur_ind < n_pts - 1:
            # take a column slice of the kernel matrix
            if kernel_type == "rbf":
                col_to_use = rbf_kernel(
                    perm_mat[cur_ind + 1 :],
                    perm_mat[cur_ind : cur_ind + 1],
                    gamma=kernel_param,
                ).ravel()
            elif kernel_type == "laplace":
                col_to_use = laplacian_kernel(
                    perm_mat[cur_ind + 1 :],
                    perm_mat[cur_ind : cur_ind + 1],
                    gamma=kernel_param,
                ).ravel()
            if cur_ind == 0:
                # if the first column is processed
                G[cur_ind + 1 :, cur_ind] = col_to_use / new_el
            else:
                # compute the update
                G[cur_ind + 1 :, cur_ind] = (
                    col_to_use - G[cur_ind + 1 :, :cur_ind] @ G[cur_ind, :cur_ind]
                ) / new_el
            # update diagonal entries
            # works for gaussian kernel
            diagG[cur_ind + 1 :] = np.ones(n_pts - cur_ind - 1) - (
                G[cur_ind + 1 :, : cur_ind + 1] ** 2
            ).sum(axis=1)
        cur_ind += 1
        cur_error = sum(diagG[cur_ind:])
    return p_vec, G, cur_error
