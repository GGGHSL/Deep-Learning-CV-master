import numpy as np


def ransacMatching(A, B, METHOD = 2):
    """
    Find the best Perspective Transformation Homography Matrix 'H' using RANSAC Algorithm.
    In RANSAC iterations, each time we recompute least-squares 'H' estimate using all of the inliers.

    # Follow up 1: For step 3, How to do the "test"?
        * Set a threshold 'D2', if the distance between point B.i and the one computed by 'H.dot(A.i)'
          is smaller than 'D2', then (A.i, B.i) will be added to 'new_inliers';
    # Follow up 2: How to decide the "k" mentioned in step 5. Think about it mathematically!
        * Set:
            s = 4  # minimum number of data points required to estimate model parameters;
            p = 0.95: probability having at least picked one set of inliers after iterations;
            e = 0.5: probability that a point is an inlier;
          then the maximum number of iterations 'K' allowed in the algorithm equals to 72 according to:
            1 - p = (1 - e ** s) ** K,
            then K = log(1-p) / log(1 - e ** s).
    -----------------------------------------------------------------------------
    Parameters:
        A & B: list of list.
    """
    assert len(A) == len(B)
    S = 4  # minimum number of data points required to estimate model parameters
    K = 72  # maximum number of iterations allowed in the algorithm
    N = len(A)

    def get_distance(p1, p2):
        # p1 & p2: [x,y], list of position
        d2 = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        return d2

    distances = []
    for p1, p2 in zip(A, B):
        distances.append(get_distance(p1, p2))
    distances = np.array(distances)
    miu = np.mean(distances)
    sigma = np.std(distances)
    D2 = 5.99 * sigma ** 2

    def get_perspective_trans(src, H):
        src.append(1)
        src = np.array(src, dtype=np.float64).reshape(3, 1)
        u, v, w = H.dot(src).ravel()
        dst = [u / w, v / w]
        return dst

    """ Least Squares Estimation for Homography Matrix:
        # method 1: 'one_pair_coeff_1(src, dst)'
        #     compute 'A_i' for each pair of points and splice to one matrix 'A' by the row,
        #     then 'h' is the eigenvector of 'A.T.dot(A)' with smallest eigenvalue.
        
        # method 2: 'one_pair_coeff_2(src, dst)'
        #     structure a (8x8) matrix 'A' with a pair of point;
        #     compute 'A_i' for each pair, then get the matrix 'A' by summing them.
    """

    def one_pair_coeff_1(src, dst):
        A = np.zeros(shape=(2, 9))
        A[0, 0] = src[0]  # x
        A[0, 1] = src[1]  # y
        A[0, 2] = 1.
        A[0, 6] = 0. - dst[0] * src[0]  # -u_i * x_i
        A[0, 7] = 0. - dst[0] * src[1]  # -u_i * y_i
        A[0, 8] = 0. - dst[0]  # -u_i
        A[1, 3] = src[0]  # x
        A[1, 4] = src[1]  # y
        A[1, 5] = 1.
        A[1, 6] = 0. - dst[1] * src[0]  # -v_i * x_i
        A[1, 7] = 0. - dst[1] * src[1]  # -v_i * y_i
        A[1, 8] = 0. - dst[1]  # -v_i
        return A

    def one_pair_coeff_2(src, dst):
        x, y = src
        A = np.zeros(shape=(2, 8))
        A[0, 0:3] = [x, y, 1.]
        A[1, 3:6] = [x, y, 1.]
        A[:, 6:8] = np.dot(np.array([dst]).T, np.array([src]))
        return np.dot(A.T, A), np.dot(A.T, np.array([dst]).T)

    def get_homography(src, dst):
        if METHOD == 2:
            P = np.zeros(shape=(8, 8))
            b = np.zeros(shape=(8, 1))
            for s, d in zip(src, dst):
                Pi, bi = one_pair_coeff_2(s, d)
                P += Pi
                b += bi
            h = np.linalg.solve(P, b)
            H = np.append(h, 1).reshape(3, 3)
        else:  # METHOD == 1
            P = np.zeros(shape=(2, 9))
            for s, d in zip(src, dst):
                Pi = one_pair_coeff_1(s, d)
                if P.sum() == 0:
                    P = Pi
                else:
                    P = np.r_[P, Pi]
            w, v = np.linalg.eig(P.T.dot(P))
            H = v[np.argmin(w)].reshape(3, 3)
        return H

    # (1) Choose 'S' pair of points randomly in matching points:
    inliers = np.random.choice(range(N), size=S, replace=False).tolist()
    src = [A[_] for _ in inliers]
    dst = [B[_] for _ in inliers]

    """ RANSAC Iterations: """
    # (2) Get the homography 'H' of the inliers:
    for k in range(K):
        H = get_homography(src, dst)

        # (3) Use this computed homography to test all the other outliers
        #     and separated them by using a threshold into two parts:
        new_inliers = []
        for _ in [_ for _ in range(N) if _ not in inliers]:  # outliers
            a = A[_]
            b = B[_]
            b_hat = get_perspective_trans(a, H)
            if np.abs(get_distance(b, b_hat) - miu) <= D2:
                new_inliers.append(_)
        if len(new_inliers) > 0:
            # (4) Get all inliers (new inliers + old inliers) and goto step (2)
            src += [A[_] for _ in new_inliers]
            dst += [B[_] for _ in new_inliers]
        else:
            # (5) If there's no changes or we have already repeated step (2)-(4) K times, jump out of the recursion.
            # The final homography matrix 'H' will be the wanted one.
            break

    return H