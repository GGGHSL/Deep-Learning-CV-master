import numpy as np
import torch
from torch.autograd import Variable


def ransacMatching(A, B):
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
    p = 0.95  # probability having at least picked one set of inliers after iterations
    S = 4  # minimum number of data points required to estimate model parameters
    E = 0.5  # probability that a point is an inlier
    K = 72  # maximum number of iterations allowed in the algorithm
    N = len(A)
    sigma = torch.FloatTensor([get_distance(p1, p2) for p1, p2 in zip(A, B)]).std()
    T = torch.sqrt(5.99) * sigma
    dtype = torch.FloatTensor  # torch.float32
    LR = 1e-2
    MOMENTUM = 0.9

    # (0) Normalization：
    A, B = get_normalized(A, B)  # torch.FloatTensor

    # (1) Choose 'S' pair of points randomly in matching points:
    inliers = np.random.choice(range(N), size=S, replace=False).tolist()
    src = [A[_] for _ in inliers]
    dst = [B[_] for _ in inliers]

    # (2) Initialize the homography 'H' & loss:
    H = Variable(get_init_H(src, dst), requires_grad=True)
    optimizer = torch.optim.SGD(params=H, lr=LR, momentum=MOMENTUM)

    """ RANSAC Iterations: """
    for i in range(K):
        optimizer.zero_grad()
        loss = torch.FloatTensor([0])
        H_invs = H.inverse()  # TODO: Singular matrix?
        for p1, p2 in zip(src, dst):
            f2 = get_perspective(p1, H)
            f1 = get_perspective(p2, H_invs)
            loss += (get_distance(p2, f2) + get_distance(p1, f1))

        # (3) Use this computed homography to test all the other outliers
        #     and separated them by using a threshold into two parts:
        new_inliers = []
        outliers = [_ for _ in range(N) if _ not in inliers]
        for _ in outliers:
            p1 = A[_]
            p2 = B[_]
            f2 = get_perspective(p1, H)
            f1 = get_perspective(p2, H_invs)
            d1 = get_distance(p1, f1)
            d2 = get_distance(p2, f2)
            if d2 <= T and d1 <= T:
                new_inliers.append(_)
                loss += (d1 + d2)

        if len(new_inliers) > 0:
            # (4) Get all inliers (new inliers + old inliers) and goto step (2)
            inliers += new_inliers
            src += [A[_] for _ in new_inliers]
            dst += [B[_] for _ in new_inliers]
        else:
            # (5) If there's no changes or we have already repeated step (2)-(4) K times, jump out of the recursion.
            # The final homography matrix 'H' will be the wanted one.
            break
        loss.backup()
        optimizer.step()
    return H.numpy()


def get_distance(p1, p2):
    if isinstance(p1, list):
        p1 = torch.FloatTensor(p1)
    if isinstance(p2, list):
        p2 = torch.FloatTensor(p2)
    d = torch.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return d

def get_normalized(P1, P2):
    if isinstance(P1, list):
        P1 = torch.FloatTensor(P1)
    if isinstance(P2, list):
        P2 = torch.FloatTensor(P2)
    n = P1.shape[0]
    # 1.中心化
    p1_c, p2_c = P1.mean(dim=0), P2.mean(dim=0)
    P1 -= p1_c
    P2 -= p2_c
    # 2.归一化
    D1, D2 = [], []
    for p1, p2 in zip(P1, P2):
        D1.append(get_distance(p1, p1_c))
        D2.append(get_distance(p2, p2_c))
    d1 = max(D1)  # -> tensor
    d2 = max(D2)
    P1 /= d1
    P2 /= d2
    return P1, P2

def get_perspective(p, H):
    x, y = p
    src = torch.FloatTensor([x, y, 1]).reshape(3,)
    u, v, w = H.mv(src)
    dst = torch.FloatTensor([u/w, v/w])
    return dst

def get_init_H(src, dst):
    a = []
    b = []
    for p1, p2 in zip(src, dst):
        x, y = p1
        u, v = p2
        ai = torch.zeros(2, 8)
        ai[0, 0:3] = torch.FloatTensor([x, y, 1.])
        ai[1, 3:6] = torch.FloatTensor([x, y, 1.])
        ai[:, 6:8] = torch.FloatTensor([[-u*x, -u*y],
                                        [-v*x, -v*y]])
        bi = torch.FloatTensor([u, v]).reshape(2,1)
        if len(a) == 0:
            a = ai
            b = bi
        else:
            a = torch.cat((a, ai), 0)
            b = torch.cat((b, bi), 0)
    H = torch.FloatTensor(np.linalg.solve(a, b))
    return H