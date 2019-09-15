import numpy as np
import torch
import numpy.linalg as la
import scipy.linalg as spla

from functools import partial
from torch.autograd import Variable
from cachetools.keys import hashkey
from cachetools import LRUCache, cachedmethod
from operator import attrgetter


class RansacMatching(object):
    # @cachedmethod(attrgetter('cache'), key=partial(hashkey, 'homography'))
    def get_homography(self, P1, P2, MAX_ITERS=100):
        """
        Find the matching point subsets using RANSAC Algorithm.
        -----------------------------------------------------------------------------
        Parameters
            P1 & P2: coordinates of matched keypoints.
            MAX_ITERS: maximum times of RANSAC iterations allowed in the algorithm
        """
        S = 4  # minimum number of data points required to estimate model parameters
        # LR = 1e-3
        # MOMENTUM = 0.9
        N = P1.shape[0]

        # (0) Normalization：
        P1, P2 = self.__get_normalized(P1, P2)  # -> torch.FloatTensor

        # P1 = torch.FloatTensor(P1)
        # P2 = torch.FloatTensor(P2)

        # mu = torch.FloatTensor(np.sqrt((P1[:, 0] - P2[:, 0]) ** 2 + (P1[:, 1] - P2[:, 1]) ** 2)).mean()
        # sigma = torch.FloatTensor(np.sqrt((P1[:, 0] - P2[:, 0]) ** 2 + (P1[:, 1] - P2[:, 1]) ** 2)).std()
        # D = mu + sigma.mul(2)
        mu = np.sqrt((P1[:, 0] - P2[:, 0]) ** 2 + (P1[:, 1] - P2[:, 1]) ** 2).mean()
        print("mu", mu)
        sigma = np.sqrt((P1[:, 0] - P2[:, 0]) ** 2 + (P1[:, 1] - P2[:, 1]) ** 2).std()
        print("sigma", sigma)
        # D = mu + sigma * 2
        D = np.sqrt(5.99) * sigma
        print(D)

        currInlier = S
        H_final = np.zeros((3, 3))
        """ RANSAC Iterations: """
        for i in range(MAX_ITERS):
            # (1) Choose 'S' pairs of points randomly in matching points:
            inliers = np.random.choice(range(N), size=S, replace=False).tolist()
            src = [P1[_] for _ in inliers]
            dst = [P2[_] for _ in inliers]

            # (2) Initialize the homography 'H' & loss:
            # H = Variable(self.__get_H(src, dst), requires_grad=True)
            # optimizer = torch.optim.SGD([{'params': H}, ],
            #                             lr=LR, momentum=MOMENTUM)
            # optimizer.zero_grad()
            # loss = H.sum() * 0
            H = self.__get_H(src, dst)
            # loss = 0
            try:
                # H_invs = H.inverse()
                H_invs = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                for _ in range(H.shape[0]):
                    H[_, _] += 1e-6
                # H_invs = H.inverse()
                H_invs = np.linalg.inv(H)
            # for p1, p2 in zip(src, dst):
            #     f2 = self.__get_perspective(p1, H)
            #     f1 = self.__get_perspective(p2, H_invs)
            #     d1 = self.__get_distance(p1, f1)
            #     d2 = self.__get_distance(p2, f2)
            #     loss += torch.add(d2, d1)
            """"""
            # (3) Use this computed homography to test all the other outliers
            #     and separated them by using Des1 threshold into two parts:
            consensus_set = []
            outliers = [_ for _ in range(N) if _ not in inliers]
            for _ in outliers:
                p1, p2 = P1[_], P2[_]
                f2 = self.__get_perspective(p1, H)
                f1 = self.__get_perspective(p2, H_invs)
                d1 = self.__get_distance(p1, f1)
                d2 = self.__get_distance(p2, f2)
                # print(d1, d2)
                if d1 < D and d2 < D:
                    consensus_set.append(_)
            # print(loss)
            # loss.backward()
            # optimizer.step()
            # print("len", len(consensus_set))
            inliers += consensus_set
            if i == 0:
                # finalLoss = loss / len(inliers)
                if len(consensus_set) > 0:
                    currInlier = len(inliers)
                    src += [P1[_] for _ in consensus_set]
                    dst += [P2[_] for _ in consensus_set]
                    H = self.__get_H(src, dst)
                # H_final = H.detach().numpy()
                H_final = H
                continue
            # if finalLoss > loss / len(inliers):
            if len(inliers) > currInlier:
                currInlier = len(inliers)
                src += [P1[_] for _ in consensus_set]
                dst += [P2[_] for _ in consensus_set]
                # (5) Re-estimating H using all members of the inliers set.
                H = self.__get_H(src, dst)
                # H_final = H.detach().numpy()
                H_final = H
            else:
                continue
        print(currInlier)
        return H_final

    def __get_distance(self, p1, p2):
        # if isinstance(p1, np.ndarray):
        #     p1 = torch.FloatTensor(p1)
        # if isinstance(p2, np.ndarray):
        #     p2 = torch.FloatTensor(p2)
        # d = torch.sqrt(torch.sum((p1 - p2) ** 2))
        d = np.sqrt(np.sum((p1 - p2) ** 2))
        return d

    def __get_normalized(self, P1, P2):  # TODO: 修改
        # if isinstance(P1, np.ndarray):
        #     P1 = torch.FloatTensor(P1)
        # if isinstance(P2, np.ndarray):
        #     P2 = torch.FloatTensor(P2)
        """"""
        # 1.中心化
        # p1_c, p2_c = P1.mean(dim=0), P2.mean(dim=0)
        p1_c, p2_c = np.mean(P1, axis=0), np.mean(P2, axis=0)
        P1 -= p1_c
        P2 -= p2_c
        # 2.归一化
        # D1, D2 = [], []
        # for p1, p2 in zip(P1, P2):
        #     D1.append(self.__get_distance(p1, p1_c))
        #     D2.append(self.__get_distance(p2, p2_c))
        # P1 /= max(D1)
        # P2 /= max(D2)
        return P1, P2

    def __get_perspective(self, p, H):
        x, y = p
        # src = torch.FloatTensor([x, y, 1]).reshape(3, 1)
        src = np.array([x, y, 1]).reshape(3, 1)
        # u, v, w = torch.mm(H, src)
        u, v, w = np.matmul(H, src)
        # dst = torch.FloatTensor([u/w, v/w])
        dst = np.array([u / w, v / w])
        return dst

    def __get_H(self, src, dst):
        A = []
        b = []
        for p1, p2 in zip(src, dst):
            x, y = p1
            u, v = p2
            # ai = torch.zeros(2, 8)
            # ai[0, 0:3] = torch.FloatTensor([x, y, 1.])
            # ai[1, 3:6] = torch.FloatTensor([x, y, 1.])
            # ai[:, 6:8] = torch.FloatTensor([[-u*x, -u*y],
            #                                 [-v*x, -v*y]])
            ai = np.zeros(shape=(2, 8))
            ai[0, 0:3] = np.array([x, y, 1.])
            ai[1, 3:6] = np.array([x, y, 1.])
            ai[:, 6:8] = np.array([[-u * x, -u * y],
                                            [-v * x, -v * y]])
            # bi = torch.FloatTensor([u, v]).reshape(2, 1)
            bi = np.array([u, v]).reshape(2, 1)
            if len(A) == 0:
                A = ai
                b = bi
            else:
                # A = torch.cat((A, ai), 0)
                # b = torch.cat((b, bi), 0)
                A = np.r_[A, ai]
                b = np.r_[b, bi]

        Q, R = la.qr(A)
        H = spla.solve_triangular(R, Q.T.dot(b), lower=False)
        # try:
        #     H = np.linalg.solve(A, b)
        # except np.linalg.LinAlgError:
        #     m = min(A.shape[0], A.shape[1])
        #     for i in range(m):
        #         A[i, i] += 1e-6
        #     H = np.linalg.solve(A, b)
        # H = torch.FloatTensor(np.append(H, 1).reshape(3, 3))
        H = np.append(H, 1).reshape(3, 3)
        return H
