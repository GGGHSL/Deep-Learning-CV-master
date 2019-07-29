import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings


def inference(x, theta):
    if x.ndim == 1:
        x.reshape(1, len(x))
    if theta.ndim == 1:
        theta.reshape(len(theta), 1)
    h = np.matmul(x, theta)
    return h


def eval_loss(x, y, theta):
    if y.ndim == 1:
        y.reshape(1, len(y))
    h = inference(x, theta)
    loss = np.mean((h - y) ** 2)
    return loss


def update(x, y, theta, lr):
    if y.ndim == 1:
        y.reshape(1, len(y))
    h = inference(x, theta)
    if x.ndim == 1:
        x.reshape(1, len(x))
    deriv = np.matmul(x.T, h - y) / x.shape[0]
    theta -= lr * deriv
    return theta


def train(x, y, lr, batch_size, max_iter, draw):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    n = x.shape[0]
    theta = np.zeros((x.shape[1], 1))
    loss_list = []
    for _ in range(max_iter):
        idx = np.random.choice(range(n), size=batch_size, replace=False)
        x_, y_ = x[idx], y[idx]
        theta = update(x_, y_, theta, lr)
        loss = eval_loss(x_, y_, theta)
        loss_list.append(loss)
        # if _ % 10 == 0:
        #     print(loss)
        print(loss)
        if loss > loss_list[_ - 1]:
            break
    if draw:
        plt.plot(loss_list)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("Train Loss")
        plt.savefig("./train_loss.jpg", dpi=1000)
    return loss_list, theta


def eval_R2(x, y, theta):
    h = inference(x, theta)
    score = r2_score(y, h)
    return score


def gen_sample_data(sample_size, features=1, intercept=True):
    theta = np.random.randint(0, 10, (features, 1)) + np.random.randint(0, 1)		# for noise random.random[0, 1)
    X = np.random.randint(0, 100, (sample_size, features)) * np.random.randint(0, 1, (sample_size, 1))

    if intercept:
        b = np.random.randint(0, 5, (1, 1)) + np.random.randint(0, 1)
        theta = np.r_[b, theta]
        X = np.c_[np.ones(shape=(sample_size, 1)), X]

    print(X.shape, theta.shape)
    y = np.matmul(X, theta) + np.random.randint(0, 1, (sample_size, 1))
    print("Generte: ")
    print("X: ", X.shape)  # , " include intercept: ", intercept
    print("y: ", y.shape)
    print("theta: ", theta.shape)
    return X, y, theta


def main():
    """ Hyper parameters:"""
    sample_size = 200
    features = 1
    intercept = True
    draw = True
    lr = 1e-2
    batch_size = 5
    max_iters = 100

    # ---
    X, y, gt_theta = gen_sample_data(sample_size, features, intercept)
    warnings.warn(
        "Warning: Fitting data generated randomly, evaluation result 'trainning R^2' may have no meaning!")
    tra_loss, theta = train(X, y, lr, batch_size, max_iters, draw)
    score = eval_R2(X, y, theta)
    print("trainning R^2: ", score)


if __name__ == "__main__":
    main()