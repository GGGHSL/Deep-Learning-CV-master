import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def inference(x, theta):
    if x.ndim == 1:
        x.reshape(1, len(x))
    if theta.ndim == 1:
        theta.reshape(len(theta), 1)
    g = np.matmul(x, theta)
    h = np.divide(np.exp(g), np.exp(g) + 1)
    return h


def classify(h, threshold=0.5):
    h[np.argwhere(h < threshold)] = 0
    h[np.argwhere(h >= threshold)] = 1
    return h


def eval_loss(x, y, theta):
    if y.ndim == 1:
        y.reshape(1, len(y))
    h = inference(x, theta)
    loss = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return loss


def update(x, y, theta, lr):
    if y.ndim == 1:
        y = y.reshape(len(y), 1)
    h = inference(x, theta)
    if x.ndim == 1:
        x.reshape(1, len(x))
    deriv = np.matmul(x.T, h - y) / x.shape[0]
    theta -= lr * deriv
    return theta


def train(x, y, lr, batch_size, max_iter):
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
        print(loss)
        if loss > loss_list[_ - 1]:
            break
    return loss_list, theta


def main():
    """ Hyper parameters:"""
    sample_size = 200
    features = 2
    classes = 2
    lr = 1e-3
    batch_size = 5
    max_iters = 100

    # ---
    X, y = make_classification(sample_size, features, n_classes=classes,
                               n_informative=2, n_redundant=0, n_repeated=0)
    print(X.shape, y.shape)
    tra_loss, theta = train(X, y, lr, batch_size, max_iters)
    h = inference(X, theta)
    y_hat = classify(h, 0.5)

    plt.figure()
    plt.scatter(X[np.argwhere(y == 1), 0], X[np.argwhere(y == 1), 1], c='r', alpha='0.5')
    plt.scatter(X[np.argwhere(y == 0), 0], X[np.argwhere(y == 0), 1], c='g', alpha='0.5')
    plt.scatter(X[np.argwhere(y_hat == 1), 0], X[np.argwhere(y_hat == 1), 1],
                c='r', marker='x')
    plt.scatter(X[np.argwhere(y_hat == 0), 0], X[np.argwhere(y_hat == 0), 1],
                c='g', marker='x')
    plt.savefig("./classification_result.jpg", dpi=1000)
    plt.show()


if __name__ == "__main__":
    main()
