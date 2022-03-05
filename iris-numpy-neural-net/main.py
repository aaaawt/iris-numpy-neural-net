import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def fc(weight, feature):
    one = np.ones((feature.shape[0], 1))
    xx = np.concatenate((feature, one), axis=1)
    return xx @ weight


def cross_entropy_loss(predict, label):
    ln = np.log(np.sum(np.exp(predict), axis=1)) - predict[np.arange(label.shape[0]), label]
    return np.mean(ln)


def fc_grad(yy_grad, feature):
    """calculate the gradient of fully connected layer.

    output = [input, 1] @ weight    let x = [input, 1]
    o_{ij} = x_{ik} * w_{kj}
    {dl / dw}{kj} = x_{ik} * {dl / do}{ij}
    """
    one = np.ones((feature.shape[0], 1))
    xx = np.concatenate((feature, one), axis=1)
    return np.transpose(xx) @ yy_grad


def cross_entropy_loss_grad(predict, label):
    """calculate the gradient of cross entropy loss.

    dl/dx = 1 / n * (exp x / (sum(exp X)) - {i == label})
    """
    ee = np.exp(predict)
    grad = ee / np.sum(ee, axis=1)[:, np.newaxis]
    grad[np.arange(label.shape[0]), label] -= 1
    grad /= label.shape[0]
    return grad


def train(x_train, x_test, y_train, y_test, weight, epoch=200, lr=0.05):
    for i in range(epoch):
        pre = fc(weight, x_train)
        train_loss = cross_entropy_loss(pre, y_train)
        valid_loss, acc = validate(x_test, y_test, weight)
        print("Iter %3d: train loss = %f, valid loss = %f, accuracy = %f" %
              (i, train_loss.item(), valid_loss.item(), acc))
        pre_grad = cross_entropy_loss_grad(pre, y_train)
        weight_grad = fc_grad(pre_grad, x_train)
        weight -= weight_grad * lr
    return weight


def validate(x, y, weight):
    predict = fc(weight, x)
    loss = cross_entropy_loss(predict, y)
    acc = np.sum(np.argmax(predict, axis=1) == y)/y.shape[0]
    return loss, acc


def main():
    dataset = load_iris()
    x = dataset['data']
    y = dataset['target']
    np.random.seed(42)
    init_weight = np.random.rand(x.shape[1] + 1, len(dataset['target_names']))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    train(x_train, x_test, y_train, y_test, init_weight)


if __name__  == '__main__':
    main()
