from typing import List, Tuple, Optional, Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Linear:
    """全连层"""

    weight: np.ndarray  # 权重 [(num_in + 1) x num_out]
    weight_grad: Optional[np.ndarray]  # loss相对于权重的梯度 [(num_in + 1) x num_out]
    forward_x: Optional[np.ndarray]  # 记录最近一次调用forward的x，方便backward的时候计算梯度 [n x (num_in + 1)]

    def __init__(self, num_in: int, num_out: int):
        """随机初始化权重，权重的每个元素是-1到1的均匀分布"""
        self.weight = np.random.uniform(-1, 1, (num_in + 1, num_out))
        self.forward_x = None
        self.weight_grad = None

    # [n x num_in] -> [n x num_out]
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。完成Y = [X, 1] @ W，此外，保存[X, 1]方便反向传播"""
        one = np.ones((x.shape[0], 1))
        self.forward_x = np.concatenate((x, one), axis=1)
        return self.forward_x @ self.weight

    # [n x num_out] -> [ n x num_in ]
    def backward(self, y_grad: np.ndarray) -> np.ndarray:
        """反向传播。计算loss对W的梯度，存入weight_grad；返回loss对X的梯度。y_grad是loss对Y的梯度。

        W_grad = X^T @ Y_grad
        X_grad = Y_grad @ W^T"""
        assert self.forward_x is not None
        self.weight_grad = np.transpose(self.forward_x) @ y_grad
        x_grad = y_grad @ np.transpose(self.weight)
        return x_grad[:, :-1]

    def weights(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """返回，权重及其梯度，方便优化"""
        return [(self.weight, self.weight_grad)]


# def fc(weight, feature):
#     one = np.ones((feature.shape[0], 1))
#     xx = np.concatenate((feature, one), axis=1)
#     return xx @ weight
#
#
# def fc_grad(yy_grad, feature):
#     """calculate the gradient of fully connected layer.
#
#     output = [input, 1] @ weight    let x = [input, 1]
#     o_{ij} = x_{ik} * w_{kj}
#     {dl / dw}{kj} = x_{ik} * {dl / do}{ij}
#     """
#     one = np.ones((feature.shape[0], 1))
#     xx = np.concatenate((feature, one), axis=1)
#     return np.transpose(xx) @ yy_grad


class ReLU:
    """ReLU激活层"""
    mask: Optional[np.ndarray]  # 输入中大于<0的掩码（元素为bool），形状与forward输入一致

    # return same shape as x
    def forward(self, x: np.ndarray) -> np.ndarray:
        """计算出x<0的部分，用bool数组表示（掩码），保存掩码，以便反向传播。并且将x拷贝一份，置这部分为0"""
        self.mask = x < 0
        x = x.copy()
        x[self.mask] = 0
        return x

    def backward(self, y_grad: np.ndarray) -> np.ndarray:
        """拷贝y_grad，并且置掩码对应的部分为0"""
        assert self.mask is not None
        y_grad = y_grad.copy()
        y_grad[self.mask] = 0
        return y_grad

    # noinspection PyMethodMayBeStatic
    def weights(self):
        return []


class CrossEntropyLoss:
    predict: Optional[np.ndarray]
    label: Optional[np.ndarray]

    def forward(self, predict: np.ndarray, label: np.ndarray) -> float:
        ln = np.log(np.sum(np.exp(predict), axis=1)) - predict[np.arange(label.shape[0]), label]
        self.label = label
        self.predict = predict
        return np.mean(ln).item()

    def backward(self) -> np.ndarray:
        assert self.predict is not None and self.label is not None
        ee = np.exp(self.predict)
        grad = ee / np.sum(ee, axis=1)[:, np.newaxis]
        grad[np.arange(self.label.shape[0]), self.label] -= 1
        grad /= self.label.shape[0]
        return grad


# def cross_entropy_loss(predict, label):
#     ln = np.log(np.sum(np.exp(predict), axis=1)) - predict[np.arange(label.shape[0]), label]
#     return np.mean(ln)
#
# def cross_entropy_loss_grad(predict, label):
#     """calculate the gradient of cross entropy loss.
#
#     dl/dx = 1 / n * (exp x / (sum(exp X)) - {i == label})
#     """
#     ee = np.exp(predict)
#     grad = ee / np.sum(ee, axis=1)[:, np.newaxis]
#     grad[np.arange(label.shape[0]), label] -= 1
#     grad /= label.shape[0]
#     return grad


class Compose:
    layers: List[Any]

    def __init__(self, layers: List[Any]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            x = layer.backward(x)
        return x

    def weights(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        weights = []
        for layer in self.layers:
            weights.extend(layer.weights())
        return weights


class SGD:
    lr: float

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, weights: List[Tuple[np.ndarray, Optional[np.ndarray]]]):
        for w, w_grad in weights:
            assert w_grad is not None
            w -= self.lr * w_grad


def train(x_train, y_train, model, loss, optimizer) -> float:
    predict = model.forward(x_train)
    train_loss = loss.forward(predict, y_train)
    predict_grad = loss.backward()
    model.backward(predict_grad)
    optimizer.step(model.weights())
    return train_loss


def test(x_test, y_test, model, loss) -> Tuple[float, float]:
    predict = model.forward(x_test)
    test_loss = loss.forward(predict, y_test)
    test_acc = np.sum(np.argmax(predict, axis=1) == y_test) / y_test.shape[0]
    return test_loss, test_acc


# def train(x_train, x_test, y_train, y_test, weight, epoch=200, lr=0.05):
#     for i in range(epoch):
#         pre = fc(weight, x_train)
#         train_loss = cross_entropy_loss(pre, y_train)
#         valid_loss, acc = validate(x_test, y_test, weight)
#         print("Iter %3d: train loss = %f, valid loss = %f, accuracy = %f" %
#               (i, train_loss.item(), valid_loss.item(), acc))
#         pre_grad = cross_entropy_loss_grad(pre, y_train)
#         weight_grad = fc_grad(pre_grad, x_train)
#         weight -= weight_grad * lr
#     return weight
#
#
# def validate(x, y, weight):
#     predict = fc(weight, x)
#     loss = cross_entropy_loss(predict, y)
#     acc = np.sum(np.argmax(predict, axis=1) == y)/y.shape[0]
#     return loss, acc


def main():
    dataset = load_iris()
    x = dataset['data']
    y = dataset['target']
    np.random.seed(42)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    model = Compose([
        Linear(4, 10),
        ReLU(),
        Linear(10, 6),
        ReLU(),
        Linear(6, 3),
    ])
    loss = CrossEntropyLoss()
    optimizer = SGD(lr=0.03)
    for i in tqdm(range(200)):
        train_loss = train(x, y, model, loss, optimizer)
        test_loss, test_acc = test(x, y, model, loss)
        tqdm.write("Iter %4d:  train_loss=%.4f  test_loss=%.4f  test_acc=%.2f%%" %
                   (i, train_loss, test_loss, test_acc * 100))


if __name__ == '__main__':
    main()
