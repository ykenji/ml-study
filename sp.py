import numpy as np
import matplotlib.pyplot as plt


# 単純パーセプトロン Simple perceptron(SP)

def input_sum(inputs, weights):
    """
    学習パターンの入力ベクトル X = {x1, x2, x3, ... xn} と
    重み W = {w1, w2, w3, ... wn} の
    内積 Z = x1 * w1 + x2 * w2 + ... + xn * wn を返す関数

    数式
          n
      z = Σ xi * wi
         i=1

    Parameters
    ----------
    inputs : array-like
        学習パターンの入力ベクトル。X = {x1, x2, ..., xn} のようなもの
    weights : array-like
        学習パターンの入力ベクトルにかける重み。W = {w1, w2, w3, ... wn} のようなもの

    Returns
    -------
    sum : float
        x と weights の内積
        z = x1 * w1 + x2 * w2 + ... + xn * wn のように計算される
    """
    return np.dot(inputs, weights)

def activation(sum, bias):
    """
    ニューロンへの入力 z に対して、出力 y を返す活性化関数
    ある閾値(バイアス) Θ を超えていれば 1, 超えていなければ 0 を出力する

    Parameters
    ----------
    sum : float
        input_sum の計算結果
    bias : float
        閾値(バイアス)

    Returns
    -------
    output : int
        活性化関数の結果
    """
    return 1 if sum > bias else 0

def step(x):
    """
    ステップ関数

    数式
    f(x) = 1 (x > 0), 0 (x <= 0)
    """
    return 1 if x > 0 else 0

def activation(sum):
    """
    x1 * w1 + x2 * w2 + ... + xn * wn > Θ
    の Θ を左辺に移項して
    -Θ + x1 * w1 + x2 * w2 + ... + xn * wn > 0
    Θ を重みの一つ w0 として考えると
    -w0 + x1 * w1 + x2 * w2 + ... + xn * wn > 0
    のようになってステップ関数を使えるようになる
    そのために、input_sum に与えるパラメータを
    学習パターンの入力ベクトル X = {-1, x1, x2, x3, ... xn}
    重み W = {w0, w1, w2, w3, ... wn} のようにする

    Parameters
    ----------
    sum : float
        input_sum の計算結果

    Returns
    -------
    output : int
        活性化関数(ステップ関数)の結果
    """
    return step(sum)


def error(output, label):
    """
    誤差関数
    出力 output が現在の重みで正しいクラスに分類できているかどうか、
    どの程度重みの修正が必要かを計算する
    
    Parameters
    ----------
    output : int
        学習パターンの入力ベクトル X に対する出力

    label : int
        教師信号
    
    Returns
    -------
    E : int
        誤差関数は重みの修正の向き
        E = 0 なら修正する必要なし
        E = 1 なら重みを大きくする方向に修正する
        E = -1 なら重みを小さくする方向に修正する
    """
    return label - output

def update(weights, err, inputs):
    """
    E の値をそのまま使うと修正量が大きすぎるので、学習率 η をかけて調整する。
    重み更新式は、以下のようになる。
    Δw = η * E * X
       = η * E * (x0 + x1 + x2 + ... + xn)
    重み更新で X をかけているのは、X の大きさに応じて重みを大きくすることで
    入力に応じた重みに変化しやすくなるからか？

    Parameters
    ----------
    weights : array-like
        学習パターンの入力ベクトルにかける重み
    err : int
        誤差関数の結果
    inputs : array-like
        学習パターンの入力ベクトル
    """
    for i in range(len(inputs)):
        weights[i] += 0.001 * float(err) * inputs[i]

def init_weights():
    """
    バイアス、重みを初期化
    w0, w1, ... w4 の初期値をランダムに生成する
    w0 は バイアス
    w1, w2, w3, w4 は アヤメの学習データが 4 つだから 
    """
    return np.random.rand(5)

def load_iris():
    """
    アヤメ学習データを読み込む
    """
    dataset = []
    with open('iris.data', 'r') as file:
        lines = file.read().split()
        for line in lines:
            pattern = line.split(',')
            dataset.append(pattern)
    return dataset

def test(weights, data, labels):
    """
    学習した重みで学習データを分類した結果と教師データを比較し、精度を求める
    
    Parameters
    ----------
    weights : array-like
        学習して得られた重み
    data : array-like
        アヤメデータ
    labels : array-like
        教師データ
    """
    correct = 0
    for i in range(len(data)):
        if activation(input_sum(data[i], weights)) == labels[i]:
            correct += 1
    print(f'accuracy: {correct * 100 / len(data)}%')


if __name__ == '__main__':
    dataset = load_iris()

    patterns = []
    labels = []

    # 訓練データを作成する。
    # つまり、学習パターンの入力ベクトルのリスト patterns と
    # それに対応する教師信号のリストを作成する。
    for pattern in dataset:
        if pattern[-1] != 'Iris-virginica':
            # 閾値(バイアス)入力を 0 番目に設定する
            patterns.append([-1.0] + list(map(float, pattern[0:-1])))
            # setosa は 0, versicolor は 1 の教師信号とする
            labels.append(0 if pattern[-1] == 'Iris-setosa' else 1)
 
    # 重みを初期化する
    weights = init_weights()
    
    # 1 つの訓練データを繰り返し学習して重みを修正していく
    epoch = 100
    for e in range(epoch):
        error_sum = 0
        for p in range(len(patterns)):
            err = error(activation(input_sum(patterns[p], weights)), labels[p])
            update(weights, err, patterns[p])
            error_sum += err ** 2

        print(f'{str(e + 1)} / {str(epoch)} epoch. [error_sum={error_sum}]')
        if error_sum == 0:
            break

    test(weights, patterns, labels)