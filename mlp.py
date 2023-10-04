import numpy as np
import matplotlib.pyplot as plt


# 多層パーセプトロン Multilayer Perceptron(MLP)

"""
Network(4, 10, 3) は以下のようなネットワークになる

入力層    中間層    出力層
第0層     第1層     第2層
  0         0        0
       +--- 1
  1    |    :        1
       w    :
  2 ---+    :        2
            :
  3         9

第 n 層の第 i ニューロンと第 n + 1 層の第 j ニューロンの間の重みを weights[n][j][i] とする。
たとえば、第 0 層の第 2 ニューロンと第 1 層の第 1 ニューロンの重み w は weights[0][1][2] である。

第 0-1 層間の重みは以下のような 10 x 4 の行列で表せる
w[0][0][0] w[0][0][1] w[0][0][2] w[0][0][3]
w[0][1][0] w[0][1][1] w[0][1][2] w[0][1][3]
w[0][2][0] w[0][2][1] w[0][2][2] w[0][2][3]
w[0][3][0] w[0][3][1] w[0][3][2] w[0][3][3]
w[0][4][0] w[0][4][1] w[0][4][2] w[0][4][3]
w[0][5][0] w[0][5][1] w[0][5][2] w[0][5][3]
w[0][6][0] w[0][6][1] w[0][6][2] w[0][6][3]
w[0][7][0] w[0][7][1] w[0][7][2] w[0][7][3]
w[0][8][0] w[0][8][1] w[0][8][2] w[0][8][3]
w[0][9][0] w[0][9][1] w[0][9][2] w[0][9][3]

第 1-2 層間の重みは以下のような 3 x 10 の行列で表せる
w[1][0][0] w[1][0][1] w[1][0][2] w[1][0][3] w[1][0][4] w[1][0][5] w[1][0][6] w[1][0][7] w[1][0][8] w[1][0][9]
w[1][1][0] w[1][1][1] w[1][1][2] w[1][1][3] w[1][1][4] w[1][1][5] w[1][1][6] w[1][1][7] w[1][1][8] w[1][1][9]
w[1][2][0] w[1][2][1] w[1][2][2] w[1][2][3] w[1][2][4] w[1][2][5] w[1][2][6] w[1][2][7] w[1][2][8] w[1][2][9]

"""
class Network:
    rate = 0.01
    def __init__(self, *args):
        self.layers = list(args)
        self.weights = []
        self.patterns = []
        self.labels = []

    def init_weights(self, a = 0.0, b = 1.0):
        """
        重みを [a, b] の一様乱数で生成する。
        Network(4, 10, 3) の重みを初期化すると以下のようになる。
        # 第 0-1 層間の重み
        [[0.07094166, 0.05255842, 0.53307851, 0.50960566],
        [0.1344915 , 0.88652122, 0.19228051, 0.21441718],
        [0.69756915, 0.36936117, 0.13604888, 0.31815905],
        [0.77379649, 0.30812321, 0.43408164, 0.08794032],
        [0.05748019, 0.13136268, 0.93272662, 0.21275778],
        [0.1837135 , 0.14936261, 0.75498656, 0.78742599],
        [0.58629062, 0.69982728, 0.77854615, 0.1025681 ],
        [0.80904313, 0.27483846, 0.93328905, 0.28882756],
        [0.22379712, 0.47314493, 0.029553  , 0.08686074],
        [0.50851518, 0.30895085, 0.08231878, 0.95880906]],
        # 第 1-2 層間の重み
        [[0.17622046, 0.22128327, 0.99307706, 0.13003668, 0.18957712, 0.84001958, 0.8566045 , 0.3567937 , 0.47674821, 0.89313193],
        [0.59929473, 0.57563752, 0.67996787, 0.43355506, 0.34688862, 0.37739767, 0.79666537, 0.91795776, 0.88649513, 0.38457239],
        [0.74602564, 0.92732807, 0.73340297, 0.28577091, 0.37989426, 0.2820897 , 0.08054472, 0.6739104 , 0.46731479, 0.28834806]]
        
        Parameters
        ----------
        a : float
            最小値
        b : float
            最大値
        """
        for i in range(len(self.layers) - 1):
            self.weights.append((b - a) * np.random.rand(self.layers[i + 1], self.layers[i]) + a)

    def load_iris(self):
        """
        アヤメ学習データを読み込む
        """
        dataset = []
        with open('iris.data', 'r') as file:
            lines = file.read().split()
            for line in lines:
                pattern = line.split(',')
                dataset.append(pattern)
        for pattern in dataset:
            self.patterns.append(list(map(float, pattern[0:-1])))
            # 第 2 層(出力層)において
            # setosaは 0 番目ニューロン
            # versicolor は 1 番目ニューロン
            # virginica は 2 番目ニューロンが担当する
            if pattern[-1] == 'Iris-setosa':
                self.labels.append(0)
            elif pattern[-1] == 'Iris-versicolor':
                self.labels.append(1)
            else:
                self.labels.append(2)

    @staticmethod
    def input_sum(inputs, weights):
        """
        前ニューロンからの入力和を返す。
        第 n 層 のニューロンから第 n + 1 層の第 i ニューロンへの入力和 Z[n][i] は
        Z[n][i] = x1 * w[n][i][1] + x2 * w[n][i][2] + ...
        となる。
        たとえば、第 0 層から第 1 層の第 2 ニューロンへの入力和は
        Z[0][2] = x1 * w[0][2][1] + x2 * w[0][2][2] + x3 * w[0][2][3] + x4 * w[0][2][4]
        となる。
        """
        return np.dot(inputs, weights)

    @staticmethod
    def step(x):
        """
        ステップ関数(活性化関数)
        """
        return 1 if x > 0 else 0

    def activation(self, inputs, weights):
        """
        ニューロンへの入力 z に対して、出力 y を返す活性化関数
        ある閾値(バイアス) Θ を超えていれば 1, 超えていなければ 0 を出力する
        """
        return self.step(self.input_sum(inputs, weights))

    def forward_propagate(self, pattern):
        """
        順伝播処理を行う
        """
        inputs = pattern  # 第 0 層目ニューロンの出力はデータセットの入力パターンとする
        for n in range(1, len(self.layers)):
            outputs = []
            self.last_outputs = inputs  # 最後の入力ニューロンの重みを更新するため保持しておく
            for i in range(self.layers[n]):
                # たとえば n = 1 のとき
                # 第 1 層の 0 番目ニューロンへの入力和に使われる重みは weights[0][0]
                # 第 1 層の 1 番目ニューロンへの入力和に使われる重みは weights[0][1]
                # 第 1 層の 2 番目ニューロンへの入力和に使われる重みは weights[0][2]
                # ...
                outputs.append(self.activation(inputs, self.weights[n - 1][i]))
            inputs = outputs  # output を次の input につかう
        return outputs

    @staticmethod
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

    def train(self, epoch):
        for e in range(epoch):
            error_sum = 0
            for p in range(len(self.patterns)):
                # 入力パターンに対する出力ニューロンを求める
                outputs = self.forward_propagate(self.patterns[p])
                # 教師ニューロンを作成する
                teacher = [0, 0, 0] 
                teacher[self.labels[p]] = 1  # ラベルに該当する教師ニューロンの出力を 1 にする
                # 出力ニューロンと教師ニューロンの誤差を求める
                for o in range(len(outputs)):
                    err = self.error(outputs[o], teacher[o])
                    error_sum += abs(err)  # 誤差の絶対値の和を求める
                    for i in range(len(self.last_outputs)):
                        # 最終層の一つ前の層の重みを更新する
                        self.weights[len(self.layers) - 2][o][i] += self.rate * err * self.last_outputs[i]
            print(f'{e + 1} / {epoch} epoch. [error_sum: {error_sum}]')
            if error_sum == 0:
                break

    def test(self):
        correct = 0
        for p in range(len(self.patterns)):
            # 入力パターンに対する出力ニューロンを求める
            outputs = self.forward_propagate(self.patterns[p])
            # 教師ニューロンを作成する
            teacher = [0, 0, 0]
            teacher[self.labels[p]] = 1  # ラベルに該当する教師ニューロンの出力を 1 にする
            # 出力と教師が同じならば正解とする
            if outputs == teacher:
                correct += 1
        print(f'Accuracy: {correct / len(self.patterns) * 100} %')


if __name__ == '__main__':
    net = Network(4, 50, 3)
    net.init_weights(-0.5, 0.5)
    net.load_iris()
    net.test()  # 学習前の精度
    net.train(100)
    net.test()  # 学習後の精度
