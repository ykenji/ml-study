import numpy as np
import matplotlib.pyplot as plt
 
 
class Network:
    rate = 0.1     # 学習率
    decay = 0.1    # 学習率減衰
    per = 50       # 何エポックごとに学習率を減衰させるか
 
    epsilon = 1.0  # シグモイド関数の傾き
 
    def __init__(self, *args):
        self.layers = list(args)  # 各層のニューロン数
        self.weights = ([])       # 各層間の重みベクトル
        self.patterns = ([])      # 入力パターンベクトル
        self.labels = ([])        # 教師ニューロンインデックス
        self.outputs = ([])       # 各ニューロンの出力値ベクトル
 
    def init_weights(self, a = 0.0, b = 1.0):
        for i in range(len(self.layers)):
            if i + 1 >= len(self.layers):
                break
            self.weights.append((b - a) * np.random.rand(self.layers[i + 1], self.layers[i]) + a)
 
    def load_iris(self):
        dataset = []
        with open('iris.data', 'r') as file:
            lines = file.read().split()
            for line in lines:
                pattern = line.split(',')
                dataset.append(pattern)
        for pattern in dataset:
            self.patterns.append(list(map(float, pattern[0:-1])))
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
    def input_sum(input, weight):
        return np.dot(input, weight)
 
    def output(self, input, weight):
        return self.sigmoid(self.input_sum(input, weight))
 
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-self.epsilon * x))
 
    def forward(self, pattern):
        self.outputs.clear()  # 出力値情報をクリア
        input = pattern  # 第 0 層(入力層)ニューロンの出力はデータセットの入力パターンとする
        self.outputs.append(pattern)  # 入力層ニューロンの出力(入力パターン)を追加
        for l in range(1, len(self.layers)):
            output = []
            for i in range(self.layers[l]):
                """
                weights[n][j][i] は第 n 層の第 i ニューロンと第 n + 1 層の第 j ニューロンの間の重みを表す。
                weights[l - 1][i] は、第 l - 1 層からから第 l 層の i 番目に入る入力ベクトルにかかる重みのこと。
                """
                output.append(self.output(input, self.weights[l - 1][i]))
            input = output
            self.outputs.append(output)

    def backward(self, pattern):
        """
        Δw(k-1,i->k,j) = ηδ(k,j)O(k-1,i)
        """
        deltas = self.calc_delta(pattern)  # δを計算
        for k in reversed(range(1, len(self.layers))):  # ネットワークを逆順処理していく
            for j in range(self.layers[k]):
                for i, ii in enumerate(self.outputs[k-1]):
                    self.weights[k-1][j][i] += self.rate * deltas[k][j] * ii  # 重みを更新
 
    def calc_delta(self, pattern):
        """
        δを計算する関数
        :param pattern: パターンインデックス
        :return: δベクトル
        """
        teacher = ([0.1, 0.1, 0.1])  # 教師ニューロンを作成
        teacher[self.labels[pattern]] = 0.9  # ラベルに該当する教師ニューロンの出力を0.9に
        deltas = ([])  # 全ニューロンのδ
 
        for l in reversed(range(len(self.layers))):  # ネットワークを逆順処理していく
            tmp_delta = ([])
            #print(f'layer={l}')
            for j in range(self.layers[l]):
                #print(f'nuron={j}')
                if l == len(self.layers) - 1:  # 最終層ならば
 
                    delta = self.epsilon * \
                            (teacher[j] - self.outputs[l][j]) * \
                            self.outputs[l][j] * (1 - self.outputs[l][j])
 
                    tmp_delta.append(delta)
 
                else:  # 最終層以外
                    #print('----')
                    #print(self.weights[l])
                    t_weights = np.array(self.weights[l]).T
                    
                    delta = self.epsilon * \
                            self.outputs[l][j] * (1 - self.outputs[l][j]) * \
                            np.dot(deltas[-1], t_weights[j])
 
                    tmp_delta.append(delta)
 
            deltas.append(tmp_delta)

        return deltas[::-1]  # δは逆順で返す
 
    def train(self, epoch):
        """
        訓練関数
        :param epoch:
        :return:
        """
        errors = ([])
        for e in range(epoch):  # 最大学習回数で回す
            error = 100 - self.test()
            for p, pat in enumerate(self.patterns):  # 入力パターンごとに回す
 
                self.forward(pat)  # 順伝播処理で出力値を計算
 
                self.backward(p)  # 逆伝播で重みを更新
 
            print(str(e+1) + ' / ' + str(epoch) + ' epoch.' + ' [ error: ' + str(error) + ' ]')
 
            if (e+1) % self.per == 0:  # 学習率減衰
                self.rate *= self.decay
 
            errors.append(error)  # 不正解率
 
        #  不正解立を描画
        plt.xlabel('epochs')
        plt.ylabel('error (%)')
        plt.plot(errors)
        plt.savefig('error.png', dpi=300)
 
    def test(self):
        """
        テスト関数
        :return: accuracy (%)
        """
        correct = 0
        for p in range(len(self.patterns)):
            self.forward(self.patterns[p])
            max = 0
            ans = -1
            # 一番出力値の高いニューロンを取ってくる
            for o, out in enumerate(self.outputs[len(self.layers)-1]):
                if max < out:
                    max = out
                    ans = o
            # もしそのニューロンの番号とラベルの番号があっていれば正解！
            if ans == self.labels[p]:
                correct += 1
 
        accuracy = correct / len(self.patterns) * 100
        return accuracy
 
 
if __name__ == '__main__':
    net = Network(4, 20, 3)
    net.init_weights(-1.0, 1.0)
    net.load_iris()
    net.train(100)
    acc = net.test()
    print('Accuracy: ' + str(acc) + '%')
