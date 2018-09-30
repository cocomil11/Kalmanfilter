# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
以下xが真の値であるとするが、これは観測できないこととする。
'''
df = pd.read_csv('temparture_data.csv', header=None)
trueValues = df[2]
y = np.append(0, trueValues)
#図示のため、データがわからない0期も0という値を入れておく。
n_iter = y.shape[0]
#これが合計データ数である。

plt.rcParams['figure.figsize'] = (10, 8)


'''
ここまででデータ作成完了。
'''

'''初期分布が悪い場合。'''
xhat=np.zeros(n_iter)
# a posteri estimate of x
P=np.zeros(n_iter)
# a posteri error estimate
#この２つが事後予測（つまりx_kを推定しているときに、y_kの情報も入れて予想したもの）

xhatminus=np.zeros(n_iter)
# a priori estimate of x
Pminus=np.zeros(n_iter)
# a priori error estimate
#この２つが事前予測（つまりx_kを推定しているときに、x_(k-1)からシミュレーション内部での変化を考えて予想したもの）

K=np.zeros(n_iter)
# カルマンゲイン

R = 0.001
#観測データを得る際のシミュレーションとの誤差の分散
#すべての期に渡って、ノイズは同じであるとしている。

Q = 0.01
#シミュレーション内部のノイズ（撹乱項）の分散
#すべての期に渡って、ノイズは同じであるとしている。(これはレポートにちゃんと書くようにする。）

xhat[0] = 10
#ヒューリスティックにだいたい15度くらいだろうとする、
P[0] = 1.0
#初期の予測。これはヒューリスティックに与える


'''
以下、初期予測をもとに、更新していく。はじめに0期の初期予測があるとする。(最初に仮定する初期分布は事後分布扱いして計算していく。）
まず、これを事後分布として、次の期:1期の事前分布をもとめる。
次に1期の観測データをもちいて、この事前分布を更新し、事後分布を求める。
そして、これをもとに、次の期:2期の事前分布をもとめる。
'''


for k in range(1,n_iter):#n_iter回更新されるn_iter個データはあるので。1, 2, ..., n_iter-1 まで出てくる。
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q
    #まずシミュレーション内部で推測の変数を更新する。

    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(y[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]
    #今季に得られたデータを用いて更新する。
    #y[0]は用いられないようだ。

#以下図示のために、None value これは表示されない。を入れる。

y[0] = None
xhatminus[0] = None

plt.figure()
plt.plot(y,'k+',label='observed')
plt.plot(xhatminus,'s',label='a priori estimate')
plt.plot(xhat,'b-',label='a posteri estimate')
plt.legend()
plt.title('Estimation', fontweight='bold')
plt.xlabel('Number of Days from 01/Nov./2011')
plt.ylabel('Average Temparture')
plt.show()

print('各日付のその日のデータを反映する前の推定平均気温は、{0}'.format(xhatminus[1:]))
print('各日付の観測された平均気温は、{0}'.format(y[1:]))
print('各日付の推定平均気温は、{0}'.format(xhat[1:]))


