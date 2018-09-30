# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('temparture_data.csv', header=None)
observed = df[2]
y = np.append(0, observed)
#図示のため、データがわからない0期も0という値を入れておく。
n_iter = y.shape[0]
#これが合計データ数である。

plt.rcParams['figure.figsize'] = (10, 8)


R = 0.01
#観測データを得る際のシミュレーションとの誤差の分散
#すべての期に渡って、ノイズは同じであるとしている。

Q = 0.01
#シミュレーション内部のノイズ（撹乱項）の分散
#すべての期に渡って、ノイズは同じであるとしている。


M = 3
#これがアンサンブルの大きさである。（20個データを毎回とってきて、分布に近似させる感じ。）


x_tilda=np.zeros((M, n_iter))
#各期のフィルター前の予測値がそれぞれアンサンブルメンバーごとに入る感じ。
x_hat=np.zeros((M, n_iter))
#各期のフィルター後の予測値がそれぞれのアンサンブルメンバーごとに入る感じ。x_tildaの各要素が更新される感じ。

x_hat[:,0]=10
#初期値を与えた。最初のアンサンブルのイメージ。全部0とした。つまり０期目のアンサンブルを作った。

x_bar=np.zeros(n_iter)
P_bar=np.zeros(n_iter)
#計算のためのスペース

x_hat_output=np.zeros(n_iter)
#推定値を利用する場合に、アンサンブルの平均値を用いて計算されたもの、
x_hat_output[0]=M**(-1)*np.sum(x_hat[:,0])
#初期値の推定値。

K_bar=np.zeros(n_iter)
# カルマンゲイン

v=np.zeros((M, n_iter))
w=np.zeros((M, n_iter))
#vには、各フィルター後の予測値から、次の期に移行する際のノイズを格納する。
#uには、シミュレーションの値が観測されるときのノイズを格納する。

for k in range(1,n_iter):#n_iter回更新されるn_iter個データはあるので。1, 2, ..., n_iter-1 まで出てくる
    '''予測'''
    v_temp = []
    for i in range(0,v.shape[0]):
        v_temp.append(np.random.normal(0, Q**(0.5)))
    v[:,k]=v_temp
    x_tilda[:,k] = x_hat[:,k-1] + v[:,k]
    print('{0}期の予測分布（フィルター前）のアンサンブルは{1}'.format(k,x_tilda[:,k]))
    #前の期のアンサンブルを今期のアンサンブルとする。
    '''フィルタリング'''
    x_bar[k]=M**(-1)*np.sum(x_tilda[:,k])
    P_bar[k]=(M-1)**(-1)*np.sum((x_tilda[:,k] - x_bar[k])**2)
    #スカラーなのでこのようにできる。
    K_bar[k]=P_bar[k]*1*(1*P_bar[k]*1+R)**(-1)

    w_temp = []
    for i in range(0,w.shape[0]):
        w_temp.append(np.random.normal(0, R**(0.5)))
    w[:,k]=w_temp
    x_hat[:,k]=x_tilda[:,k]+K_bar[k]*(y[k]-1*x_tilda[:,k]+w[:,k])
    x_hat_output[k]=M**(-1)*np.sum(x_hat[:,k])
    print('今回の観測データは{0}でした'.format(y[k]))
    print('{0}期のフィルター後予測のアンサンブルは{1}'.format(k, x_hat[:,k]))
    #以下図示のために、None value これは表示されない。を入れる。

y[0] = None


plt.figure()
plt.plot(y,'k+',label='observed')
plt.plot(x_hat_output,'b-',label='a posteri estimate')
plt.legend()
plt.title('Estimation', fontweight='bold')
plt.xlabel('Number of Days from 01/Nov./2011')
plt.ylabel('Average Temparture')
plt.show()



