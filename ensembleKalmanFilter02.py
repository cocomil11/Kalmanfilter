# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def lorenz(u, t, s, b, r):
    x = u[0]
    y = u[1]
    z = u[2]
    dxdt = -s * x + s * y
    dydt = -x * z + r * x - y
    dzdt = x * y - b * z
    return([dxdt, dydt, dzdt])

# parameters
s = 10
b = 8.0 / 3.0
r = 28

u0 = [0.1, 0.1, 0.1]

dt = 0.01
T = 200.0
times = np.arange(0.0, T, dt)

args = (s, b, r)
orbit = odeint(lorenz, u0, times, args)
#このorbitが真の値になる。(2000, 3)

true_x = orbit[:,0].tolist()
true_y = orbit[:,1].tolist()
true_z = orbit[:,2].tolist()

t = 0.1
#観測できる時間間隔を0.1とする。


n_iter = int(T*t**(-1))

y_x = []
y_y = []
y_z = []
#y_x, y_y, y_z はxについての観測データ

for i in range(int(T*t**(-1))):
    y_x.append(np.random.normal(true_x[i*10],0.1))
    y_y.append(np.random.normal(true_y[i*10],0.1))
    y_z.append(np.random.normal(true_z[i*10],0.1))
    #多分これで合ってる。
Y = np.array([y_x,y_y,y_z])
#これで列は期を表し、縦にそれぞれの変数ごとの観測値（x,y,z) が並ぶような行列を作ることができた。
Y[0,0]=None
Y[1,0]=None
Y[2,0]=None
#k=0の部分は使わないので後の可視化のため、0とした。

'''多分、以下変数は縦に並べるようにしたほうが資料と同じなのでそのようにする。'''

plt.rcParams['figure.figsize'] = (10, 8)


R = np.zeros((3,3))
R[0,0] = 0.01
R[1,1] = 0.01
R[2,2] = 0.01
'''あとで1に変える！！！'''
#観測データを得る際のシミュレーションとの誤差の分散
#すべての期に渡って、ノイズは同じであるとしている。それぞれの変数のノイズは互いに独立である。

Q = np.zeros((3,3))
Q[0,0] = 0.01
Q[1,1] = 0.01
Q[2,2] = 0.01
#シミュレーション内部のノイズ（撹乱項）の分散
#すべての期に渡って、ノイズは同じであるとしている。それぞれの変数のノイズは互いに独立であるとしている。

M = 20
#これがアンサンブルの大きさである。（20個データを毎回とってきて、分布に近似させる感じ。）

'''以下システム内のfの計算を行う。'''
K1=np.zeros(3)
K2=np.zeros(3)
K3=np.zeros(3)
K4=np.zeros(3)
u0 = np.zeros(3)
u0[0] = 0.01
u0[1] = 0.01
u0[2] = 0.01

x0 = u0

def g(x,y,z):
    output = np.zeros(3)
    s = 10
    b = 8.0 / 3.0
    r = 28
    output[0] = -s*(x-y)
    output[1] = r*x-y-x*z
    output[2] = x*y-b*z
    return output
deltat = 0.01
K1 = g(x0[0],x0[1],x0[2])
K2 = g(x0[0]+deltat*2**(-1)*K1[0], x0[1]+deltat*2**(-1)*K1[1], x0[2]+deltat*2**(-1)*K1[2])
K3 = g(x0[0]+deltat*2**(-1)*K2[0], x0[1]+deltat*2**(-1)*K2[1], x0[2]+deltat*2**(-1)*K2[2])
K4 = g(x0[0]+deltat*K3[0],x0[1]+deltat*K3[1],x0[2]+deltat*K3[2])
temp = K1+ 2*K2 + 2*K3 + K4
systemAddition = deltat*6**(-1)*temp
#これはシミュレーション内部で、x,y,z に対してどれだけ加算がなされるかを示す部分である。

systemAdditionCalc = np.array(systemAddition.tolist()*M)
#アンサンブルの大きさだけ作成した。
''''''

x_tilda = np.zeros((3*M, n_iter))
#各期のフィルター前の予測値がそれぞれアンサンブルメンバーごとに入る感じ。列は期を表し、行にアンサンブルメンバーが入る。
#このメンバーはM=2なら、x,y,z,x,y,z (実際は縦）というように格納される３つの塊が一つのアンサンブルメンバーである。


x_hat = np.zeros((3*M,n_iter))
#各期のフィルター後の予測値がそれぞれのアンサンブルメンバーごとに入る感じ。x_tildaの各要素が更新される感じ。


x_hat[:,0]=1
#初期値を与えた。最初のアンサンブルのイメージ。全部1とした。つまり０期目のアンサンブルを作った。

x_bar=np.zeros((3, n_iter))
#計算のためのスペース。列が期をあらわし、各列に変数が縦にはいっているようなかんじ。

P_bar=np.zeros((3, 3))
#計算のためのスペース。これは毎回の上書きを想定。

x_hat_output=np.zeros((3, n_iter))
#推定値を利用する場合に、アンサンブルの平均値を用いて計算されたもの、

K_bar=np.zeros((3,3))
# カルマンゲイン。これは変数の個数の列と行を持つ正方行列になると考えられる。

H = np.zeros((3,3))
H[0,0] = 1
H[1,1] = 1
H[2,2] = 1
#Hは対角行列を想定した。

for k in range(1,n_iter):#n_iter回更新されるn_iter個データはあるので。1, 2, ..., n_iter-1 まで出てくる
    '''予測'''
    v_temp = []
    for i in range(0,3*M):
        v_temp.append(np.random.normal(0, Q[0,0]**(0.5)))
        #本来は、各変数に対して撹乱項を計算すべきであるが、全部同じ分散なのでまとめて作成した。
        v = np.array(v_temp)
   # v = np.reshape(v,(15,1))    #これで縦ベクトルで、上からアンサンブルメンバー1に関する撹乱項3つ、メンバー2に関する撹乱項3つ、、という感じで収納されている。
   # x_hat_kminus = np.reshape(x_hat[:,k-1],(15,1))
    x_tilda[:,k] =x_hat[:,k-1] + systemAdditionCalc + v


    print('{0}期の予測分布（フィルター前）のアンサンブルは{1}'.format(k,x_tilda[:,k]))
    #前の期のアンサンブルを今期のアンサンブルとする。
    #ここは(15,)のarrayで計算している。
    #縦ベクトルではないが、こうしないとx_tildaの要素に上手く格納できなかった。


    '''フィルタリング'''
    reshaped_x_tilda = np.reshape(x_tilda[:,k], (M,3)).transpose()
    #これは、k期のアンサンブルメンバーが各列ごとに入っている感じ。M=2で、x = (x,y,z) の変数３つであれば、
    #0列目には最初のアンサンブルメンバーのx,y,z 、1列目には次のアンサンブルメンバーのx,y,z が入っている。
    summed_x_tilda = np.sum(reshaped_x_tilda, axis=1)
    x_bar[:,k] = M**(-1) * summed_x_tilda
    #(3,)の形に格納した。縦ベクトルではないが、便宜的にこれでOK

    tilda_bar = reshaped_x_tilda - np.reshape(x_bar[:,k], (3,1))
    #3行M列である。
    summed_tilda_bar = np.dot(tilda_bar, tilda_bar.transpose())
    #標本分散の計算であるが、試した結果これで良いっぽいぞ。
    #分解してそれぞれのアンサンブルメンバーごとにやる必要はなさおう。これは3行3列になる。

    P_bar = (M-1)**(-1)*summed_tilda_bar


    first = np.dot(P_bar, H.transpose())
    second_1 = np.dot(H, P_bar)
    second_2 = np.dot(second_1,H.transpose())
    second_inverse = np.linalg.inv(second_2 + R)

    K_bar = np.dot(first, second_inverse)

    #最後は逆行列の計算

    u_temp = []
    for i in range(0,3*M):
        u_temp.append(np.random.normal(0, Q[0,0]**(0.5)))
        #本来は、各変数に対して撹乱項を計算すべきであるが、全部同じ分散なのでまとめて作成した。
    u = np.array(u_temp)
    #これで縦ベクトルで、上からアンサンブルメンバー1に関する撹乱項3つ、メンバー2に関する撹乱項3つ、、という感じで収納されている。

    #ここでは、一つひとつのアンサンブルについて行列計算しないといけない。HやK_barは3*3の行列である。
    for i in range(0,M):
        inner = np.reshape(Y[:,k],(3,1)) - np.dot(H,np.reshape(x_tilda[i*3:(i+1)*3,k],(3,1)))+ np.reshape(u[i*3:(i+1)*3],(3,1))
        x_hat[i*3:(i+1)*3, k] = x_tilda[i*3:(i+1)*3,k] + np.reshape(np.dot(K_bar,inner),(3,))
        #最後に行列の形にしてしまうと上手く格納できないので(3,)の形にした。
    xSum = 0
    ySum = 0
    zSum = 0
    for i in range(0,M):
        xSum = xSum + x_hat[i*3,k]
        ySum = ySum + x_hat[i*3+1,k]
        zSum = zSum + x_hat[i*3+2,k]
    x_hat_output[0,k] = M**(-1)*xSum
    x_hat_output[1,k] = M**(-1)*ySum
    x_hat_output[2,k] = M**(-1)*zSum


'''以下Estimationとしては、各期間のそれぞれの変数ごとの平均を用いたことを明示する。'''


plt.figure()
plt.plot(true_x ,'k+')
plt.title('true scenario', fontweight='bold')
plt.xlabel('time')
plt.ylabel('the value of x')

plt.figure()
plt.plot(true_y,'k+')
plt.title('true scenario', fontweight='bold')
plt.xlabel('time')
plt.ylabel('the valus of y')

plt.figure()
plt.plot(true_z,'k+')
plt.title('true scenario', fontweight='bold')
plt.xlabel('time')
plt.ylabel('the value of z')



plt.figure()
plt.plot(Y[0,:],'k+',label='noisy measurements')
plt.plot(x_hat_output[0,:],'b-',label='a posteri estimate of x')
plt.legend()
plt.title('Estimation', fontweight='bold')
plt.xlabel('the number of observation')
plt.ylabel('the value of x')

plt.figure()
plt.plot(Y[1,:],'k+',label='noisy measurements')
plt.plot(x_hat_output[1,:],'b-',label='a posteri estimate of y')
plt.legend()
plt.title('Estimation', fontweight='bold')
plt.xlabel('the number of observation')
plt.ylabel('the valus of y')

plt.figure()
plt.plot(Y[2,:],'k+',label='noisy measurements')
plt.plot(x_hat_output[2,:],'b-',label='a posteri estimate of z')
plt.legend()
plt.title('Estimation', fontweight='bold')
plt.xlabel('the number of observation')
plt.ylabel('the value of z')

plt.show()
