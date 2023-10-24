import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft

"""
Filling in the file name defaults to the last feature as the decomposition sample
"""
datatype = 'NP'
filename = './datasets/{}.csv'.format(datatype)
vmdsavename = './result/VMD_{}.csv'.format(datatype)
savename = './result/VMD_SE_{}.csv'.format(datatype)
f = pd.read_csv(filename,usecols=[3])

plt.plot(f.values)

alpha = 7000  # moderate bandwidth constraint
tau = 0.  # noise-tolerance (no strict fidelity enforcement)
K = 8  # 3 modes
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

"""  
alpha、tau、K、DC、init、tol 六个输入参数的无严格要求； 
alpha 带宽限制 经验取值为 抽样点长度 1.5-2.0 倍； 
tau 噪声容限 ；
K 分解模态（IMF）个数； 
DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1； 
init 初始化 w 值，当初始化为 1 时，均匀分布产生的随机数； 
tol 控制误差大小常量，决定精度与迭代次数
"""

u, u_hat, omega = VMD(f.values, alpha, tau, K, DC, init, tol)

"""
u表示分解模式的集合，u_hat表示模式的光谱范围，omega 表示估计模态的中心频率
"""
plt.figure()

plt.plot(u.T)
plt.title('Decomposed modes')
plt.show()
fig1 = plt.figure()
plt.plot(f.values)

fig1.suptitle('Original input signal and its components')
plt.show()
# 中心模态
fi = pd.read_csv(filename,index_col=0)
colorchose = {0:'coral',1:'orange',2:'steelblue'}
for i in range(3):
    a = fi.values[:,i]
    plt.figure(figsize=(6, 1), dpi=300)
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.plot(a[200:2000], color=colorchose[i])
    plt.savefig('./figure/dim{}.jpg'.format(i), bbox_inches='tight')
# 根据中心频率可以求分解子序列的个数
b=0
for i in range(3):
    a = fi.values[:,i]
    plt.figure(figsize=(6, 1), dpi=300)
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    b=b+a[40000:41000]
    plt.plot(a[50000:51000], color='grey')
    plt.savefig('./figure/dimuse3{}.jpg'.format(i), bbox_inches='tight')

pdflie = pd.read_csv(filename,usecols=[0])

# 保存子序列数据到文件中

for i in range(K):
    a = u[i, :]
    dataframe = pd.DataFrame({'v{}'.format(i + 1): a})
    pdflie = pd.concat([pdflie,dataframe],axis=1)
    # dataframe.to_csv("./result/VMDban-%d.csv" % (i + 1), index=False, sep=',')
    # plt.subplot(K,1,i+
    plt.figure(figsize=(6, 1),dpi=300)
    # ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    # plt.axis('off')  # 去掉坐标轴标数
    plt.plot(a[200:2000],color = 'b')
    plt.savefig('./figure/imf{}.jpg'.format(i),bbox_inches='tight')
    # plt.show()
dataframe = pd.DataFrame({'Price':f.values.ravel()})
"""

"""
pdflie = pd.concat([pdflie,dataframe],axis=1)
pdflie.to_csv(vmdsavename, index=False, sep=',')

for i in range(K):
    a = u_hat[i, :]
    plt.subplot(K, 1, i + 1)
    plt.plot(a)
plt.show()

from sampen import sampen2
"""
VMD模态SE
"""
import os
if(os.path.exists(savename)):
    f = pd.read_csv(savename, index_col=0)
    for i in range(3):
        a = f.values[:, i]
        dataframe = pd.DataFrame({'v{}'.format(i + 1): a})
        plt.figure(figsize=(6, 1),dpi=300)
        plt.plot(a[200:500])
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        # plt.axis('off')  # 去掉坐标轴
        plt.savefig('./figure/flu{}.jpg'.format(i), bbox_inches='tight')
        # plt.show()
        pdflie = pd.concat([pdflie,dataframe],axis=1)
else:
    f = pd.read_csv(vmdsavename,index_col=0)


    print(f.values.shape)
    reset=0
    se_set = {}
    se_list =[]
    for i in range(0,f.values.shape[1]-1):



        sampen_of_series = sampen2(f.values[:10000, i])
        se_set.update({i:sampen_of_series[2][1]})
        se_list.append(sampen_of_series[2][1])
    print(se_set)
    print(se_list)
    se_list.sort(reverse=True)
    print(se_list)
    out_put = np.zeros((f.values.shape[0],3))
    high =0
    middle =0
    low = 0
    for i in range(0,f.values.shape[1]-1):
        if se_set[i] in se_list[0:2]:
            high += f.values[:,i]
        elif se_set[i] in se_list[2:6]:
            middle += f.values[:, i]
        elif se_set[i] in se_list[6:8]:
            low += f.values[:, i]
    out_put[:,0] = high
    out_put[:,1] = middle
    out_put[:,2] = low
    pdflie = pd.read_csv(filename,usecols=[0])
    for i in range(3):
        a = out_put[:, i]
        dataframe = pd.DataFrame({'v{}'.format(i + 1): a})
        plt.figure(figsize=(6, 1))
        plt.plot(a)
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')  # 去掉坐标轴
        plt.show()
        pdflie = pd.concat([pdflie,dataframe],axis=1)
    for i in range(1,4):
        f = pd.read_csv(filename,usecols=[i])
        dataframe = pd.DataFrame({'feature{}'.format(i):f.values.ravel()})
        pdflie = pd.concat([pdflie, dataframe], axis=1)
    pdflie.to_csv(savename, index=False, sep=',')