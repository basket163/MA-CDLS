import numpy as np
import matplotlib.pyplot as plt
#import pylab
from pylab import mpl
import os

import edge
import globalVar

def drawSimple(dictGraphData):
    for k,v in dictGraphData.items():
        #print("{}\n{}\n".format(k,v))
        xlist = v[0,:]
        ylist = v[1,:]
        _lbl = k
        _mrk = 'o'
        ins = ax.plot(xlist,ylist,label = _lbl,marker=_mrk)
        #lns += ins
        lns.append(ins)
        labs.append(k)

    x = np.arange(0, 2*np.pi, 0.02)  
    y = np.sin(x)  
    y1 = np.sin(2*x)  
    y2 = np.sin(3*x)  
    ym1 = np.ma.masked_where(y1 > 0.5, y1)  
    ym2 = np.ma.masked_where(y2 < -0.5, y2)  
    
    lines = plt.plot(x, y, x, ym1, x, ym2, 'o')  
    #设置线的属性
    plt.setp(lines[0], linewidth=1)  
    plt.setp(lines[1], linewidth=2)  
    plt.setp(lines[2], linestyle='-',marker='^',markersize=4)  
    #线的标签
    plt.legend(('No mask', 'Masked if > 0.5', 'Masked if < -0.5'), loc='upper right')  
    plt.title('Masked line demo')  
    plt.show()  

def drawMethodCompare(dictGraphData,folderName,titleY):
    mpl.rcParams['font.sans-serif'] = ['Times New Roman'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

    samplenum1=np.arange(25,500+2,25)
    x25 = samplenum1
    samplenum2=np.arange(10,200+2,10)
    x10 = samplenum2
    samplenum3=np.arange(2,40+2,2)
    x2 = samplenum3
    #print("\nx25\n{}\nx10\n{}\nx2\n{}".format(x25,x10,x2))

    accuracy10sigmoid_test=[0.863, 0.898, 0.964, 0.985, 0.975, 0.985, 0.989, 0.992, 0.992, 0.99, 0.989, 0.991, 0.988, 0.995, 0.994, 0.995, 1.0, 0.999, 0.996, 0.995]
    accuracy10tanh_test=[0.88, 0.968, 0.99, 0.985, 0.987, 0.988, 0.979, 0.986, 0.989, 0.988, 0.99, 0.987, 0.985, 0.993, 0.992, 0.993, 0.989, 0.99, 0.981, 0.991]
    accuracy10relu_test=[0.931, 0.9, 0.933, 0.947, 0.953, 0.967, 0.98, 0.985, 0.973, 0.981, 0.985, 0.985, 0.986, 0.979, 0.985, 0.984, 0.984, 0.982, 0.978, 0.976]
    #print("len x25:{}, len ac10sig:{}".format(len(x25),len(accuracy10sigmoid_test)))
    #input()
    #面向对象的绘图方式
    rect1 = [0.14, 0.35, 0.77, 0.6]
    fig,ax = plt.subplots()
    ax.figsize=(48,48)
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
    plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
    plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.rcParams['figure.dpi'] = 300 #分辨率

    #dic to ins
    print("\n in drawing " + folderName)
    listIns = []
    labs = []
    #lns = []
    idx = 0
    _marker = edge.cfg.marker
    for k,v in dictGraphData.items():
        #print("{}\n{}\n".format(k,v))
        xlist = v[0,:]
        ylist = v[1,:]
        _lbl = k
        _mrk = _marker[idx]
        ins = ax.plot(xlist,ylist,label = _lbl,marker=_mrk)
        #lns += ins
        listIns.append(ins)
        labs.append(k)
        idx = idx+1

    lns = ""
    dictNum = len(dictGraphData)
    if dictNum == 2:
        lns = listIns[0]+listIns[1]
    if dictNum == 3:
        lns = listIns[0]+listIns[1]+listIns[2]
    if dictNum == 4:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]
    if dictNum == 5:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]
    if dictNum == 6:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]+listIns[5]
    if dictNum == 7:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]+listIns[5]+listIns[6]
    if dictNum == 8:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]+listIns[5]+listIns[6]+listIns[7]
    if dictNum == 9:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]+listIns[5]+listIns[6]+listIns[7]+listIns[8]
    if dictNum == 10:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]+listIns[5]+listIns[6]+listIns[7]+listIns[8]+listIns[9]
    if dictNum == 10:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]+listIns[5]+listIns[6]+listIns[7]+listIns[8]+listIns[9]
    if dictNum == 11:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]+listIns[5]+listIns[6]+listIns[7]+listIns[8]+listIns[9]+listIns[10]
    if dictNum == 12:
        lns = listIns[0]+listIns[1]+listIns[2]+listIns[3]+listIns[4]+listIns[5]+listIns[6]+listIns[7]+listIns[8]+listIns[9]+listIns[10]+listIns[11]
    labs = [l.get_label() for l in lns]
    
    '''ins0=ax.plot(x10,accuracy10tanh_test, label = 'tanh',marker='o')
    ins1=ax.plot(x10,accuracy10relu_test, label = 'relu',marker='s')
    ins2=ax.plot(x10,accuracy10sigmoid_test, label = 'sigmoid',marker='v')
    lns = ins0+ins1+ins2
    print("{}\n{}".format(ins0,lns))
    input()'''
    #labs = [l.get_label() for l in lns]

    ax.legend(lns, labs, loc="lower right")#loc="lower right" 图例右下角
    ax.set_xlabel("Time slot")
    ax.set_ylabel(titleY)
    #ax.set_title("xxx0-10")
    ax.set_xticks(xlist)
    #ax.set_yticks([0.7,0.8, 0.9,0.95,1.0])
    #ax.grid()
    folderPath = os.getcwd()+globalVar.pathSep#"\\"
    path = folderPath+"record"+globalVar.pathSep+folderName+'.png'
    plt.savefig(path)
    #plt.savefig(folderName+"-latency.png")
    
    if(edge.cfg.showPic == 1):
        plt.show()
    else:
        plt.close()
    #plt.savefig('kdd-iteration.eps', dpi=1000, bbox_inches='tight')
    