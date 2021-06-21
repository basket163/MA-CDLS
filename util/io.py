import os
import time
import pandas as pd
import numpy as np

import globalVar
from globalVar import enumVar
import util


def modifyFolder(old,new,timeSlot):
    folderPath = os.getcwd()+globalVar.pathSep#"\\"
    folderPathNew = globalVar.ResultData
    #judge folderPath
    isExists=os.path.exists(folderPathNew)
    if not isExists:
        os.makedirs(folderPathNew)
    fileTime = ioGetTimeAsFileName()
    folderResult = folderPath  + old
    curData = globalVar.get_value(enumVar.currentFolder)
    curPara = globalVar.get_value(enumVar.nowEvoPara) + "runIdx"+str(globalVar.runIdx)+ "slot"+str(timeSlot)
    prefix = util.ioGetTimeAsFileName() + '_' + curData +'_' #+ curPara + "_"
    if os.path.exists(folderResult):
        os.rename(os.path.join(folderPath,old),os.path.join(folderPathNew,prefix+new+curPara))
    #if os.path.exists(folderPath+new):
    #    print("new path success")
    #input()

def ioGetPath():
    folderPath = os.getcwd()+globalVar.pathSep#"\\"
    return folderPath

def setRunTime():
    """ record program start time. """
    runTime = time.strftime("%Y-%m%d-%H%M", time.localtime())
    return runTime

def ioGetTimeAsFileName():
    return globalVar.runTime

def ioSaveNumpy2Csv(array,name):
    np.savetxt(globalVar.ResultImg + name + '.csv',array,delimiter=',',fmt='%.4f')

def saveArray2Excel(array,folder,filename):
    df = pd.DataFrame(array)
    df.to_csv(r''+folder+filename,encoding='gbk')

def convert_dict_to_table(dict_data):
    # dict_data is {col_key: [val]}
    list_title = []
    list_val = []
    len_dict = len(dict_data)
    for key, val in dict_data.items():
        list_title.append(key)
        list_val.append(val)
    table = np.array(list_val).T.reshape((-1,len_dict))
    return table, list_title