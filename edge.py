import numpy as np
import copy
import yaml
import sys #sys.setrecursionlimit(1000000)
import pandas as pd
import os
import datetime
import globalVar
from scipy import stats

import util
import pkgDevice
import globalVar
from globalVar import enumVar
from globalVar import enumScene


def loadUser(config, dfUser):
    listUser = []
    userCount = len(dfUser)
    
    config.userCount = userCount
    
    for u in range(userCount):
        u1=dfUser.loc[u,'userId']
        u2=dfUser.loc[u,'userType']
        u3=dfUser.loc[u,'userInCell']
        u4=dfUser.loc[u,'profileInServ']
        u5=dfUser.loc[u,'requestUnit']
        
        clsTemp = pkgDevice.clsUser(u1,u2,u3,u4,u5)
        listUser.append(clsTemp)
    return listUser

def loadServ(config, dfServ):
    listServ = []
    servCount = len(dfServ)
    servIdList = []
    servIdCapa = []
    for s in range(servCount):
        s1=dfServ.loc[s,'serverId']
        s2=dfServ.loc[s,'servInCell']
        s3=dfServ.loc[s,'capa']
        s4=dfServ.loc[s,'haveUserNum']
        s5=dfServ.loc[s,'haveUserList']
        clsTemp = pkgDevice.clsServ(s1,s2,s3,s4,s5)
        listServ.append(clsTemp)
        servIdList.append(s1)
        servIdCapa.append(s3)
    config.servIdList = servIdList
    config.servIdCapa = servIdCapa
    return listServ

def getUserRequ(config, _userId):
    npUser = config.npUser #globalVar.get_value(enumVar.npUser)
    _userId = np.array(_userId)
    value = npUser[_userId-1,5]
    return value

def getServCapa(config, _servId):
    npServ = config.npServ #globalVar.get_value(enumVar.npServ)
    _servId = np.array(_servId)
    value = npServ[_servId-1,2]
    return value

def getServInCell(_servId):
    npServ = globalVar.get_value(enumVar.npServ)
    _servId = np.array(_servId)
    value = npServ[_servId,1]
    return value

def getServListInHop(config, hop, a_pInServ):
    matrix = config.cellMatrix #globalVar.get_value(enumVar.cellMatrix)

    #get p in row
    rowIdx = a_pInServ - 1
    row = matrix[rowIdx,:]
    row = row.astype('int32')
    idxLessThanHop = np.where(row<=hop)[0]
    cellNum = len(idxLessThanHop)
    cellLessThanHop = idxLessThanHop + np.ones(cellNum,int)
    return cellLessThanHop

def findBestServForUser(config, s,userIdx,servLessThanHop,scene):
    
    servNum = len(servLessThanHop)
    listObjV = np.zeros(servNum)
    
    for servIdx in np.arange(servNum):
        #replace the serv in scene
        testServ = servLessThanHop[servIdx]
        tmpScene = copy.deepcopy(scene)
        tmpScene[config._pInServ,userIdx] = testServ
        #compute scene
        cmni,cmpt,miga = computeScene(config, tmpScene)
        best_ObjV = statisticLatency(cmni,cmpt,miga)
        listObjV[servIdx] = best_ObjV
    # find the best personal evaluation    
    bestIdx = np.where(listObjV==np.min(listObjV))[0]
    # attention! bestIdx exists multi values
    bestServ = servLessThanHop[bestIdx]
    return bestServ

def getDictVal(arrayKey,dictPara):
    val = []
    for a in arrayKey:
        if( a in dictPara.keys()):
            v = dictPara[a]
            val.append(v)
        else:
            print(f'edge.py, line 130: {arrayKey},\n{dictPara}\na: {a}')
            input()
    return np.array(val)

def stadictUserNumInServ(config, _pInServ):
    #sta Serv 1->n have user num, uInCell = pInServ
    dictUserInServCount = {}
    servIdMin = config.servIdMin #globalVar.cfg.cellIdMin
    servIdMax = config.servIdMax #globalVar.cfg.servIdMax
    _pInServ = np.array(_pInServ)
    for x in range(servIdMin,servIdMax+1,1):
        y = np.sum(_pInServ==x)
        #print("{}:{}\n".format(x,y))
        #input()
        dictUserInServCount[x] = y
    #_pCount = dictUserInServCount[_pInServ]
    _pCount = getDictVal(_pInServ,dictUserInServCount)
    #print(_pCount)
    return _pCount
    #sta every serv  time

def staAllServUserNum(config, scene):
    allServUserNum = np.zeros(config.servCount)
    for x in range(config.servCount):
        for y in range(config.userCount):
            if(scene[config._pInServ,y] == x+1):
                allServUserNum[x] += 1
    return allServUserNum

def staAllServQueue(config, scene):
    allServQueue = np.zeros(config.servCount)
    for x in range(config.servCount):
        for y in range(config.userCount):
            if(scene[config._pInServ,y] == x+1):
                allServQueue[x] += (scene[config._userQueu,y] +
                                 scene[config._userQueuLast,y])
    return allServQueue

def get_similarity(ideal_a1, ideal_a2, ideal_a3, a1, a2, a3):
    #print(f'ideal a:{ideal_a1}, {ideal_a2}, {ideal_a3}')
    #print(f'compare a:{a1}, {a2}, {a3}')
    a = np.array([ideal_a1, ideal_a2, ideal_a3])
    b = np.array([a1, a2, a3])
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    sim = (np.matmul(a,b))/(ma*mb)
    return sim

def computeUserQueu(userQueuLast,userRequ,inServCapa,inServPflNum):
    inServPflNum_1 = np.maximum(inServPflNum,1)
    ret = userQueuLast + userRequ - np.true_divide(inServCapa,inServPflNum_1)
    #if array has val < 0, the queue is 0
    ret_0 = np.maximum(ret,0)
    return ret_0
    

def computePlacement(arrUserId,arrUserInCell,arrUserInLastCell,arrPinLastServ):
    #follow strategy
    pInServ = arrUserInCell
    return pInServ

def loadSlot(dfMove):
    listSlot = []
    #first col is userId
    colCount = dfMove.columns.size
    rowCount = len(dfMove)

    colUserId = dfMove.iloc[:,0]
    userId = colUserId.T
    for s in range(1,colCount):
        colSlot = dfMove.iloc[:,s]
        colSlotLast = dfMove.iloc[:,1] if s == 1 else dfMove.iloc[:,s-1]
        colProfile = copy.deepcopy(colSlot)
        #listSlot.append(temp.values)
        tensor = np.dstack((
        np.array(colUserId),
        np.array(colSlot),
        np.array(colProfile),
        np.array(colSlotLast)
        ))
        #minus []
        tensor = tensor[0]
        listSlot.append(tensor)
        #print(temp.values)
    return listSlot

def updateCurrentScene(config, updatedScene,newPInServ):
    #current scene, next scene
    #updatedScene = listScene[currentIdx]
    newPInServ = newPInServ.astype("int32")

    #pInServ = computePlacement(userId,uInCell,uInCellLast,pInServLast)
    pInServ = newPInServ
    inServPflNum = stadictUserNumInServ(config, pInServ) #cellId = servId
    inServCapa = getServCapa(config, pInServ) #cellId = servId

    userQueuLast = updatedScene[config._userQueuLast,:]
    userRequ = updatedScene[config._userRequ,:]
    userQueu = computeUserQueu(userQueuLast,userRequ,inServCapa,inServPflNum)
    
    updatedScene[config._pInServ,:] = pInServ
    updatedScene[config._inServPflNum,:] = inServPflNum
    updatedScene[config._inServCapa,:] = inServCapa
    updatedScene[config._userQueu,:] = userQueu

    return updatedScene

def updateNextScene(listScene,currentIdx):
    total = len(listScene)
    if currentIdx >= total:
        return
    nextIndx = currentIdx+1
    updatedScene = listScene[currentIdx]
    nextScene = listScene[nextIndx]

    _userQueuLast = globalVar.get_value(enumScene._userQueuLast)
    _userRequ = globalVar.get_value(enumScene._userRequ)
    _pInServ = globalVar.get_value(enumScene._pInServ)
    _inServPflNum = globalVar.get_value(enumScene._inServPflNum)
    _inServCapa = globalVar.get_value(enumScene._inServCapa)
    _userQueu = globalVar.get_value(enumScene._userQueu)
    _pInServLast = globalVar.get_value(enumScene._pInServLast)
    _inServPflNumLast = globalVar.get_value(enumScene._inServPflNumLast)
    _inServCapaLast = globalVar.get_value(enumScene._inServCapaLast)

    nextScene[_pInServLast,:] = updatedScene[_pInServ,:]
    nextScene[_inServPflNumLast,:] = updatedScene[_inServPflNum,:]
    nextScene[_inServCapaLast,:] = updatedScene[_inServCapa,:]
    nextScene[_userQueuLast,:] = updatedScene[_userQueu,:]
    return nextScene

def computeScene(config, newScene):
    """scene: 0 userId, 1 uInCell,  * 2 userRequ, 3 inServPflNum, 4 inServCapa,
    5 uInCellLast, * 6 userRequLast, 7 inServPflNumLast, 8 inServCapaLast, 9 pInServLast,
    10 userQueuLast, 11 userQueu, 12 pInServ (solution)
    """
    uInCell = newScene[1,:]
    pInServ = newScene[12,:]
    userQueuLast = newScene[10,:]
    userRequ = newScene[2,:]
    inServCapa = newScene[8,:]
    inServPflNum = newScene[3,:]
    uInCellLast = newScene[5,:]
    pInServLast = newScene[9,:]

    cmni = communicationLatency(config, uInCell, pInServ)
    cmpt = computationLatency(userQueuLast,userRequ,inServCapa,inServPflNum)
    miga = migrationLatency(config, uInCellLast,uInCell,pInServLast,pInServ)
    return cmni,cmpt,miga

def computeEnergy(config, newScene):
    """scene: 0 userId, 1 uInCell,  * 2 userRequ, 3 inServPflNum, 4 inServCapa,
    5 uInCellLast, * 6 userRequLast, 7 inServPflNumLast, 8 inServCapaLast, 9 pInServLast,
    10 userQueuLast, 11 userQueu, 12 pInServ (solution)
    """
    userId = newScene[0,:]
    #uInCell = newScene[1,:]
    pInServ = newScene[12,:]
    #userQueuLast = newScene[10,:]
    #userRequ = newScene[2,:]
    inServCapa = newScene[8,:]
    inServPflNum = newScene[3,:]
    #uInCellLast = newScene[5,:]
    #pInServLast = newScene[9,:]

    energy = computationEnergyConsume(config, userId,pInServ,inServCapa,inServPflNum)
    return energy

def initialSceneFunction(config, initialSlot):
    ###slot: 1 userId, 2 uInCell, 3 U inLastCell
    colUserId = initialSlot[:,0]
    colSlot = initialSlot[:,1]
    colSlotLast = initialSlot[:,2]
    #userCount = globalVar.get_value(enumVar.userCount)
    userCount = config.userCount
    """scene: 0 userId, 1 uInCell,  * 2 userRequ, 3 inServPflNum, 4 inServCapa,
    5 uInCellLast, * 6 userRequLast, 7 inServPflNumLast, 8 inServCapaLast, 9 pInServLast,
    10 userQueuLast, 11 userQueu, 12 pInServ (solution)
    """
    userId = colUserId.T
    uInCell = colSlot.T
    
    userRequ = getUserRequ(config, userId)
    inServPflNum = stadictUserNumInServ(config, uInCell) #cellId = servId
    inServCapa = getServCapa(config, uInCell) #cellId = servId
   
    uInCellLast = uInCell
    userRequLast = np.zeros(userCount,dtype=np.int)
    inServPflNumLast = np.zeros(userCount,dtype=np.int)
    inServCapaLast = np.zeros(userCount,dtype=np.int)
    pInServLast = uInCell
    userQueuLast = np.zeros(userCount,dtype=np.int)
    userQueu = np.zeros(userCount,dtype=np.int)
    pInServ = uInCell   
    
    scene = np.vstack((userId, uInCell,  userRequ, inServPflNum, inServCapa,
    uInCellLast, userRequLast, inServPflNumLast, inServCapaLast, pInServLast,
    userQueuLast, userQueu, pInServ))
    #print("initialScene")
    #print(scene)
    #input()
    #slice 
    return scene

def generateScene(config, slot,lastScene):
    ###slot: 1 userId, 2 uInCell, 3 U inLastCell
    colUserId = slot[:,0]
    colSlot = slot[:,1]
    colSlotLast = slot[:,2]
    #userCount = globalVar.get_value(enumVar.userCount)
    ######### compute slice
    """scene: 0 userId, 1 uInCell,  * 2 userRequ, 3 inServPflNum, 4 inServCapa,
    5 uInCellLast, * 6 userRequLast, 7 inServPflNumLast, 8 inServCapaLast, 9 pInServLast,
    10 userQueuLast, 11 userQueu, 12 pInServ (solution)
    """
    userId = colUserId.T
    uInCell = colSlot.T
    #pInServ = decidePlacement(userId,uInCell,uInLastCell,pInLastServ)
    userRequ = getUserRequ(config, userId)
    inServPflNum = stadictUserNumInServ(config, uInCell) #follow, cellId = servId
    inServCapa = getServCapa(config, uInCell) #follow, cellId = servId
    uInCellLast = lastScene[1,:]
    userRequLast = lastScene[2,:]
    inServPflNumLast = lastScene[3,:]
    inServCapaLast = lastScene[4,:]
    pInServLast = lastScene[12,:]
    userQueuLast = lastScene[11,:]
    userQueu = computeUserQueu(userQueuLast,userRequ,inServCapa,inServPflNum)
    pInServ = computePlacement(userId,uInCell,uInCellLast,pInServLast)
    
    newScene = np.vstack((userId, uInCell,  userRequ, inServPflNum, inServCapa,
    uInCellLast, userRequLast, inServPflNumLast, inServCapaLast, pInServLast,
    userQueuLast, userQueu, pInServ))
    return newScene


def demo():
    M = []
    N = []
    C = []
    S = []

    idxServer = range(1,19,1)
    M = [("M"+str(m)) for m in idxServer]
    #print(M)

    idxCell = range(1,19,1)
    C = [("C"+str(c)) for c in idxCell]
    #print(C)
    #print([m for m in M])
    #M = np.random.random(18)

    idxUser = range(1,31,1)
    U = [("u"+str(u)) for u in idxUser]
    S = [("s"+str(s)) for s in idxUser]
    #print(U)
    #print(S)

    npUser = np.arange(1,19,1)
    npServ = np.arange(1,31,1)
    #let user random in Cell
    #print("User: {}\n Cell: {}\n".format(U,C))
    location = np.random.randint(1,19,1)

    dictServ = {}
    for x in np.nditer(npUser):
        loc = np.random.randint(1,19,1)
        dictServ[loc]=x

    print(dictServ)

def getCellIdRange():
    r = np.arange(1,19)
    print(r)

def generateRandomSolution():
    # 1 col, n rows, with range
    userCount = globalVar.get_value(enumVar.userCount)
    a = np.random.randint(1,19,userCount)
    b = a.reshape(userCount,1)
    return b


def getCellDistance(config, cellx,celly):
    matrix = config.cellMatrix #globalVar.get_value(enumVar.cellMatrix)
    #print("matrix")
    #print(matrix)
    #input()
    row = cellx - 1
    col = celly - 1
    if not isinstance(row,int):
        row = row.astype('int32')
    if not isinstance(col, int):
        col = col.astype('int32')
    #print("{}\n{}".format(row, col))
    value = matrix[row,col]
    #print(value)
    #input()
    return value

def communicationLatency(config, userInCell, profileInServ):
    #getDistance between cellx and celly
    #result = util.getCellDistance(userInCell,profileInServ)
    userInCell = np.array((userInCell),dtype=np.int)
    #print("{}\n{}".format(userInCell, profileInServ))
    result = getCellDistance(config, userInCell,profileInServ)
    #print(result)
    avgVal = np.average(result)
    #print(avg)
    return avgVal

def compareMin(x,y):
    #print("compare")
    #print(x)
    #print(y)
    num = len(x)
    z = np.zeros(num)
    for i in range(num):
        z[i] = x[i] if x[i] < y[i] else y[i]
    #print(z)
    #input()
    return z

def computationLatency(userQueuLast,userRequ,inServCapa,inServPflNum):
    #if queu + requ > c/n, the computation amoutn is c/n
    #so computation amount = min(queu+requ, c/n)
    a = userQueuLast + userRequ
    inServPflNum_1 = np.maximum(inServPflNum,1)
    b = np.true_divide(inServCapa,inServPflNum_1)
    a = a.astype('float32')
    b = b.astype('float32')
    #print("a:\n{}\nb:\n{}\n".format(a,b))
    computationAmount = compareMin(a,b) #np.min(a,b)
    #print("c:\n{}\n".format(computationAmount))
    computationTime = np.true_divide(inServCapa,computationAmount)
    computationTime = computationTime.astype('float32')
    #print("computationTime:\n{}".format(computationTime))
    avgVal = np.average(computationTime)
    avgVal = np.around(avgVal, decimals=2)
    #print(avgVal)
    #input()
    return avgVal

def shrinkToOne(x):
    num = len(x)
    z = np.zeros(num)
    for i in range(num):
        z[i] = 1 if x[i] > 0 else 0
    #print(z)
    #input()
    return z

def computationEnergyConsume(config, userId,pInServ,inServCapa,inServPflNum):
    #inServPflNum: the number of profiles in the server which user located
    servIdMin = config.servIdMin #globalVar.get_value(enumVar.servIdMin)
    servIdMax = config.servIdMax #globalVar.get_value(enumVar.servIdMax)
    servIdList = np.arange(servIdMin,servIdMax+1,1)
    servNum = len(servIdList)
    servIdHasPflNum = np.zeros((servNum,),dtype = int)

    for uInServId in pInServ:
        uInServIdx = uInServId - 1
        servIdHasPflNum[uInServIdx] += 1

    isServhavePfl = shrinkToOne(servIdHasPflNum)
    #servIdList = globalVar.get_value(enumVar.servIdList)
    servIdCapa = np.array(config.servIdCapa)
    #usingServNum = np.sum(isServhavePfl==1)
    isServhavePfl = isServhavePfl.astype('float32')
    servIdCapa = servIdCapa.astype('float32')
    fixRate = 0.2
    dynamicRate = 0.1
    consume = fixRate*servIdCapa + dynamicRate*isServhavePfl*servIdCapa
    #consume = 
    avgVal = np.mean(consume)
    avgVal = np.around(avgVal, decimals=2)

    return avgVal


def migrationLatency(config, uInCellLast,uInCell,pInServLast,pInServ):
    #print("migration")
    #print(pInServLast)
    #print(pInServ)
    result = getCellDistance(config, pInServLast,pInServ)
    #print(result)
    avgVal = np.average(result)
    #input()
    return avgVal

def computeFitness(config, slot):
    #1 userId; 2 inCell; 3 inServ; 4 inLastServ
    #print("\n---")
    rowNum = slot.shape[0]
    colNum = slot.shape[1]
    #print(slot)
    
    colUserId = slot[:,0]
    colUserId = colUserId.reshape(rowNum,1)
    #print(colUserId)
    colUserInCell = slot[:,1]
    colUserInCell = colUserInCell.reshape(rowNum,1)
    #print(colUserInCell)
    colProfileInServ = slot[:,2]
    colProfileInServ = colProfileInServ.reshape(rowNum,1)
    #print(colProfileInServ)
    colInLastCell = slot[:,3]
    colInLastCell = colInLastCell.reshape(rowNum,1)
  
    #computation latency, communication latency, migration latency
    com = computationLatency(colUserId,colUserInCell,colProfileInServ,colInLastCell)
    mi = migrationLatency(colUserId,colUserInCell,colProfileInServ,colInLastCell)
    avgLatency = communicationLatency(config, colUserInCell, colProfileInServ)
    return avgLatency
    
def computeSolution(config, slot, solution):
    userCount = globalVar.get_value(enumVar.userCount)
    #replace col 3 with solution
    replacedSlot = copy.deepcopy(slot)
    replacedSlot[:,2] = solution.reshape(userCount)
    return computeFitness(config, replacedSlot)

def compute(config, phenSolutions):
    slot = globalVar.get_value(enumVar.currentSlot)
    popSize = globalVar.get_value(enumVar.nowPopSize)
    userCount = globalVar.get_value(enumVar.userCount)
    #a solution is a column
    solutions_t = phenSolutions.T
    #print(solutions_t)
    #print("test recombine")
    listSolution = []
    #sta.append(phenSolutions[0,:].reshape(userCount,1))
    for i in range(popSize):
        recombine = phenSolutions[i,:].reshape(userCount,1)
        #print("recombine")
        #print(recombine)
        listSolution.append(recombine)
    #print("all")
    #print([s for s in sta])
    #input()
    
    listResult = []
    for s in listSolution:
        result = computeSolution(config, slot, s)
        listResult.append(result)
    #print("listResult")
    #print(listResult)
    #input()
    columnResult = (np.array(listResult)).reshape(popSize,1)
    #print("columnResult")
    #print(columnResult)
    #input()
    return columnResult

def aim(Phen, LegV): # define the aim function
    '''
    print("-----aim")
    print("--Phen")
    print(Phen)
    print("---LegV")
    print(LegV)
    input()
    '''
    evalution = compute(Phen)
    #print("evalution:")
    #print(evalution)
    #input()
    return [evalution, LegV]
    #return [np.array([evalution]).T, LegV]

def setInitGlobalVar():
    sys.setrecursionlimit(10000)
    #print(clsCfg.popSize)
    #print(clsCfg.color)

    globalVar._init()

    #corss platform: windows \\, linux /
    sep = os.path.sep
    if sep == "\\":
        globalVar.pathSep = "\\"
    if sep == "/":
        globalVar.pathSep = "/"

    util.setRunTime()

    globalVar.set_value(enumVar.listFolder,cfg.listFolder)
    globalVar.set_value(enumVar.listMethod,cfg.listMethod)
    globalVar.set_value(enumVar.listEvoPara,cfg.listEvoPara)

    globalVar.set_value(enumVar.cellMatrix,"")
    globalVar.set_value(enumVar.currentSlot,"")
    globalVar.set_value(enumVar.listScene,[])

    globalVar.set_value(enumVar.userCount,0)
    globalVar.set_value(enumVar.cellIdMin,1)
    globalVar.set_value(enumVar.cellIdMax,18)
    globalVar.set_value(enumVar.servIdMin,1)
    globalVar.set_value(enumVar.servIdMax,18)

    globalVar.set_value(enumVar.nowFitnessCount,0)

    #scene
    globalVar.set_value(enumScene._userId,int(0))
    globalVar.set_value(enumScene._uInCell,int(1))
    globalVar.set_value(enumScene._userRequ,int(2))
    globalVar.set_value(enumScene._inServPflNum,int(3))
    globalVar.set_value(enumScene._inServCapa,int(4))
    globalVar.set_value(enumScene._uInCellLast,int(5))
    globalVar.set_value(enumScene._userRequLast,int(6))
    globalVar.set_value(enumScene._inServPflNumLast,int(7))
    globalVar.set_value(enumScene._inServCapaLast,int(8))
    globalVar.set_value(enumScene._pInServLast,int(9))
    globalVar.set_value(enumScene._userQueuLast,int(10))
    globalVar.set_value(enumScene._userQueu,int(11))
    globalVar.set_value(enumScene._pInServ,int(12))

def recursiveScene(config, listSlot,n,scene):
    n += 1
    if n >= len(listSlot):
        return scene
    else:
        scene = generateScene(config, listSlot[n],scene)
        listScene = config.listScene #globalVar.get_value(enumVar.listScene)
        listScene.append(scene)
        return recursiveScene(config, listSlot,n,scene)

def avgStack(ret_valStack):
    run = 1
    runIdx = 0
    listScene = globalVar.get_value(enumVar.listScene)
    sceneNum = len(listScene)
    xlist = np.arange(1,sceneNum+1)
    #print("run {}, sceneNum {}".format(run,sceneNum))
    iterSet = np.zeros((run,sceneNum))
    callSet = np.zeros((run,sceneNum))
    delaySet = np.zeros((run,sceneNum))
    powerSet = np.zeros((run,sceneNum))

    iterSet[runIdx,:] = ret_valStack[1,:]
    callSet[runIdx,:] = ret_valStack[2,:]
    delaySet[runIdx,:] = ret_valStack[3,:]
    powerSet[runIdx,:] = ret_valStack[4,:]

    xlist = np.arange(1,sceneNum+1)

    meanIterSingle = iterSet.mean(axis=0)
    meanCallSingle = callSet.mean(axis=0)
    meanValSingle = delaySet.mean(axis=0)
    meanPowerSingle = powerSet.mean(axis=0)

    avgIter = np.vstack((xlist,meanIterSingle))
    avgCall = np.vstack((xlist,meanCallSingle))
    avgDelay = np.vstack((xlist,meanValSingle))
    avgPower = np.vstack((xlist,meanPowerSingle))

    return avgIter,avgCall,avgDelay,avgPower

def runMulti(afunction):
    userCount = globalVar.get_value(enumVar.userCount)
    #print("runMulti userCount {}".format(userCount))
    listScene = globalVar.get_value(enumVar.listScene)
    run = cfg.runNum
    sceneNum = len(listScene)
    xlist = np.arange(1,sceneNum+1)
    #print("run {}, sceneNum {}".format(run,sceneNum))
    delaySet = np.zeros((run,sceneNum))
    powerSet = np.zeros((run,sceneNum))
    iterSet = np.zeros((run,sceneNum))
    callSet = np.zeros((run,sceneNum))
    #phenSet = np.zeros((run*sceneNum,userCount),dtype=np.int)
    phenSetWithHead = np.arange(1,userCount+1)
    #phenSet = ""
    #phenSet = np.zeros((run*sceneNum,userCount))
    for runIdx in np.arange(run):
        print("--runIndex: {}".format(runIdx+1),)
        globalVar.runIdx = runIdx+1
        listScene_single = copy.deepcopy(listScene)
        ret_valStack, ret_phen = afunction() #pkgMethod.conductSceneSingleOP(listScene_single)
        iterSet[runIdx,:] = ret_valStack[1,:]
        callSet[runIdx,:] = ret_valStack[2,:]
        delaySet[runIdx,:] = ret_valStack[3,:]
        powerSet[runIdx,:] = ret_valStack[4,:]
        phenSetWithHead = np.vstack((phenSetWithHead,ret_phen[1:,:]))
    phenSet = phenSetWithHead[1:,:]
    minVal = delaySet.min(axis=0)
    minValIdx = delaySet.argmin(axis=0)
    
    minPower = powerSet.min(axis=0)
    minPowerIdx = powerSet.argmin(axis=0)

    minIter = iterSet.min(axis=0)
    minIterIdx = iterSet.argmin(axis=0)

    meanValSingle = delaySet.mean(axis=0)
    meanPowerSingle = powerSet.mean(axis=0)
    meanIterSingle = iterSet.mean(axis=0)
    meanCallSingle = callSet.mean(axis=0)

    avgDelay = np.vstack((xlist,meanValSingle))
    avgPower = np.vstack((xlist,meanPowerSingle))
    avgIter = np.vstack((xlist,meanIterSingle))
    avgCall = np.vstack((xlist,meanCallSingle))

    #保留原始值，计算p-value

    return avgIter,avgCall,avgDelay,avgPower

def statisticLatency(cmni,cmpt,miga):
    #print("cmni {}, cmpt {}, miga {}".format(cmni,cmpt,miga))
    #input()
    result = np.around(cmni + cmpt + miga, decimals=2)
    return result

def methodStart():
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,"%Y-%m-%d %H:%M:%S")
    print("*begin at {}".format(time_str))
    return curr_time

def methodEnd(startTime):
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,"%Y-%m-%d %H:%M:%S")
    print("**end at {}\n".format(time_str))

    #dtStart = datetime.datetime.strptime(str(startTime),"%Y-%m-%d %H:%M:%S")
    #dtEnd = datetime.datetime.strptime(str(time_str),"%Y-%m-%d %H:%M:%S")
    #usedTime = (time_str - startTime).seconds
    #m, s = divmod(timestamp, 60)
    #h, m = divmod(m, 60)
    #difftime = "%02d时%02d分%02d秒" % (h, m, s)
    #print("used time {}".format(difftime))

def runMain(folder):
#if __name__ == '__main__':
    #setInitGlobalVar()
    #folder = "init"
    print("\n#####run data folder: {}".format(folder))
    globalVar.set_value(enumVar.currentFolder,folder)

    data = globalVar.dataFolder
    sep = globalVar.pathSep

    fileUser = "./"+folder+"/user.csv"
    fileServ = "./"+folder+"/server.csv"
    fileMove = "./"+folder+"/move.csv"
    fileHop = "./"+folder+"/hopMatrix.xlsx"

    fileUser = data+sep + folder+sep + "user.csv"
    fileServ = data+sep + folder+sep + "server.csv"
    fileMove = data+sep + folder+sep + "move.csv"
    fileHop = data+sep + folder+sep + "hopMatrix.xlsx"

    if not os.path.exists(fileUser):
        print(f'{fileUser} not exist')
    if not os.path.exists(fileServ):
        print(f'{fileServ} not exist')
    if not os.path.exists(fileMove):
        print(f'{fileMove} not exist')
    if not os.path.exists(fileHop):
        print(f'{fileHop} not exist')

    dfUser = util.readCsv(fileUser)
    dfServ = util.readCsv(fileServ)
    npUser = np.array(dfUser)
    npServ = np.array(dfServ)
    dfMove = util.readCsv(fileMove)
    matrix = util.readExcel2Matrix(fileHop)
    globalVar.set_value(enumVar.dfUser,dfUser)
    globalVar.set_value(enumVar.dfServ,dfServ)
    globalVar.set_value(enumVar.npUser,npUser)
    globalVar.set_value(enumVar.npServ,npServ)
    globalVar.set_value(enumVar.cellMatrix,matrix)

    listClsUser = loadUser(dfUser)
    listClsServ = loadServ(dfServ)
    globalVar.set_value(enumVar.listClsUser,listClsUser)
    globalVar.set_value(enumVar.listClsServ,listClsServ)

    #important operation!
    userCount = len(listClsUser)
    globalVar.set_value(enumVar.userCount,userCount)

    #print("userCount is {}".format(userCount))
    #input()
    if userCount == 0:
        print("input error, userCount is 0")
        input()

    listSlot = loadSlot(dfMove)
    #conductSingleObjective(listSlot)

    listScene = globalVar.get_value(enumVar.listScene)
    listScene.clear()

    initialScene = initialSceneFunction(listSlot[0])
    #print("initialScene:\n{}".format(initialScene))
    recursiveScene(listSlot,0,initialScene) #return listScene
    listScene = globalVar.get_value(enumVar.listScene)
   
    #use follow strategy for initial test
    '''index = 0
    for ls in listScene:
        index += 1
        print("scene{}\n{}".format(index,ls))
        cmni,cmpt,miga = computeScene(ls)
        print("cmni: {:.2f}, cmpt: {:.2f}, miga: {:.2f}\n".format(cmni,cmpt,miga))
    print("test ok ---\n")'''

    #conductMethodList = ["Dealy Follow","SingleOP","Lyapnov"]
    conductMethodList = globalVar.get_value(enumVar.listMethod)
    method = ""
    dictResult = {}
    dictPower = {}
    dictGen = {}
    dictCall = {}
    '''
    dictResult["Instant Follow"] = valFollow
    dictResult["Dealy Follow"] = valDealyFollow
    dictResult["Nash"] = valNash
    dictResult["SingleOP"] = avgValSingle#[0:2,:]
    dictResult["Lyapnov"] = avgValLyap#[0:2,:]
    dictResult["Community"] = avgValComu#[0:2,:]
    dictResult["Recommend"] = avgValRcmd#[0:2,:]
    '''

    # methods: follow, singleOP, multiOP, 
    #(1)
    method = "Instant Follow"
    if(method in conductMethodList):
        startTime = methodStart()
        listScene_follow = copy.deepcopy(listScene)
        valFollow, phenFollow = pkgMethod.conductFollow(listScene_follow)
        avgIterFollow,avgCallFollow,avgDelayFollow,avgPowerFollow = avgStack(valFollow)
        print("\n---(1) method instant follow\n{}\n--\n{}\n---".format(valFollow,phenFollow))
        dictResult[method] = avgDelayFollow
        dictPower[method] = avgPowerFollow
        methodEnd(startTime)

    #(2)
    method = "Dealy Follow"
    if(method in conductMethodList):
        startTime = methodStart()
        listScene_delayFollow = copy.deepcopy(listScene)
        valDealyFollow, phenDealyFollow = pkgMethod.conductDealyFollow(listScene_delayFollow)
        avgIterDlFollow,avgCallDlFollow,avgDelayDlFollow,avgPowerDlFollow = avgStack(valDealyFollow)
        print("\n---(2) method delay follow\n{}\n--\n{}\n---".format(valDealyFollow,phenDealyFollow))
        dictResult[method] = avgDelayDlFollow
        dictPower[method] = avgPowerDlFollow
        methodEnd(startTime)

    #(3)
    method = "Nash"
    if(method in conductMethodList):
        startTime = methodStart()
        listScene_nash = copy.deepcopy(listScene)
        valNash, phenNash = pkgMethod.conductNash(listScene_nash)
        avgIterNash,avgCallNash,avgDelayNash,avgPowerNash = avgStack(valNash)
        print("\n---(3) method nash\n{}\n--\n{}\n---".format(valNash, phenNash))
        dictResult[method] = avgDelayNash
        dictPower[method] = avgPowerNash
        methodEnd(startTime)

    #(4)
    '''
    funcSingle = pkgMethod.conductSceneSingleOP
    run = cfg.runNum
    sceneNum = len(listScene)
    xlist = np.arange(1,sceneNum+1)
    #print("run {}, sceneNum {}".format(run,sceneNum))
    valSet = np.zeros((run,sceneNum))
    iterSet = np.zeros((run,sceneNum))
    #phenSet = np.zeros((run*sceneNum,cfg.userCount),dtype=np.int)
    phenSetWithHead = np.arange(1,cfg.userCount+1)
    #phenSet = ""
    #phenSet = np.zeros((run*sceneNum,cfg.userCount))
    for runIdx in np.arange(run):
        #listScene_single = copy.deepcopy(listScene)
        run_valSingle, phenSingle = funcSingle() #pkgMethod.conductSceneSingleOP()
        valSet[runIdx,:] = run_valSingle[1,:]
        iterSet[runIdx,:] = run_valSingle[2,:]
        phenSetWithHead = np.vstack((phenSetWithHead,phenSingle[1:,:]))
        #print("\n---(4).{} method single\n{}\n--\n{}\n---".format(runIdx+1,run_valSingle,phenSingle))
    phenSet = phenSetWithHead[1:,:]
    #print("valSet\n{}\niterSet\n{}".format(valSet,iterSet))
    #print("phenSet\n{}".format(phenSet))
    minVal = valSet.min(axis=0)
    minValIdx = valSet.argmin(axis=0)
    #print("minVal\n{}".format(minVal))
    #print("minValIdx\n{}".format(minValIdx))
    minIter = iterSet.min(axis=0)
    minIterIdx = iterSet.argmin(axis=0)
    #print("minIter\n{}".format(minIter))
    #print("minIterIdx\n{}".format(minIterIdx))

    meanValSingle = valSet.mean(axis=0)
    meanIterSingle = iterSet.mean(axis=0)
    valSingle = np.vstack((xlist,meanValSingle))
    #print("meanVal\n{}".format(meanVal))
    #print("meanIter\n{}".format(meanIter))
    #input()
    '''
    method = "SingleOP"
    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterSingle,avgCallSingle,avgValSingle,avgPowerSingle = runMulti(pkgMethod.conductSceneSingleOP)
            print("\n---(4) method single {}\n{}\n-\n{}\n---".format(para,avgValSingle,avgIterSingle))
            dictResult[method+para] = avgValSingle#[0:2,:]
            dictPower[method+para] = avgPowerSingle
        methodEnd(startTime)
    
    #(5)
    #listScene_lyap = copy.deepcopy(listScene)
    #valLyap, phenLyap = pkgMethod.conductLyapSingleOP()
    method = "Lyapnov"
    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterLyap,avgCallLyap,avgValLyap,avgPowerLyap = runMulti(pkgMethod.conductLyapSingleOP)
            print("\n---(5) method lyapunov single {}\n{}\n-\n{}\n---".format(para,avgValLyap,avgIterLyap))
            dictResult[method+para] = avgValLyap#[0:2,:]
            dictPower[method+para] = avgPowerLyap
        methodEnd(startTime)

    #(6)
    #listScene_comu = copy.deepcopy(listScene)
    #valComu, phenComu = pkgMethod.conductComuSingleOP()
    method = "Community"
    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterComu,avgCallComu,avgValComu,avgPowerComu = runMulti(pkgMethod.conductComuSingleOP)
            print("\n---(6) method community single {}\n{}\n-\n{}\n---".format(para,avgValComu,avgIterComu))
            nowCount = globalVar.get_value(enumVar.nowFitnessCount)
            print("call fitness: {}".format(nowCount))
            dictResult[method+para] = avgValComu#[0:2,:]
            dictPower[method+para] = avgPowerComu
        methodEnd(startTime)

    #(7)
    #listScene_rcmd = copy.deepcopy(listScene)
    #valRcmd, phenRcmd = pkgMethod.conductRcmdSingleOP()
    method = "Recommend"
    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterRcmd,avgCallRcmd,avgValRcmd,avgPowerRcmd =  runMulti(pkgMethod.conductRcmdSingleOP)
            print("\n---(7) method lyapunov + community recommmend single {}\n{}\n-\n{}\n---".format(para,avgValRcmd,avgIterRcmd))
            dictResult[method+para] = avgValRcmd#[0:2,:]
            dictPower[method+para] = avgPowerRcmd
            dictGen[method+para] = avgIterRcmd
        methodEnd(startTime)
    
    #(8)
    #pkgMethod.conductSceneMultiOP()

    #(9)
    method = "GeneticSimple"
    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterGeneSimp,avgCallGeneSimp,avgValGeneSimp,avgPowerGeneSimp =  runMulti(pkgMethod.conductGeneticSimple)
            print("\n---(9) method GeneticSimple {}\n{}\n-\n{}\n---".format(para,avgValGeneSimp,avgIterGeneSimp))
            dictResult[method+para] = avgValGeneSimp#[0:2,:]
            dictPower[method+para] = avgPowerGeneSimp
            dictGen[method+para] = avgIterGeneSimp
            dictCall[method+para] = avgCallGeneSimp
        methodEnd(startTime)

    #(10)
    method = "MemeticCommunity"
    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterMemeCmut,avgCallMemeCmut,avgValMemeCmut,avgPowerMemeCmut =  runMulti(pkgMethod.conductMemeticCommunity)
            print("\n---(10) method MemeticCommunity {}\n{}\n-\n{}\n---".format(para,avgValMemeCmut,avgIterMemeCmut))
            dictResult[method+para] = avgValMemeCmut#[0:2,:]
            dictPower[method+para] = avgPowerMemeCmut
            dictGen[method+para] = avgIterMemeCmut
            dictCall[method+para] = avgCallMemeCmut
        methodEnd(startTime)

    #(11)
    method = "GeneMain"
    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterGeneMain,avgCallGeneMain,avgValGeneMain,avgPowerGeneMain =  runMulti(pkgMethod.conductGeneMain)
            print("\n---(11) method GeneMain {}\n{}\n-\n{}\n---".format(para,avgValGeneMain,avgIterGeneMain))
            dictResult[method+para] = avgValGeneMain#[0:2,:]
            dictPower[method+para] = avgPowerGeneMain
            dictGen[method+para] = avgIterGeneMain
            dictCall[method+para] = avgCallGeneMain
        methodEnd(startTime)

    #(12)
    method = "MemeMain"

    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterMemeMain,avgCallMemeMain,avgValMemeMain,avgPowerMemeMain =  runMulti(pkgMethod.conductMemeMain)
            print("\n---(12) method MemeMain {}\n{}\n-\n{}\n---".format(para,avgValMemeMain,avgIterMemeMain))
            dictResult[method+para] = avgValMemeMain#[0:2,:]
            dictPower[method+para] = avgPowerMemeMain
            dictGen[method+para] = avgIterMemeMain
            dictCall[method+para] = avgCallMemeMain
        methodEnd(startTime)

    #(13)
    method = "MultiObj"
    listScene = globalVar.get_value(enumVar.listScene)
    listScene_mobj = copy.deepcopy(listScene)

    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        #valStack = np.zeros((3,slotNum))
        #valStack[0,:] = np.arange(1,slotNum+1)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterMobjMain,avgCallMobjMain,avgValMobjMain,avgPowerMobjMain =  runMulti(pkgMethod.conductMObjMain) #pkgMethod.runMultiObj(listScene_mobj[index],index)
            #for index in range(len(listScene_mobj)):
                #avgValMobjMain,avgEnyMobjMain,avgIterMobjMain,avgCallMobjMain =  pkgMethod.runMultiObj(listScene_mobj[index],index)
            print("\n---(13) method MultiObj para: {}\n\n{}\n-\n{}\n---".format(para,avgValMobjMain,avgIterMobjMain))
            dictResult[method+para] = avgValMobjMain#[0:2,:]
            dictPower[method+para] = avgPowerMobjMain
            dictGen[method+para] = avgIterMobjMain
            dictCall[method+para] = avgCallMobjMain
        methodEnd(startTime)

    #(14)
    method = "my_GA_single"
    listScene = globalVar.get_value(enumVar.listScene)
    listScene_mobj = copy.deepcopy(listScene)

    if(method in conductMethodList):
        startTime = methodStart()
        listEvoPara = globalVar.get_value(enumVar.listEvoPara)
        #valStack = np.zeros((3,slotNum))
        #valStack[0,:] = np.arange(1,slotNum+1)
        for para in listEvoPara:
            setNowEvoPara(para)
            avgIterMobjMain,avgCallMobjMain,avgValMobjMain,avgPowerMobjMain =  runMulti(pkgMethod.conduct_my_GA_single) #pkgMethod.runMultiObj(listScene_mobj[index],index)
            #for index in range(len(listScene_mobj)):
                #avgValMobjMain,avgEnyMobjMain,avgIterMobjMain,avgCallMobjMain =  pkgMethod.runMultiObj(listScene_mobj[index],index)
            print("\n---(14) method my_GA_single para: {}\n\n{}\n-\n{}\n---".format(para,avgValMobjMain,avgIterMobjMain))
            dictResult[method+para] = avgValMobjMain#[0:2,:]
            dictPower[method+para] = avgPowerMobjMain
            dictGen[method+para] = avgIterMobjMain
            dictCall[method+para] = avgCallMobjMain
        methodEnd(startTime)
    

    #for every method, get the performance
    
    title, table = convertTable(dictResult)
    saveCsv(title, table,folder+'_fitDealy')
    computingPvalue(title, table, folder+'_fitDealy_Pvalue')

    title, table = convertTable(dictPower)
    saveCsv(title, table,folder+'_fitPower')
    computingPvalue(title, table, folder+'_fitPower_Pvalue')

    titleGen, tableGen = convertTable(dictGen)
    saveCsv(titleGen, tableGen,folder+'_fitGen')

    titleCall, tableCall = convertTable(dictCall)
    saveCsv(titleCall, tableCall,folder+'_fitCall')

    if (len(dictResult)>0):
        util.drawMethodCompare(dictResult,folder+"_runVal","User-perceived latency")
        util.drawMethodCompare(dictPower,folder+"_runPower","Energy")
        util.drawMethodCompare(dictGen,folder+"_runGen","Number of generations")
        util.drawMethodCompare(dictCall,folder+"_runCall","Number of evalutions")

def computingPvalue(title, table, savePath):
    #table: first row is title, first col is run index.
    row,col = table.shape
    result = np.zeros((5,col-1))

    lastAlg = table[:,col-1]
    avgLast = np.mean(lastAlg)
    varLast = np.var(lastAlg)
    stdLast = np.std(lastAlg)
    result[1,col-2] = avgLast
    result[2,col-2] = varLast
    result[3,col-2] = stdLast
    #print(f'lastAlg: {lastAlg}')
    #listT = []
    #listP = []
    
    #range(1,col-1) means delete first col and last col
    for lastAlgIdx in range(1,col-1):
        compareAlg = table[:,lastAlgIdx]
        avg = np.mean(compareAlg)
        var = np.var(compareAlg)
        std = np.std(compareAlg)
        #每一列与最后一列做t-test
        ttest, pvalue = stats.ttest_ind(compareAlg,lastAlg)
        #listT.append(ttest)
        #listP.append(pvalue)
        result[0,lastAlgIdx-1] = ttest
        result[1,lastAlgIdx-1] = avg
        result[2,lastAlgIdx-1] = var
        result[3,lastAlgIdx-1] = std
        result[4,lastAlgIdx-1] = pvalue
    colTitle = title[1:] # delete index col
    #print(f'title\n{title}\nresult\n{result}')
    rowTitle = ['t-test','avg','var','std','P-value']
    saveCsvWithColRowTitle(colTitle, rowTitle, result, savePath)
    #saveCsv(title, result, savePath)
    


def computeEnergyWithPhen(config, phen):
    #tmpScene = globalVar.get_value(enumVar.currentScene)
    tmpScene = config.currentScene
    newScene = updateCurrentScene(config, tmpScene, phen)
    engy = computeEnergy(config, newScene)
    return engy

def setNowEvoPara(config, para):
    paraInt = [float(x) for x in para.split(',',6)]
    '''
    globalVar.set_value(enumVar.nowEvoPara,para)
    globalVar.set_value(enumVar.nowPopSize,int(paraInt[0]))
    globalVar.set_value(enumVar.nowMaxGen,int(paraInt[1]))
    globalVar.set_value(enumVar.nowMaxFitness,int(paraInt[2]))
    globalVar.set_value(enumVar.nowpc,paraInt[3])
    globalVar.set_value(enumVar.nowpm,paraInt[4])
    globalVar.set_value(enumVar.nowpl,paraInt[5])
    '''
    config.nowEvoPara = para
    config.nowPopSize = int(paraInt[0])
    config.nowMaxGen = int(paraInt[1])
    config.nowMaxFitness = int(paraInt[2])
    config.nowpc = int(paraInt[3])
    config.nowpm = int(paraInt[4])
    config.nowpl = int(paraInt[5])

    pop_size = int(paraInt[0])
    max_gen = int(paraInt[1])
    max_fit = int(paraInt[2])
    pc = paraInt[3]
    pm = paraInt[4]
    pl = paraInt[5]

    return pop_size, max_gen, max_fit, pc, pm, pl

def convertTable(dictResult):
    #num = len(dictResult.items())
    #print(num)
    title = ['x']
    for key in dictResult.keys():
        newkey = key.replace(',','-')
        title.append(newkey)
    #title = np.array(title)
    #print(title)

    table = []
    i = 0
    for k,v in dictResult.items():
        x = v[0]
        y = v[1]
        if(i == 0):
            table.append(x)
        table.append(y)
        i+=1
        #print(x)
        #print(y)
    #avgVal = np.vstack((xlist,meanValSingle))
    #print(table)
    table = np.array(table).T
    #table = table.T
    #print(table)
    return title, table
    
def saveCsv(title, table, fileName):
    #np.savetxt(ioGetPath()+"ResultImg"+globalVar.pathSep+name+'.csv',array,delimiter=',')
    folderPath = os.getcwd()+globalVar.pathSep+"record"+globalVar.pathSep   #"\\"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    path = folderPath+fileName+'.csv'
    df = pd.DataFrame(table)
    df.columns = title
    df.to_csv(r''+path,encoding='gbk')

def saveCsvWithColRowTitle(colTitle, rowTitle, table, fileName):
    #np.savetxt(ioGetPath()+"ResultImg"+globalVar.pathSep+name+'.csv',array,delimiter=',')
    folderPath = os.getcwd()+globalVar.pathSep+"record"+globalVar.pathSep   #"\\"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    path = folderPath+fileName+'.csv'
    df = pd.DataFrame(table)
    df.columns = colTitle
    df.index = rowTitle
    df.to_csv(r''+path,encoding='gbk')

    
def saveTimeCsv(table, fileName):
    dt = datetime.datetime.now()
    folder = globalVar.get_value(enumVar.currentFolder)
    surfix = dt.strftime("%Y-%m-%d-%H-%M-%S")
    #np.savetxt(ioGetPath()+"ResultImg"+globalVar.pathSep+name+'.csv',array,delimiter=',')
    folderPath = os.getcwd()+globalVar.pathSep#"\\"
    path = folderPath+"record"+globalVar.pathSep+folder+"_"+fileName+'.csv'
    df = pd.DataFrame(table)
    #df.columns = title
    df.to_csv(r''+path,encoding='gbk')

def save_table_to_csv(config, table, title, fileName, prefix='', surfix=''):
    #folder = globalVar.get_value(enumVar.currentFolder)
    #surfix = dt.strftime("%Y-%m-%d-%H-%M-%S")
    #np.savetxt(ioGetPath()+"ResultImg"+globalVar.pathSep+name+'.csv',array,delimiter=',')
    folderPath = os.getcwd()+config.pathSep#"\\"
    path = folderPath+"record"+config.pathSep+prefix+'_'+fileName+'_'+surfix+'.csv'
    df = pd.DataFrame(table)
    df.columns = title
    df.to_csv(r''+path, encoding='gbk', index=False)
    return path


    
if __name__ == '__main__':
    setInitGlobalVar()
    #folderList = ["init"]#,"init-100","init-1000"]
    #popSize: 100
    #maxEpochs: 100
    #maxFitness: 1500
    #globalVar.set_value(enumVar.listFolder,["init","init-100"])
    #globalVar.set_value(enumVar.listMethod,["SingleOP","Lyapnov"])
    #globalVar.set_value(enumVar.listEvoPara,[(50,100,5001),(100,100,5002),(200,100,5003)])
    
    folderList = globalVar.get_value(enumVar.listFolder)
    for folder in folderList:
        runMain(folder)
