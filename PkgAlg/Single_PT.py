import numpy as np

import PkgAlg
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

def run_Single_PT(query):
    config = query.config
    scene = query.scene
    #globalVar.set_value(enumVar.currentScene,scene)
    config.currentScene = scene

    _pInServ = config._pInServ #globalVar.get_value(enumScene._pInServ)
    best_phen = scene[_pInServ,:]
    ### important update
    scene = edge.updateCurrentScene(config, scene,best_phen)

    cmni,cmpt,miga = edge.computeScene(config, scene)
    best_val = edge.statisticLatency(cmni,cmpt,miga)
    engy = edge.computeEnergy(config, scene)
    
    query.result.best_val = best_val
    query.result.best_phen = best_phen
    query.result.is_evo = False
    return query