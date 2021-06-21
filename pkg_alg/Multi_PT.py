import numpy as np

import pkg_alg
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge

def run_Multi_PT(query):
    query.is_single = False
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

    query.result.mo_best_phen = best_phen
    query.result.mo_best_f1 = best_val
    query.result.mo_best_f2 = engy
    query.result.mo_best_fit_idx = 1
    query.result.mo_best_gen_idx = 1
    return query