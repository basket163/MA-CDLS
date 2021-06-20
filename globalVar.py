# -*- coding: utf-8 -*-
#globalVar
from enum import Enum
import yaml
import numpy as np
import os

class Result(object):
    """ Query result class.

    Attributes:
        is_single: a boolean indicating single objective or multi-objectives.
        is_done: a boolean indicating if all attributes is computed.
        phen: a string indicating a solution.
    """

    def __init__(self, is_single):
        """ Inits a Result class with default data. """
        self.is_single = is_single
        self.is_evo = True
        self.is_done = False

        self.best_gen_idx = 0
        self.best_gen_val = 0
        self.best_gen_phen = None

        self.best_fit_idx = 0
        self.best_fit_val = 0
        self.best_fit_phen = None

        self.best_val = 0
        self.best_phen = None

        
        self.mo_best_fit_idx = 0
        self.mo_best_fit_f1 = 0
        self.mo_best_fit_f2 = 0
        self.mo_best_fit_phen = None
        self.mo_dict_fit_better = {}
        self.mo_dict_fit_all = {}

        self.mo_best_gen_idx = 0
        self.mo_best_gen_f1 = 0
        self.mo_best_gen_f2 = 0
        self.mo_best_gen_phen = None
        self.mo_dict_gen_better = {}
        self.mo_dict_gen_all = {}

        self.mo_best_f1 = 0
        self.mo_best_f2 = 0
        self.mo_best_phen = 0
        

        # {idx: (best_val, best_phen)}
        self.dict_gen = {}
        self.dict_fit = {}
        self.dict_all_gen = {}
        self.dict_all_fit = {}

    def set_single(self, is_single):
        """ is single objective or multi-objectives. """
        self.is_single = is_single

    def add_gen_result(self, gen_idx, best_val, best_phen):
        """ if best_fit changes at a gen, update dict"""
        if(self.is_single):
            self.dict_gen[gen_idx] = (best_val, best_phen)

            self.best_gen = gen_idx
            self.best_val = best_val
            self.best_phen = best_phen
            self.best_gen_val = best_val

    def add_fit_result(self, fit_idx, best_val, best_phen):
        """ if best_fit changes at a fit, update dict"""
        if(self.is_single):
            self.dict_fit[fit_idx] = (best_val, best_phen)

            self.best_fit = fit_idx
            self.best_val = best_val
            self.best_phen = best_phen
            self.best_fit_val = best_val

    def add_all_gen_result(self, gen_idx, best_val, best_phen):
        """ add all best_fits at each gen"""
        if(self.is_single):
            self.dict_all_gen[gen_idx] = (best_val, best_phen)

    def add_all_fit_result(self, fit_idx, best_val, best_phen):
        """ add all best_fits at each fit"""
        if(self.is_single):
            self.dict_all_fit[fit_idx] = (best_val, best_phen)

    def add_mo_gen_result(self, gen_idx, f1, f2, best_phen):
        """ if best_fit changes at a gen, update dict"""
        self.mo_best_gen_idx = gen_idx
        self.mo_best_gen_f1 = f1
        self.mo_best_gen_f2 = f2
        self.mo_best_gen_phen = best_phen
        self.mo_dict_gen_better[gen_idx] = (f1, f2, best_phen)
        

    def add_mo_fit_result(self, fit_idx, f1, f2, best_phen):
        """ if best_fit changes at a fit, update dict"""
        self.mo_best_fit_idx = fit_idx
        self.mo_best_fit_f1 = f1
        self.mo_best_fit_f2 = f2
        self.mo_best_fit_phen = best_phen
        self.mo_dict_fit_better[fit_idx] = (f1, f2, best_phen)


    def add_mo_all_gen_result(self, gen_idx, f1, f2, best_phen):
        """ add all best_fits at each gen"""
        self.mo_dict_gen_all[gen_idx] = (f1, f2, best_phen)

    def add_mo_all_fit_result(self, fit_idx, f1, f2, best_phen):
        """ add all best_fits at each fit"""
        self.mo_dict_fit_all[fit_idx] = (f1, f2, best_phen)

    def get_dict_as_table(self, dict_data):
        # dict_data is {idx: (best_val, best_phen)}
        list_idx = []
        list_val = []
        for idx, (val,phen) in dict_data.items():
            list_idx.append(idx)
            list_val.append(val)
        array_idx = np.array(list_idx).reshape((-1,1))
        array_val = np.array(list_val).reshape((-1,1))
        table = np.hstack((array_idx,array_val))
        return table

    def get_dict_f1_f2_as_table(self, dict_data):
        # dict_data is {idx: (best_val, best_phen)}
        list_idx = []
        list_f1 = []
        list_f2 = []
        for idx, (f1, f2, phen) in dict_data.items():
            list_idx.append(idx)
            list_f1.append(f1)
            list_f2.append(f2)
        array_idx = np.array(list_idx).reshape((-1,1))
        array_f1 = np.array(list_f1).reshape((-1,1))
        array_f2 = np.array(list_f2).reshape((-1,1))
        table = np.hstack((array_idx,array_f1,array_f2))
        return table

class Query(object):
    def __init__(self, config, query_idx, query_count,
                    folder_idx, folder_count, folder_name, 
                    user_count, list_scene,  cell_id_min, cell_id_max,
                    scene_idx, scene_count, scene, 
                    alg_idx, alg_count, alg_name, 
                    para_idx, para_count, para_str, 
                    run_idx, run_count,
                    pop_size, max_gen, max_fit, pc, pm, pl):
        self.config = config
        self.query_idx = query_idx
        self.query_count = query_count
        self.is_single = True

        self.folder_idx = folder_idx
        self.folder_count = folder_count
        self.folder_name = folder_name

        self.user_count = user_count
        self.list_scene = list_scene
        self.cell_id_min = cell_id_min
        self.cell_id_max = cell_id_max

        self.scene_idx = scene_idx
        self.scene_count = scene_count 
        self.scene = scene

        self.alg_idx = alg_idx
        self.alg_count = alg_count
        self.alg_name = alg_name
        

        self.para_idx = para_idx
        self.para_count = para_count
        self.para_str = para_str
        
        self.run_idx = run_idx
        self.run_count = run_count

        self.pop_size = pop_size
        self.max_gen = max_gen
        self.max_fit = max_fit
        self.pc = pc
        self.pm = pm
        self.pl = pl

        self.result = Result(self.is_single)

    def add_result(self, result):
        self.result = result

    def get_file_name(self):
        file_name = f'{self.folder_name}_slot{self.scene_idx+1}_{self.alg_name}_{self.para_str}_{self.run_idx+1}'
        return file_name

    def show_param(self):
        query_info = f'query({self.query_idx+1}/{self.query_count})'
        folder_info = f'folder({self.folder_idx+1}/{self.folder_count}): {self.folder_name}'
        scene_info = f'slot({self.scene_idx+1}/{self.scene_count})'
        alg_info = f'alg({self.alg_idx+1}/{self.alg_count}): {self.alg_name}'
        para_info = f'para({self.para_idx+1}/{self.para_count}): {self.para_str}'
        run_info = f'run({self.run_idx+1}/{self.run_count})'
        para_info = f'---{query_info} {folder_info} {scene_info} {alg_info} {para_info} {run_info}'
        return para_info


    def show_result(self):
        query_info = f'query({self.query_idx+1}/{self.query_count})'
        folder_info = f'folder({self.folder_idx+1}/{self.folder_count}): {self.folder_name}'
        scene_info = f'slot({self.scene_idx+1}/{self.scene_count})'
        alg_info = f'alg({self.alg_idx+1}/{self.alg_count}): {self.alg_name}'
        para_info = f'para({self.para_idx+1}/{self.para_count}): {self.para_str}'
        run_info = f'run({self.run_idx+1}/{self.run_count})'
        if self.is_single:
            result_info = f'best_gen: {self.result.best_gen_idx}, best_fit: {self.result.best_fit_idx}, best_val: {self.result.best_val}' # \n{self.result.best_phen}
        else:
            result_info = f'best_gen: {self.result.mo_best_gen_idx}, best_fit: {self.result.mo_best_fit_idx}, best_f1: {self.result.mo_best_f1}, best_f2: {self.result.mo_best_f2}'
        return result_info

    def show_title(self):
        #para: 50,100,20000,0.4,0.6,0.1 == pop_size,max_gen,max_fit,pc,pm,pl
        #title_str = 'data_name,slot,alg,pop_size,max_gen,max_fit,pc,pm,pl,run,best_gen,best_fit,best_val'
        #title_list = ['data_name', 'slot', 'alg', 'pop_size', 
        #            'max_gen', 'max_fit', 'pc', 'pm', 'pl', 'run',
        #            'best_gen', 'best_fit', 'best_val']
        if self.is_single:
            title_str = 'data_name,slot,alg,para,run,best_gen,best_fit,best_val'
            title_list = ['data_name', 'slot', 'alg', 'para', 'run',
                        'best_gen', 'best_fit', 'best_val']
        else:
            title_str = 'data_name,slot,alg,para,run,best_gen,best_fit,best_f1,best_f2'
            title_list = ['data_name', 'slot', 'alg', 'para', 'run',
                        'best_gen', 'best_fit', 'best_f1', 'best_f2']
        return title_str, title_list

    def show_log(self):
        folder = self.folder_name
        slot = str(self.scene_idx+1)

        alg = self.alg_name
        para = self.para_str
        run = str(self.run_idx+1)
        para_new = para.replace(',', '_')

        if self.is_single:
            best_gen_idx = str(self.result.best_gen_idx)
            best_fit_idx = str(self.result.best_fit_idx)
            best_val = str(self.result.best_val)
            log_str = folder+','+slot+','+alg+','+para_new+','+run+','+best_gen_idx+','+best_fit_idx+','+best_val
            log_list = [folder, slot, alg, para_new, run, best_gen_idx, best_fit_idx, best_val]
        else:
            best_gen_idx = str(self.result.mo_best_gen_idx)
            best_fit_idx = str(self.result.mo_best_fit_idx)
            best_f1 = str(self.result.mo_best_f1)
            best_f2 = str(self.result.mo_best_f2)
            log_str = folder+','+slot+','+alg+','+para_new+','+run+','+best_gen_idx+','+best_fit_idx+','+best_f1+','+best_f2
            log_list = [folder, slot, alg, para_new, run, best_gen_idx, best_fit_idx, best_f1, best_f2]

        return log_str, log_list

class Context(object):
    def __init__(self, config, folder_name, scene_idx, query_list):
        """Query class"""
        self.config = config
        self.query_list = query_list
        self.folder_name = folder_name
        self.scene_idx = scene_idx


enumScene = Enum('enumScene',(
    '_userId', #0 
    '_uInCell', #1 
    '_userRequ', #2 
    '_inServPflNum', #3 
    '_inServCapa', #4 
    '_uInCellLast', #5 
    '_userRequLast', #6 
    '_inServPflNumLast', #7 
    '_inServCapaLast', #8 
    '_pInServLast', #9 
    '_userQueuLast', #10 
    '_userQueu', #11 
    '_pInServ' #12 (solution)
    ))

enumVar = Enum('enumVar', (
    'cellMatrix',
    'dfUser',
    'dfServ',
    'npUser',
    'npServ',
    'userCount',
    'cellIdMin',
    'cellIdMax',
    'servIdMin',
    'servIdMax',
    'currentSlot',
    'currentScene',
    'listClsUser',
    'listClsServ',
    'listClsCell',
    'listScene',
    'listFolder',
    'currentFolder',
    'listMethod',
    'listEvoPara',
    'nowEvoPara',
    'nowPopSize',
    'nowMaxGen',
    'nowMaxFitness',
    'nowFitnessCount',
    'nowpc',
    'nowpm',
    'nowpl',
    'servIdList',
    'servIdCapa',
    'reserve'
    ))


class Config(object):

    def __init__(self, cfg_file=''):
        #print("config init.")

        if cfg_file == '':
            cfg_file = 'globalCfg.yml'
            print(f'config is null, set to default value: {cfg_file}.')
            #cfg_file = r'./globalCfg.yml'
        isExists=os.path.exists(cfg_file)
        if not isExists:
            input(f'config file {cfg_file} not exists.')
        else:
            print(f'load config file {cfg_file}.')
        f = open(cfg_file)
        configFile = yaml.safe_load(f)

        self.pathSep = ''

        self.listFolder = configFile["listFolder"]
        self.listMethod = configFile["listMethod"]
        self.listEvoPara = configFile["listEvoPara"]

        self.folder_count = len(self.listFolder)
        self.alg_count = len(self.listMethod)
        self.para_count = len(self.listEvoPara)
        self.run_count = int(configFile["runNum"])
        self.runNum = int(configFile["runNum"])

        self.userCount = configFile["userCount"] # init value is 0
        self.cellIdMin = configFile["cellIdMin"]
        self.cellIdMax = configFile["cellIdMax"]
        self.servCount = configFile["servCount"]  # init value is 0
        self.servIdMin = configFile["servIdMin"]
        self.servIdMax = configFile["servIdMax"]

        self.color = configFile["color"]
        self.marker = configFile["marker"]
        
        self. _userId = int(configFile["_userId"])
        self._uInCell = int(configFile["_uInCell"])
        self._userRequ = int(configFile["_userRequ"])
        self._inServPflNum = int(configFile["_inServPflNum"])
        self._inServCapa = int(configFile["_inServCapa"])
        self._uInCellLast = int(configFile["_uInCellLast"])
        self._userRequLast = int(configFile["_userRequLast"])
        self._inServPflNumLast = int(configFile["_inServPflNumLast"])
        self._inServCapaLast = int(configFile["_inServCapaLast"])
        self._pInServLast = int(configFile["_pInServLast"])
        self._userQueuLast = int(configFile["_userQueuLast"])
        self._userQueu = int(configFile["_userQueu"])
        self._pInServ = int(configFile["_pInServ"])

        self.nashRunCount = int(configFile["nashRunCount"])
        self.nashHopIn = int(configFile["nashHopIn"])
        self.lyapHop = int(configFile["lyapHop"])

        self.showPic = configFile["showPic"]
        

        self.dataFolder = 'data'
        self.ResultData = "ResultData"
        self.log_str_sta = 'sta'
        self.log_str_detail = 'log'
        self.log_str_gen = 'gen'
        self.log_str_fit = 'fit'
        self.log_str_val = 'val'
        self.log_folder = configFile["log_folder"]
        self.log_folder_path = os.getcwd() + os.path.sep +self.log_folder + os.path.sep
        if not os.path.exists(self.log_folder_path):
            os.makedirs(self.log_folder_path)

        self.query_idx = 0  # the total query index
        self.query_count = 0  # the total query count
        self.curDataFolder = ''

        self.runTime = ''
        self.runIdx = 0

        self.init_var()

    def init_var(self):
        self.cellMatrix = None
        self.dfUser = None
        self.dfServ = None
        self.npUser = None
        self.npServ = None

        self.listScene = []
        self.scene_count = None
        self.currentSlot = None
        self.currentScene = None
        self.currentFolder = None
        self.listClsUser = None
        self.listClsServ = None
        self.listClsCell = None
        self.servIdList = None
        self.servIdCapa = None

        self.nowEvoPara = None
        self.nowPopSize = None
        self.nowMaxGen = None
        self.nowMaxFitness = None
        self.nowFitnessCount = None
        self.nowpc = None
        self.nowpm = None
        self.nowpl = None



cfg = '' #Config()

query_idx = 0  # the total query index
query_count = 0  # the total query count
curDataFolder = ''
pathSep = ''
runTime = ''
runIdx = 0

dataFolder = 'data'
ResultData = "ResultData"
log_str_sta = 'sta'
log_str_detail = 'log'
log_str_gen = 'gen'
log_str_fit = 'fit'
log_str_val = 'val'

def _init():#初始化
	global _global_dict
	_global_dict = {}

def set_value(key,value):
	""" 定义一个全局变量 """
	_global_dict[key] = value

def get_value(key,defValue=None):
	""" 获得一个全局变量,不存在则返回默认值 """
	try:
		return _global_dict[key]
	except KeyError:
		return defValue
