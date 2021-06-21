import sys
import os
import numpy as np
import pandas as pd
import copy
import itertools
import threading
from multiprocessing import Process

import globalVar
import edge
import util
import pkg_alg
from globalVar import enumVar
from globalVar import enumScene


class MyThread(threading.Thread):
    def __init__(self, threadLock, threadID, name, function, para):
        threading.Thread.__init__(self)
        self.threadLock = threadLock
        self.threadID = threadID
        self.name = name
        self.function = function
        self.para = para

    def run(self):
        # self.threadLock.acquire()
        print(f'thread {self.threadID} start')
        self.function(self.para)
        print(f'thread {self.threadID} stop')
        # self.threadLock.release()


def single_thread(switch, context):
    title_str = None
    title_list = None
    log_array = []
    for i, query in enumerate(context.query_list):
        info_para = query.show_param()
        util.sta_show_now_time_msg(msg=info_para)
        switch[query.alg_name](query)
        info_result = query.show_result()
        util.sta_show_now_time_msg(msg=info_result+'\n')
    for i, query in enumerate(context.query_list):
        # get title once in the 'for' section
        if (i == 0):
            title_str, title_list = query.show_title()

        log_str, log_list = query.show_log()
        log_array.append(log_list)
    return title_str, title_list, log_array


def multi_thread(switch, context):
    title_str = None
    title_list = None
    log_array = []

    threadLock = threading.Lock()
    threadID = 0
    threads = []

    for i, query in enumerate(context.query_list):
        threadID += 1
        t = MyThread(threadLock, threadID, query.alg_name, switch[query.alg_name], query)
        threads.append(t)
        # t.setDaemon(True)
        t.start()

    for t in threads:
        t.join()

    for i, query in enumerate(context.query_list):
        # get title once in the 'for' section
        if (i == 0):
            title_str, title_list = query.show_title()

        log_str, log_list = query.show_log()
        log_array.append(log_list)
    return title_str, title_list, log_array


def multi_process(switch, context):
    title_str = None
    title_list = None
    log_array = []

    process_id = 0
    process_list = []

    for i, query in enumerate(context.query_list):
        process_id += 1
        p = Process(target=switch[query.alg_name], args=(query,))
        p.start()

    [p.join() for p in process_list]

    for i, query in enumerate(context.query_list):
        # get title once in the 'for' section
        if (i == 0):
            title_str, title_list = query.show_title()

        log_str, log_list = query.show_log()
        log_array.append(log_list)
    return title_str, title_list, log_array


def begin_query(context):
    """ start query.

    Args:
        context: folder_name, slot_idx, query_list

    Returns:
        Query result class in globalVar.py
    """
    config = context.config
    switch = {
        'Single_PT': pkg_alg.run_Single_PT,
        'Single_GT': pkg_alg.run_Single_GT,
        'Single_GA': pkg_alg.run_single_GA,
        'Single_MA': pkg_alg.run_Single_MA,
        'Single_MA_CDLS': pkg_alg.run_Single_MA_CDLS,
        'Multi_PT': pkg_alg.run_Multi_PT,
        'Multi_GT': pkg_alg.run_Multi_GT,
        'Multi_GA': pkg_alg.run_Multi_GA,
        'Multi_MA': pkg_alg.run_Multi_MA,
        'Multi_MA_CDLS': pkg_alg.run_Multi_MA_CDLS
    }

    '''
    start_time = util.sta_get_start_time()
    title_str, title_list, log_array = multi_thread(switch, context)
    util.sta_show_used_time_msg(start_time, msg='multi_thread')
    
    start_time = util.sta_get_start_time()
    title_str, title_list, log_array = multi_process(switch, context)
    util.sta_show_used_time_msg(start_time, msg='multi_process')
    input('pause')
    '''

    start_time = util.sta_get_start_time()
    title_str, title_list, log_array = single_thread(switch, context)
    util.sta_show_used_time_msg(start_time, msg='single_thread')

    table_all = np.array(log_array)

    if 'Multi' in context.query_list[0].alg_name:
        sta_multi(context, table_all, title_list)
    else:
        sta_single(context, table_all, title_list)


def sta_single(context, table_all, title_list):
    # after all query classes finish, begin statistic.
    config = context.config
    file_name = context.folder_name + "_slot" + str(context.scene_idx + 1)
    edge.save_table_to_csv(context.config, table_all, title_list, file_name, prefix='sta', surfix='all')

    df = pd.DataFrame(table_all)
    df.columns = title_list

    # title_list = ['data_name', 'slot', 'alg', 'pop_size',
    #                'max_gen', 'max_fit', 'pc', 'pm', 'pl', 'run',
    #                'best_gen', 'best_fit', 'best_val']

    alg_list = df['alg'].unique()
    para_list = df['para'].unique()

    dict_gen = {}
    dict_fit = {}
    dict_val = {}
    combination = list(itertools.product(alg_list, para_list))
    for comb in combination:
        col_key = ''.join(comb)
        pd_sub = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]
        pd_sub_gen = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]['best_gen']
        pd_sub_fit = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]['best_fit']
        pd_sub_val = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]['best_val']

        dict_gen[col_key] = pd_sub_gen.values.astype(int)  # .tolist()  #.#T#.tolist()
        dict_fit[col_key] = pd_sub_fit.values.astype(int)  # .tolist()  #.T#.tolist()
        dict_val[col_key] = pd_sub_val.values.astype(float)  # .tolist()   #.apply(pd.to_numeric())  #.T#.tolist()

    file_path_list = []
    file_path_list.append(save_dict_to_csv(config, dict_gen, file_name, prefix='sta', surfix='gen'))
    file_path_list.append(save_dict_to_csv(config, dict_fit, file_name, prefix='sta', surfix='fit'))
    file_path_list.append(save_dict_to_csv(config, dict_val, file_name, prefix='sta', surfix='val'))
    #util.computePvalue(file_path_list)
    print('program finish.')

def sta_multi(context, table_all, title_list):
    # after all query classes finish, begin statistic.
    config = context.config

    file_name = context.folder_name + "_slot" + str(context.scene_idx + 1)
    edge.save_table_to_csv(context.config, table_all, title_list, file_name, prefix='sta', surfix='all')

    df = pd.DataFrame(table_all)
    df.columns = title_list

    # title_list = ['data_name', 'slot', 'alg', 'pop_size',
    #                'max_gen', 'max_fit', 'pc', 'pm', 'pl', 'run',
    #                'best_gen', 'best_fit', 'best_f1', 'best_f2']

    alg_list = df['alg'].unique()
    para_list = df['para'].unique()

    dict_gen = {}
    dict_fit = {}
    dict_f1 = {}
    dict_f2 = {}
    combination = list(itertools.product(alg_list, para_list))
    for comb in combination:
        col_key = ''.join(comb)
        pd_sub = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]
        pd_sub_gen = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]['best_gen']
        pd_sub_fit = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]['best_fit']
        pd_sub_f1 = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]['best_f1']
        pd_sub_f2 = df[(df[u'alg'] == comb[0]) & (df[u'para'] == comb[1])]['best_f2']

        dict_gen[col_key] = pd_sub_gen.values.astype(int)  # .tolist()  #.#T#.tolist()
        dict_fit[col_key] = pd_sub_fit.values.astype(int)  # .tolist()  #.T#.tolist()
        dict_f1[col_key] = pd_sub_f1.values.astype(float)  # .tolist()   #.apply(pd.to_numeric())  #.T#.tolist()
        dict_f2[col_key] = pd_sub_f2.values.astype(float)  # .tolist()   #.apply(pd.to_numeric())  #.T#.tolist()

    file_path_list = []
    file_path_list.append(save_dict_to_csv(config, dict_gen, file_name, prefix='sta', surfix='gen'))
    file_path_list.append(save_dict_to_csv(config, dict_fit, file_name, prefix='sta', surfix='fit'))
    file_path_list.append(save_dict_to_csv(config, dict_f1, file_name, prefix='sta', surfix='f1'))
    file_path_list.append(save_dict_to_csv(config, dict_f2, file_name, prefix='sta', surfix='f2'))
    util.computePvalue(file_path_list)
    print()

def save_dict_to_csv(config, dict_data, file_name, prefix='', surfix=''):
    table_data, title = util.convert_dict_to_table(dict_data)
    file_path = edge.save_table_to_csv(config, table_data, title, file_name, prefix, surfix)
    return file_path


def gen_query_list(config, folder_idx, folder_name, user_count, list_scene, scene_idx, scene):
    """ Generate query class with query parameters.

    Args:
        folder_idx: current data folder idx.
        folder_name: current data folder name.

    Returns:
        Query class list in globalVar.py
    """

    query_cls_list = []

    folder_list = config.listFolder
    alg_list = config.listMethod
    para_list = config.listEvoPara

    folder_count = config.folder_count
    scene_count = config.scene_count
    alg_count = config.alg_count
    para_count = config.para_count
    run_count = config.run_count

    query_count = config.query_count

    cell_id_min = config.cellIdMin
    cell_id_max = config.cellIdMax

    for alg_idx, alg_name in enumerate(alg_list):
        # globalVar.query_idx += 1
        for para_idx, para_str in enumerate(para_list):
            # globalVar.query_idx += 1
            for run_idx in np.arange(run_count):
                #globalVar.query_idx += 1
                config.query_idx += 1
                pop_size, max_gen, max_fit, pc, pm, pl = edge.setNowEvoPara(config, para_str)

                query_cls = globalVar.Query(config, config.query_idx, query_count,
                                            folder_idx, folder_count, folder_name,
                                            user_count, list_scene, cell_id_min, cell_id_max,
                                            scene_idx, scene_count, scene,
                                            alg_idx, alg_count, alg_name,
                                            para_idx, para_count, para_str,
                                            run_idx, run_count,
                                            pop_size, max_gen, max_fit, pc, pm, pl)
                query_cls_list.append(query_cls)
    return query_cls_list


def load_data(config, folder_idx, folder_name):
    """ Load 4 csv files from folder

    Args:
        folder: current data folder name.

    Returens:
        None.
    """

    folder = folder_name
    print("\n#####run data folder: {}".format(folder))
    config.curDataFolder = folder

    data_folder_name = config.dataFolder
    sep = config.pathSep

    fileUser = data_folder_name + sep + folder + sep + "user.csv"
    fileServ = data_folder_name + sep + folder + sep + "server.csv"
    fileMove = data_folder_name + sep + folder + sep + "move.csv"
    fileHop = data_folder_name + sep + folder + sep + "hopMatrix.xlsx"

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
    config.dfUser = dfUser
    config.dfServ = dfServ
    config.npUser = npUser
    config.npServ = npServ
    config.cellMatrix = matrix

    listClsUser = edge.loadUser(config, dfUser)
    listClsServ = edge.loadServ(config, dfServ)
    #globalVar.set_value(enumVar.listClsUser, listClsUser)
    #globalVar.set_value(enumVar.listClsServ, listClsServ)
    config.listClsUser = listClsUser
    config.listClsServ = listClsServ

    # important operation!
    userCount = len(listClsUser)
    config.userCount = userCount
    #globalVar.set_value(enumVar.userCount, userCount)

    # print("userCount is {}".format(userCount))
    # input()
    if userCount == 0:
        print("input error, userCount is 0")
        input()

    servCount = len(listClsServ)
    config.servCount = servCount
    if servCount == 0:
        print("input error, servCount is 0")
        input()

    listSlot = edge.loadSlot(dfMove)
    # conductSingleObjective(listSlot)

    #listScene = globalVar.get_value(enumVar.listScene)
    #listScene = config.listScene
    config.listScene.clear()

    initialScene = edge.initialSceneFunction(config, listSlot[0])
    # print("initialScene:\n{}".format(initialScene))
    edge.recursiveScene(config, listSlot, 0, initialScene)  # return listScene

    return userCount, config.listScene


def run_data_folder(config, folder_idx, folder_name):
    """  Run each data folder.

    Args:
        folder_idx: current data folder idx.
        folder_name: current data folder name.

    Returns:
        None. Output files in result folder.
    """

    user_count, list_scene_ori = load_data(config, folder_idx, folder_name)
    # list_scene_ori = globalVar.get_value(enumVar.listScene)
    list_scene = copy.deepcopy(list_scene_ori)
    # user_count = globalVar.get_value(enumVar.userCount)
    # index = 0
    scene_count = len(list_scene)
    config.list_scene = list_scene
    config.scene_count = scene_count

    folder_count = config.folder_count
    alg_count = config.alg_count
    para_count = config.para_count
    run_count = config.run_count

    # when get list_scene, compute the total query count.
    config.query_count = folder_count * scene_count * alg_count * para_count * run_count

    valStack = np.zeros((5, scene_count))
    valStack[0, :] = np.arange(1, scene_count + 1)
    phenStack = np.arange(1, user_count + 1, 1)
    for scene_idx, scene in enumerate(list_scene):
        #globalVar.set_value(enumVar.currentScene, scene)
        config.current_scene = scene
        config.currentScene = scene

        query_list = gen_query_list(config, folder_idx, folder_name, user_count, list_scene, scene_idx, scene)

        context = globalVar.Context(config, folder_name, scene_idx, query_list)
        info = f'{folder_name}_{scene_idx}'
        start = util.sta_show_now_time_msg(msg='begin a query '+info)
        begin_query(context)
        util.sta_show_used_time_msg(start, msg='')


def init_global_var(cfg_file):
    """ Initialize global variables.
    :param
        cfg_file: config file path.
    :return
        Config class.
    """

    config = globalVar.Config(cfg_file)
    globalVar.cfg = config

    # record program start time to globalVar.runTime
    run_time = util.setRunTime()
    config.runTime = run_time

    # corss platform: windows "\\", linux "/"
    config.pathSep = os.path.sep

    sys.setrecursionlimit(10000)

    config.query_idx = -1

    # remain old code
    use_old_code = 0
    if use_old_code == 1:
        globalVar._init()

        globalVar.set_value(enumVar.listFolder, globalVar.cfg.listFolder)
        globalVar.set_value(enumVar.listMethod, globalVar.cfg.listMethod)
        globalVar.set_value(enumVar.listEvoPara, globalVar.cfg.listEvoPara)

        globalVar.set_value(enumVar.cellMatrix, "")
        globalVar.set_value(enumVar.currentSlot, "")
        globalVar.set_value(enumVar.listScene, [])

        globalVar.set_value(enumVar.userCount, 0)
        globalVar.set_value(enumVar.cellIdMin, 1)
        globalVar.set_value(enumVar.cellIdMax, 18)
        globalVar.set_value(enumVar.servIdMin, 1)
        globalVar.set_value(enumVar.servIdMax, 18)

        globalVar.set_value(enumVar.nowFitnessCount, 0)

        # scene
        globalVar.set_value(enumScene._userId, int(0))
        globalVar.set_value(enumScene._uInCell, int(1))
        globalVar.set_value(enumScene._userRequ, int(2))
        globalVar.set_value(enumScene._inServPflNum, int(3))
        globalVar.set_value(enumScene._inServCapa, int(4))
        globalVar.set_value(enumScene._uInCellLast, int(5))
        globalVar.set_value(enumScene._userRequLast, int(6))
        globalVar.set_value(enumScene._inServPflNumLast, int(7))
        globalVar.set_value(enumScene._inServCapaLast, int(8))
        globalVar.set_value(enumScene._pInServLast, int(9))
        globalVar.set_value(enumScene._userQueuLast, int(10))
        globalVar.set_value(enumScene._userQueu, int(11))
        globalVar.set_value(enumScene._pInServ, int(12))

    return config


def main(cfg_file=''):
    """ program entry. """

    #cfg_file = 'globalCfg.yml'
    config = init_global_var(cfg_file)

    # run each data folder
    folder_list = config.listFolder
    for folder_idx, folder_name in enumerate(folder_list):
        # globalVar.query_idx += 1
        run_data_folder(config, folder_idx, folder_name)


if (__name__ == "__main__"):
    # command: python main.py global_cfg_single_obj.yml
    len_argv = len(sys.argv)
    cfg_file_path = ''
    if len_argv == 2:
        cfg_file_path = sys.argv[1]
    main(cfg_file = cfg_file_path)
