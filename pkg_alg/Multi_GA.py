import numpy as np
import itertools as it
import copy

import pkg_alg
import globalVar
from globalVar import enumVar
from globalVar import enumScene
import edge
import util

class Individual(object):
    def __init__(self, evo_cls, phenotypes):
        self.phenotypes = phenotypes  # phenotype
        self.fitness = None
        self.fitness = compute_fitness(evo_cls, self.phenotypes)  # value of the fitness function
        evo_cls.indexCall += 1


    def __str__(self):
        return '{0} = {1}'.format(self.phenotypes, self.fitness)

class GA_Multi(object):
    """ This is evo_cls. """

    def __init__(self, query):
        self.query = query
        #self.is_single = True
        
        self.result = query.result

        self.scene = query.scene
        self.user_count = query.user_count
        self.pop_size = query.pop_size
        self.max_gen = query.max_gen
        self.max_fit = query.max_fit
        self.pc = query.pc
        self.pm = query.pm
        self.pl = query.pl
        self.cell_id_min =  query.cell_id_min
        self.cell_id_max = query.cell_id_max
        #print(f'{scene}')
        #print(f'{userCount} {popSize} {maxGen} {maxFitness} {p_c} {p_m} {p_l} {lower} {upper}')
        #input()

        self.population = None
        self.fit_array = None
        self.parent = None
        self.offspring = None

        self.parent_f1_array = None
        self.parent_f2_array = None
        self.offspring_f1_array = None
        self.offspring_f2_array = None
        self.pareto = util.Pareto()

        #return values
        self.cur_gen_idx = 0
        self.cur_fit_idx = 0

        '''
        self.best_gen_idx = 0
        self.best_gen_val = 0
        self.best_gen_phen = None
        self.best_fit_idx = 0
        self.best_fit_val = 0
        self.best_fit_phen = None
        '''

    def generate_population(self, row, col):
        low = self.cell_id_min
        high = self.cell_id_max+1
        rand_vals = np.random.randint(low, high, (row, col) )

        #self.population = [Individual(self, p) for p in rand_vals]
        #self.fit_col = np.array([x.fitness for x in self.population]).reshape(-1,1)
        #self.fit_col = np.array([ compute_fitness(self, p) for p in rand_vals ]).reshape(-1,1)
        self.parent_f1_array, self.parent_f2_array = self.compute_fit_list(rand_vals)

        self.parent = copy.deepcopy(rand_vals)
        self.offspring = copy.deepcopy(rand_vals)

        #print([x.phenotypes for x in self.population])
        #print([x.fitness for x in self.population])

    def compute_fit_list(self, phens):
        ret = []
        ret_energy = []
        for p in phens:
            self.cur_fit_idx += 1
            fit_val, fit_energy = compute_fitness(self, p)
            ret.append(fit_val)
            ret_energy.append(fit_energy)

            if(self.cur_fit_idx == 1):
                self.result.mo_best_fit_idx = self.cur_fit_idx
                self.result.mo_best_fit_f1 = fit_val
                self.result.mo_best_fit_f2 = fit_energy
                self.result.mo_best_fit_phen = p
                self.result.add_mo_fit_result(self.result.mo_best_fit_idx, self.result.mo_best_fit_f1, self.result.mo_best_fit_f2,self.result.mo_best_fit_phen)
            else:
                if fit_val <= self.result.mo_best_fit_f1 and fit_energy <= self.result.mo_best_fit_f2:
                    if fit_val != self.result.mo_best_fit_f1 and fit_energy != self.result.mo_best_fit_f2:
                        self.result.mo_best_fit_idx = self.cur_fit_idx
                        self.result.mo_best_fit_f1 = fit_val
                        self.result.mo_best_fit_f2 = fit_energy
                        self.result.mo_best_fit_phen = p
                        self.result.add_mo_fit_result(self.result.mo_best_fit_idx, self.result.mo_best_fit_f1, self.result.mo_best_fit_f2, self.result.mo_best_fit_phen)

            self.result.add_mo_all_fit_result(self.cur_fit_idx, self.result.mo_best_fit_f1, self.result.mo_best_fit_f2, self.result.mo_best_fit_phen)
        ret_array = np.array(ret)
        ret_energy = np.array(ret_energy)
        return ret_array, ret_energy

    def crossover(self):
        if (self.pop_size & 1) != 0:
            print('in corssover, population number is not even.')
            input()
        half_pop_size = int(self.pop_size / 2)
        half_cross_num = int( half_pop_size * self.pc)
        odd_idx_list = np.arange(0, self.pop_size, 2)
        cross_chrom_idx = np.random.choice(odd_idx_list, size=half_cross_num, replace=False)
        for x in cross_chrom_idx:
            #print(f'{x}, {x+1}')
            r = np.random.random(1)
            if(r<self.pc):
                #print(f'{r}<{self.para.evoPc}')
                crossPoint = np.random.randint(0,self.user_count)
                #print(f'crossPoint {crossPoint}/{self.popCol}')
                lastPoint = self.user_count
                c1 = copy.deepcopy(self.parent[x,:])
                c2 = copy.deepcopy(self.parent[x+1,:])
                #print(f'c1\n{c1}')
                #print(f'c2\n{c2}')
                #print(f'crossPoint\n{crossPoint}/{len(c1)}')
                #print(c1[0:crossPoint])
                #print(c2[crossPoint:self.popCol+1])
                c3 = np.hstack((c1[0:crossPoint], c2[crossPoint:lastPoint] ))
                c4 = np.hstack((c2[0:crossPoint], c1[crossPoint:lastPoint] ))
                #print(f'c3\n{c3}')
                #print(f'c4\n{c4}')
                self.offspring[x,:] = c3
                self.offspring[x+1,:] = c4
        '''
        print(self.parent)
        print()
        print(self.offspring)
        input()
        '''
        return self.offspring

    def mutation(self):
        idx_list = np.arange(self.user_count)
        muta_num = int(self.user_count * self.pm)
        for x in np.arange(self.pop_size):
            # mutation_point = np.random.randint(0,self.user_count)
            muta_in_chrom_idx = np.random.choice(idx_list, size=muta_num, replace=False)
            for y in muta_in_chrom_idx:
                mutation_context = np.random.randint(self.cell_id_min, self.cell_id_max + 1)
                # print(f'mutationPoint row:{x} col:{mutation_point} -> {mutation_context}')
                self.offspring[x, y] = mutation_context
        return self.offspring

    def outputcsv(self, col_1, col_2, prefix='log', surfix='f1_f2'):
        table = np.hstack((col_1,col_2))
        fileName = self.query.get_file_name()
        title = ['f1', 'f2']
        edge.save_table_to_csv(self.query.config, table, title, fileName, prefix='log', surfix='f1_f2')
        print('output csv ok')

    def point_distance_line(self, point, line_point1, line_point2):
        #compute the distance from a point to a line
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
        return distance

    def select_extreme_points(self, f1_array, f2_array):
        far_f1_idx = np.argmax(f1_array)
        far_f2_idx = np.argmax(f2_array)
        new_point_flag = False
        if far_f1_idx == far_f2_idx:
            #print(f'\nextreme points are the same!')
            #combine = np.hstack((np.arange(len(f1_array)).reshape(-1,1),f1_array.reshape(-1,1),f2_array.reshape(-1,1)))
            #print(f'the same index: {far_f1_idx}\n{combine}')
            tmp_f1 = copy.deepcopy(f1_array)
            tmp_f2 = copy.deepcopy(f2_array)
            allow = 5
            while(allow):
                allow = allow -1
                if far_f1_idx != far_f2_idx:
                    break
                if allow == 1:
                    new_point_flag = True
                    break
                
                tmp_f1[far_f1_idx] = min(tmp_f1) - 1
                tmp_f2[far_f1_idx] = min(tmp_f2) - 1
                far_f1_idx = np.argmax(tmp_f1)
                far_f2_idx = np.argmax(tmp_f2)
                
                #print(f'new {far_f1_idx} {far_f2_idx} ')
            #input()
        point1 = np.array([f1_array[far_f1_idx], f2_array[far_f1_idx]])
        point2 = np.array([f1_array[far_f2_idx], f2_array[far_f2_idx]])
        if new_point_flag:
            sigle_point = (f1_array[far_f1_idx], f2_array[far_f2_idx])
            zero_point = (0, 0) 
        return point1, point2

    def rank_distance(self, idx_array, f1_array, f2_array):
        distance_list = []
        point1, point2 = self.select_extreme_points(f1_array, f2_array)
        for i in idx_array:
            point = np.array([f1_array[i], f2_array[i]])
            distance_list.append(self.point_distance_line(point, point1, point2))
        index_score = np.argsort(-np.array(distance_list))
        rank_score = np.argsort(index_score) #+1
        distance_arr = np.array(distance_list)
        return distance_arr, rank_score

    def selection_new(self):
        self.offspring_f1_array, self.offspring_f2_array = self.compute_fit_list(self.offspring)
        self.pareto.add_pop(self.offspring_f1_array, self.offspring_f2_array, self.offspring)
        self.pareto.add_pop(self.parent_f1_array, self.parent_f2_array, self.parent)
        f1KP, f2KP, phenKP = self.pareto.get_knee_point()
        return self.cur_gen_idx, f1KP, f2KP, phenKP

    def selection(self):
        # save first phen and select other phens by rank probobility
        self.offspring_f1_array, self.offspring_f2_array = self.compute_fit_list(self.offspring)

        combine_pop = np.vstack((self.parent, self.offspring))
        combine_idx = np.arange(2*self.pop_size)
        #combine_f1, combine_f2 = self.compute_fit_list(combine_pop)
        combine_f1_arr = np.hstack((self.parent_f1_array, self.offspring_f1_array))
        combine_f2_arr = np.hstack((self.parent_f2_array, self.offspring_f2_array))
        combine_f1_col = np.vstack((self.parent_f1_array.reshape(-1,1), self.offspring_f1_array.reshape(-1,1)))
        combine_f2_col = np.vstack((self.parent_f2_array.reshape(-1,1), self.offspring_f2_array.reshape(-1,1)))

        idx_array = combine_idx
        distance_arr, rank_score = self.rank_distance(idx_array, combine_f1_arr, combine_f2_arr)

        
        '''
        distance_arr_col = distance_arr.reshape(-1,1)
        rank_score_col = rank_score.reshape(-1,1)
        table = np.hstack((combine_f1_col,combine_f2_col,distance_arr_col,rank_score_col))
        fileName = self.query.get_file_name()
        title = ['f1', 'f2', 'distance', 'rank']
        edge.save_table_to_csv(self.query.config, table, title, fileName, prefix='log', surfix='distance')
        print('output csv distance ok')
        #input()
        '''
        
        # select head pop_size chromosomes
        first_f1 = 0
        first_f2 = 0
        first_phen = None
        for r in np.arange(0, self.pop_size, 1):
            idx = np.where(rank_score == r)[0][0] # + 1
            #print(f'{r} {idx}')
            #input('pause')
            self.parent[r] = combine_pop[idx]
            self.parent_f1_array[r] = combine_f1_arr[idx]
            self.parent_f2_array[r] = combine_f2_arr[idx]
            if r == 0:
                first_f1 = combine_f1_arr[idx]
                first_f2 = combine_f2_arr[idx]
                first_phen = combine_pop[idx] 

        #self.parent = copy.deepcopy(rand_vals)
        self.offspring = copy.deepcopy(self.parent)
        self.offspring_f1_array = copy.deepcopy(self.parent_f1_array)
        self.offspring_f2_array = copy.deepcopy(self.parent_f2_array)

        '''
        table = np.hstack((self.parent_f1_array.reshape(-1,1), self.parent_f2_array.reshape(-1,1)))
        fileName = self.query.get_file_name()
        title = ['f1', 'f2']
        edge.save_table_to_csv(self.query.config, table, title, fileName, prefix='log', surfix='new_pop')
        input()
        '''


        '''
        # 取消科学计数法显示
        np.set_printoptions(suppress=True)
        v = np.hstack((
            combine_idx.reshape(-1,1),
            np.array(combine_fit).reshape(-1,1),
            rank_score.reshape(-1,1),
            rank_probability.reshape(-1,1)
            ))
        print(v)
        print(selection)
        '''
        return self.cur_gen_idx, first_f1, first_f2, first_phen



        


    def run(self):
        #reset
        self.cur_gen_idx = 0
        self.cur_fit_idx = 0

        self.generate_population(self.pop_size, self.user_count)

        first_f1 = -1
        first_f2 = -2
        while True:
            self.cur_gen_idx += 1

            if (self.cur_gen_idx > self.max_gen):
                #print("exceed max_gen: {0}/{1} ".format(self.cur_gen_idx, self.max_gen))
                break

            if self.cur_fit_idx > self.max_fit:
                #print("exceed max_fit: {0}/{1} ".format(self.cur_fit_idx, self.max_fit))
                break

            self.crossover()
            self.mutation()
            gen_idx, first_f1, first_f2, first_phen = self.selection_new()
            

            if(self.cur_gen_idx == 1):
                self.result.mo_best_gen_idx = gen_idx
                self.result.mo_best_gen_f1 = first_f1
                self.result.mo_best_gen_f2 = first_f2
                self.result.mo_best_gen_phen = first_phen
                self.result.add_mo_gen_result(self.result.mo_best_gen_idx, 
                                            self.result.mo_best_gen_f1,
                                            self.result.mo_best_gen_f2,
                                            self.result.mo_best_gen_phen)
            else:
                if first_f1 <= self.result.mo_best_gen_f1 and first_f2 <= self.result.mo_best_gen_f2:
                    if first_f1 != self.result.mo_best_gen_f1 and first_f2 != self.result.mo_best_gen_f2:
                        self.result.mo_best_gen_idx = gen_idx
                        self.result.mo_best_gen_f1 = first_f1
                        self.result.mo_best_gen_f2 = first_f2
                        self.result.mo_best_gen_phen = first_phen
                        self.result.add_mo_gen_result(self.result.mo_best_gen_idx, 
                                                    self.result.mo_best_gen_f1,
                                                    self.result.mo_best_gen_f2,
                                                    self.result.mo_best_gen_phen)
            self.result.add_mo_all_gen_result(gen_idx, 
                                            self.result.mo_best_gen_f1,
                                            self.result.mo_best_gen_f2,
                                            self.result.mo_best_gen_phen)

        # run finish
        self.result.mo_best_f1 = self.result.mo_best_gen_f1
        self.result.mo_best_f2 = self.result.mo_best_gen_f2
        self.result.mo_best_phen = self.result.mo_best_gen_phen
        return self.result

    def memetic(self):
        #self.population = [Individual(p,self) for p in rand_vals]
        #self.best = sorted(self.population, key=lambda x: x.fitness)[0]

        scene = globalVar.get_value(enumVar.currentScene)
        popSize = globalVar.get_value(enumVar.nowPopSize)
        userCount = globalVar.get_value(enumVar.userCount)

        for i in self.population:
            arg_vec = i.phenotypes

            tmpScene = copy.deepcopy(scene)

            newScene = edge.updateCurrentScene(tmpScene,arg_vec)
            cmni,cmpt,miga = edge.computeScene(newScene)
            #fitVal = cmni,cmpt,miga
            fitVal = edge.statisticLatency(cmni,cmpt,miga)

            new_vec = copy.deepcopy(arg_vec)

            #generate local search space
            dictUserServ = generateCommunity(newScene)

            #listCommunity = []
            for k in dictUserServ.keys():
                for v in dictUserServ[k]:
                    tmp_vec = copy.deepcopy(arg_vec)
                    tmpScene = copy.deepcopy(scene)
                    tmp_vec[k-1] = v
                    comScene = edge.updateCurrentScene(tmpScene,tmp_vec)
                    #listCommunity.append(comScene)
                    cmni,cmpt,miga = edge.computeScene(comScene)
                    comVal = edge.statisticLatency(cmni,cmpt,miga)
                    if comVal < fitVal:
                        fitVal = comVal
                        new_vec = copy.deepcopy(tmp_vec)

            i.phenotypes = new_vec
            i.fitness = fitVal
            #return new_vec, fitVal





def generateCommunity(scene):
    dictUserServ = {}
    userCount = globalVar.get_value(enumVar.userCount)
    user = scene[edge.cfg._userId,:]
    inCell = scene[edge.cfg._uInCell,:]

    for idx in np.arange(userCount):
        x = inCell[idx]
        #find user xi current serv
        candiList = edge.getServListInHop(1, x)
        dictUserServ[idx+1] = candiList
    return dictUserServ

def compute_fitness(evo_cls, arg_vec):
    config = evo_cls.query.config
    tmpScene = copy.deepcopy(evo_cls.scene)
    #evo_cls.cur_fit_idx += 1
    

    newScene = edge.updateCurrentScene(config, tmpScene,arg_vec)
    cmni,cmpt,miga = edge.computeScene(config, newScene)
    fitVal = edge.statisticLatency(cmni,cmpt,miga)

    energy = edge.computeEnergyWithPhen(config, arg_vec)

    #evo_cls.result.add_all_fit_result(evo_cls.cur_fit_idx, fitVal, arg_vec)

    return fitVal, energy

    

def run_Multi_GA(query):
    #print("\n-----Running Gene algorithm ...")
    query.is_single = False
    ga = GA_Multi(query)
    config = query.config

    # result = globalVar.Result()
    ga.run()

    '''
    print('change:')
    for x,y in ga.result.dict_gen.items():
        print(f"{x}: {y}")
    print('all:')
    for x,y in ga.result.dict_all_gen.items():
        print(f"{x}: {y}")
    
    print('change:')
    for x,y in ga.result.dict_fit.items():
        print(f"{x}: {y}")
    print('all:')
    for x,y in ga.result.dict_all_fit.items():
        print(f"{x}: {y}")

    print(f'evo \nbest_gen_idx: {ga.best_gen_idx}, best_gen_val: {ga.best_gen_val}')
    print(f'best_fit_idx: {ga.best_fit_idx}, best_fit_val: {ga.best_fit_val}')
    '''

    #print(f'evo result: best_gen_idx {ga.result.best_gen_idx}, best_fit {ga.result.best_fit_idx}, best_val {ga.result.best_val}')
    #print(f'best_phen: {ga.result.best_phen}\n')


    #bestEngy = edge.computeEnergyWithPhen(config, ga.result.best_phen)
    #arrayGen = np.array(ga.listGen).reshape((-1,1))
    #arrayGenBest = np.array(ga.listGenBest).reshape((-1,1))
    #table = np.hstack((arrayGen,arrayGenBest))

    table = ga.result.get_dict_f1_f2_as_table(ga.result.mo_dict_gen_all)
    fileName = ga.query.get_file_name()
    title = ['gen_idx', 'f1', 'f2']
    edge.save_table_to_csv(config, table, title, fileName,
                            prefix = globalVar.log_str_detail, 
                            surfix = 'gen_all') #globalVar.log_str_gen

    table = ga.result.get_dict_f1_f2_as_table(ga.result.mo_dict_gen_better)
    fileName = ga.query.get_file_name()
    title = ['gen_idx', 'f1', 'f2']
    edge.save_table_to_csv(config, table, title, fileName,
                            prefix = globalVar.log_str_detail, 
                            surfix = 'gen') #globalVar.log_str_gen

    table2 = ga.result.get_dict_f1_f2_as_table(ga.result.mo_dict_fit_all)
    fileName2 = ga.query.get_file_name()
    title2 = ['fit_idx', 'f1', 'f2']
    edge.save_table_to_csv(config, table2, title2, fileName2, prefix='log', surfix='fit_all')

    table2 = ga.result.get_dict_f1_f2_as_table(ga.result.mo_dict_fit_better)
    fileName2 = ga.query.get_file_name()
    title2 = ['fit_idx', 'f1', 'f2']
    edge.save_table_to_csv(config, table2, title2, fileName2, prefix='log', surfix='fit')

    pareto_f1, pareto_f2, phen_pareto = ga.pareto.get_pareto_front_distinct_list()
    array_f1 = np.array(pareto_f1).reshape((-1, 1))
    array_f2 = np.array(pareto_f2).reshape((-1, 1))
    table_pareto = np.hstack((array_f1, array_f2))
    title_pareto = ['latency', 'energy']
    edge.save_table_to_csv(config, table_pareto, title_pareto, fileName2, prefix='log', surfix='pareto')

    return query

def conduct_GA_single(query):
    listScene_ori = globalVar.get_value(enumVar.listScene)
    listScene = copy.deepcopy(listScene_ori)
    userCount = globalVar.get_value(enumVar.userCount)
    #index = 0
    slotNum = len(listScene)
    lastIdx = slotNum-1

    valStack = np.zeros((5,slotNum))
    valStack[0,:] = np.arange(1,slotNum+1)
    phenStack = np.arange(1,userCount+1,1)
    for index in range(slotNum):
        nowPara = globalVar.get_value(enumVar.nowEvoPara)
        print("para: {}, slot index: {}".format(nowPara,index+1))

        scene = listScene[index]
        globalVar.set_value(enumVar.currentScene,scene)

        best_gen, best_call, best_ObjV, best_engy, best_Phen = runMemeMain(scene,query)

        best_Phen = np.array(best_Phen)
        if(len(best_Phen) == 0):
            print("gen {}, val {}, phen {}".format(best_gen,best_ObjV,best_Phen))
            input()
        
        ### important update
        scene = edge.updateCurrentScene(scene,best_Phen)

        #valOP = [best_ObjV, best_gen]
        
        valStack[1,index] = best_gen
        valStack[2,index] = best_call
        valStack[3,index] = best_ObjV
        valStack[4,index] = best_engy
        phenStack = np.vstack((phenStack,best_Phen))

        #index+=1
        if index < lastIdx:
            nextScene = edge.updateNextScene(listScene,index)
            listScene[index+1] = nextScene
    return valStack,phenStack