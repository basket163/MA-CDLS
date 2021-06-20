import numpy as np
import itertools as it
import copy
import heapq

import PkgAlg
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

class Single_MA_CDLS(object):
    """ This is evo_cls. """

    def __init__(self, query):
        self.query = query
        self.is_single = True
        
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
        self.fit_list = None
        self.fit_array_col = None
        self.fit_offspring = None
        self.parent = None
        self.offspring = None
        self.fit_idx_col = np.arange(self.pop_size).reshape(-1,1)

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
        self.fit_list, self.fit_array_col = self.compute_fit_list(rand_vals)

        self.parent = copy.deepcopy(rand_vals)
        self.offspring = copy.deepcopy(rand_vals)

        #print([x.phenotypes for x in self.population])
        #print([x.fitness for x in self.population])

    def compute_fit_list(self, phens):
        fit_list = []
        for p in phens:
            self.cur_fit_idx += 1
            fit_val = compute_fitness(self, p)
            fit_list.append(fit_val)

            if(self.cur_fit_idx == 1):
                self.result.best_fit_idx = self.cur_fit_idx
                self.result.best_fit_val = fit_val
                self.result.best_fit_phen = p
                self.result.add_fit_result(self.result.best_fit_idx, self.result.best_fit_val, self.result.best_fit_phen)

            if(fit_val < self.result.best_fit_val):
                self.result.best_fit_idx = self.cur_fit_idx
                self.result.best_fit_val = fit_val
                self.result.best_fit_phen = p
                self.result.add_fit_result(self.result.best_fit_idx, self.result.best_fit_val, self.result.best_fit_phen)

            self.result.add_all_fit_result(self.cur_fit_idx, self.result.best_fit_val, self.result.best_fit_phen)
        ret_array = copy.deepcopy(np.array(fit_list))
        ret_array_col = ret_array.reshape(-1,1)
        return fit_list, ret_array_col

    def crossover(self):
        if (self.pop_size & 1) != 0:
            print('in corssover, population number is not even.')
            input()
        half_pop_size = int(self.pop_size / 2)
        half_cross_num = int( half_pop_size * self.pc)
        odd_idx_list = np.arange(0, self.pop_size, 2)
        cross_chrom_idx = np.random.choice(odd_idx_list, size=half_cross_num, replace=False)
        for x in cross_chrom_idx:
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
            #mutation_point = np.random.randint(0,self.user_count)
            muta_in_chrom_idx = np.random.choice(idx_list, size=muta_num, replace=False)
            for y in muta_in_chrom_idx:
                mutation_context = np.random.randint(self.cell_id_min,self.cell_id_max+1)
                #print(f'mutationPoint row:{x} col:{mutation_point} -> {mutation_context}')
                self.offspring[x,y] = mutation_context
        return self.offspring


    def select_result(self):
        # save first phen and select other phens by rank probobility
        combine_pop = np.vstack((self.parent, self.offspring))
        combine_idx = np.arange(2*self.pop_size)
        #combine_fit = [ compute_fitness(self, p) for p in combine_pop ]
        offspring_fit_list, offspring_fit_array_col = self.compute_fit_list(self.offspring)
        combine_fit_list = self.fit_list + offspring_fit_list
        combine_fit_col = np.vstack((self.fit_array_col, offspring_fit_array_col))
        
        #better_pop_idx, top pop_size indexes
        #selection = combine_fit_array.argsort()[-self.pop_size:][::-1] # bigger, better
        #max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
        #min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
        selection=list(map(combine_fit_list.index, heapq.nsmallest(self.pop_size,combine_fit_list)))
        #print("combine_fit_col")
        #print(np.hstack((combine_idx.reshape(-1,1), combine_fit_col)))
        #print("selection")
        #print(selection)
        #input()


        #cc = np.hstack((combine_idx.reshape(-1,1), np.array(combine_fit).reshape(-1,1),combine_pop))
        #print('cc')
        #print(cc)

        #first_idx = np.argmin(combine_fit)
        first_idx = selection[0]
        first_fit = combine_fit_list[first_idx]
        first_phen = combine_pop[first_idx]
        #print('--')
        #print(first_idx)

        # sort
        #index = np.argsort(-np.array(combine_fit)) #desc
        #rank = np.argsort(index)+1
        #
        #index_score = np.argsort(-np.array(combine_fit))
        #rank_score = np.argsort(index_score)+1
        
        #combine_size = 2 * self.pop_size
        #all_score = self.pop_size*(combine_size+1)
        #rank_probability = np.true_divide(rank_score, all_score)
        '''
        if sum(rank_probability) != 1:
            print(f'probability do not sum to 1, is {sum(rank_probability)}: {rank_probability}')
            len_rank = len(rank_probability)
            rank_probability[len_rank-1] = 1 - sum(rank_probability[0:len_rank-2])
            print(f'new prob sum  is {sum(rank_probability)}: {rank_probability}')
        # rank_probability must be sum to 1
        '''

        #按排序后的概率选择pop_size个染色体
        #selection = np.random.choice(combine_idx, size=self.pop_size, replace=False, p=rank_probability)

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

        # remain elitism
        #if first_idx not in selection:
        #    selection[0] = first_idx
        #print(selection)
        #print(combine_fit_list)

        for i, s in enumerate(selection):
            self.parent[i] = combine_pop[s]
            self.fit_list[i] = combine_fit_list[s]
            self.fit_array_col[i,0] = combine_fit_list[s]

        # replace the first chrom to elitism
        #没有用概率选择，取消保留第一名的染色体
        #self.parent[0] = first_phen
        #self.fit_array[0] = first_fit

        self.offspring = copy.deepcopy(self.parent)
        self.fit_offspring = copy.deepcopy(self.fit_list)

        return self.cur_gen_idx, first_fit, first_phen



    def top_rank_idx(self, array, top_prob):
        #select the number of chromosomes with top rank of self.pl
        #print(array)
        meme_num = int(self.pop_size * self.pl)
        top_rank_list = np.arange(meme_num)
        # rank
        index_score = np.argsort(-np.array(array))
        rank_score = np.argsort(index_score)
        #idx = np.arange(self.pop_size)
        #demo = np.hstack((idx.reshape(-1,1), array.reshape(-1,1), rank_score.reshape(-1,1)))
        #print(demo)
        #print(top_rank_list)
        top_idx = []
        for i in top_rank_list:
            temp = np.where(rank_score==top_rank_list[i])[0][0]
            top_idx.append(temp)
        top_idx = np.array(top_idx)
        #print(f'top_idx: {top_idx}')
        #input()
        return top_idx

    def memetic(self):
        #print()
        #start_m = util.sta_show_now_time_msg(msg='run a memetic')
        meme_num = int(self.pop_size * self.pl)
        meme_range = np.arange(self.pop_size)
        #meme_idx = np.random.choice(meme_range, size=meme_num, replace=False)
        #select the number of chromosomes with top rank of self.pl   
        #meme_idx_old = self.top_rank_idx(self.fit_array, self.pl)
        meme_idx = list(map(self.fit_list.index, heapq.nsmallest(meme_num,self.fit_list)))
        #print(self.fit_list)
        #print(f'meme_idx:\n{meme_idx_old}\n{meme_idx}')
        #input()

        # add first_idx (elitism) into meme_idx
        #if 0 not in meme_idx and len(meme_idx) > 0:
        #    meme_idx[0] = 0

        #com = np.hstack((np.arange(self.pop_size).reshape(-1,1),self.fit_array_col))
        #print(f'before memetic:\n{com}')

        for i in meme_idx:
            arg_vec = self.offspring[i]
            arg_val = self.fit_list[i]
            #start1 = util.sta_show_now_time_msg(msg='gen_neighbour')
            neighbour = self.gen_neighbour(arg_vec)
            #util.sta_show_used_time_msg(start1, msg='gen_neighbour')
            len_nei = len(neighbour)
            #print(f'meme idx {i}, old value:{arg_val}')

            #start2 = util.sta_show_now_time_msg(msg='hill climb')
            # hill climbing local search
            for start in range(2):
                step = 0
                start_point = np.random.randint(0, len_nei)
                while(1): 
                    step += 1
                    start_point += 1

                    if step >= len_nei:
                        break
                    if start_point >= len_nei:
                        break

                    nei_val = compute_fitness(self, neighbour[start_point])
                    if nei_val < arg_val:
                        self.offspring[i] = neighbour[start_point]
                        self.fit_array_col[i, 0] = nei_val
                        self.fit_list[i] = nei_val
                    else:
                        break
                #print(f'meme idx {i}, new value:{self.fit_list[i]}')
            #util.sta_show_used_time_msg(start2, msg='hill climb')

        self.parent = copy.deepcopy(self.offspring)
        #util.sta_show_used_time_msg(start_m, msg='run a meme_idx')

        #com2 = np.hstack((np.arange(self.pop_size).reshape(-1,1),self.fit_array_col))
        #print(f'before memetic:\n{com2}')

        first_idx = np.argmin(self.fit_list)
        first_fit = self.fit_list[first_idx]
        first_phen = self.parent[first_idx]
        #print(f'first_idx {first_idx}, first_fit {first_fit} \n{first_phen}')
        #input()
        return self.cur_gen_idx, first_fit, first_phen


    def gen_neighbour(self, vec):
        dictUserServ = generateCommunity(self, vec)
        #print(dictUserServ)

        neighbour = []
        neighbour_num = int(1/self.pl) #self.cell_id_max - self.cell_id_min

        point_loc = np.random.randint(0, self.user_count)
        #point_val = vec[point_loc]

        #print(f'{vec}\npoint_loc:{point_loc}')

        travl = 0
        while(1):
            if len(neighbour) > neighbour_num:
                break

            if travl > self.user_count:
                break

            if point_loc > self.user_count-1:
                point_loc = 0

            community = dictUserServ[point_loc+1]
            #print(f'point_loc:{point_loc} com: {community}')
            # community has overlap loc
            used = []
            for c in community:
                if c in used:
                    continue
                new_vec = copy.deepcopy(vec)
                new_vec[point_loc] = c
                used.append(c)
                neighbour.append(new_vec)
                if len(neighbour) > neighbour_num:
                    break

            point_loc += 1
            travl += 1

        if len(neighbour) > neighbour_num:
            neighbour = neighbour[:neighbour_num]

        if len(neighbour) == 0:
            print('generated neighborhood is null!')
            neighbour.append(vec)
        

        neighbour = np.array(neighbour)
        return neighbour

        

    def run(self):
        #reset
        self.cur_gen_idx = 0
        self.cur_fit_idx = 0

        self.generate_population(self.pop_size, self.user_count)


        while True:
            self.cur_gen_idx += 1

            if (self.cur_gen_idx > self.max_gen):
                #print("exceed max_gen: {0}/{1} ".format(self.cur_gen_idx, self.max_gen))
                break

            if self.cur_fit_idx > self.max_fit:
                #print("exceed max_fit: {0}/{1} ".format(self.cur_fit_idx, self.max_fit))
                break

            #start = util.sta_show_now_time_msg(msg='crossover ')
            self.crossover()
            #util.sta_show_used_time_msg(start, msg='crossover')

            #start = util.sta_show_now_time_msg(msg='mutation ')
            self.mutation()
            #util.sta_show_used_time_msg(start, msg='mutation')

            gen_idx, first_val, first_phen = self.select_result()
            #print(f'select_result\n{gen_idx} {first_val} {first_phen}')

            #start = util.sta_show_now_time_msg(msg='memetic ')
            gen_idx, first_val, first_phen = self.memetic()
            #util.sta_show_used_time_msg(start, msg='memetic')
            #print(f'memetic\n{gen_idx} {first_val} {first_phen}')
            

            if(self.cur_gen_idx == 1):
                self.result.best_gen_idx = gen_idx
                self.result.best_gen_val = first_val
                self.result.best_gen_phen = first_phen
                self.result.add_gen_result(self.result.best_gen_idx, 
                                            self.result.best_gen_val, 
                                            self.result.best_gen_phen)

            if(first_val < self.result.best_gen_val):
                self.result.best_gen_idx = gen_idx
                self.result.best_gen_val = first_val
                self.result.best_gen_phen = first_phen
                self.result.add_gen_result(self.result.best_gen_idx, 
                                            self.result.best_gen_val, 
                                            self.result.best_gen_phen)

            self.result.add_all_gen_result(self.cur_gen_idx, 
                                            self.result.best_gen_val, 
                                            self.result.best_gen_phen)
        return self.query

def get_ideal_features(evo_cls, scene):
    config = evo_cls.query.config
    ideal_hop = 0
    x_inServCapa = scene[config._inServCapa]
    #print(f'x_inServCapa\n{x_inServCapa}')
    x_inServPflNum = scene[config._inServPflNum]
    #print(f'x_inServPflNum\n{x_inServPflNum}')
    # if x_inServPflNum is 0, convert to 1 for representing all capa
    x_inServPflNum[np.where(x_inServPflNum == 0)] = 1
    #print(f'x_inServPflNum\n{x_inServPflNum}')
    x_serv_capa = x_inServCapa/x_inServPflNum
    #print(f'x_serv_capa\n{x_serv_capa}')
    ideal_serv_capa = max(x_serv_capa)
    #print(f'ideal_serv_capa\n{ideal_serv_capa}')
    x_congestion =  scene[config._userQueuLast] + scene[config._userQueu]
    #print(f'x_congestion\n{x_congestion}')
    ideal_congestion = min(x_congestion)
    #print(f'ideal_congestion\n{ideal_congestion}')
    #input()
    return ideal_hop, ideal_serv_capa, ideal_congestion



def generateCommunity(evo_cls, arg_vec):
    config = evo_cls.query.config
    #print(arg_vec)
    tmpScene = copy.deepcopy(evo_cls.scene)
    #print(f'scene\n{tmpScene}')
    scene = edge.updateCurrentScene(config, tmpScene,arg_vec)

    # compute each user's hop, servCapa/UserNum, queue+lastQueue

    dictUserServ = {}
    userCount = config.userCount #globalVar.get_value(enumVar.userCount)
    user = scene[config._userId,:]
    inCell = scene[config._uInCell,:]

    allServUserNum = edge.staAllServUserNum(config, scene)
    # if put user profile in this serv, so add 1
    allServUserNum = allServUserNum + 1
    #print(allServUserNum)
    allServQueue = edge.staAllServQueue(config, scene)
    #print(f'scene\n{scene}')
    #print(f'allServUserNum\n{allServUserNum}')
    #print(f'allServQueue\n{allServQueue}')
    #input()
    ideal_a1, ideal_a2, ideal_a3 = get_ideal_features(evo_cls, scene)

    for idxUser in range(userCount):
        x = inCell[idxUser]
        similarity = np.zeros(config.servCount)
        candiList = []
        for idxServ in range(config.servCount):
            # if put user profile in serv idxServ
            s = idxServ+1 # scene[config._pInServ, idxUser]
            #print(f'x {x}, s {s}')
            a1 = edge.getCellDistance(config, x, s )
            a2 = edge.getServCapa(config, s)/allServUserNum[idxServ]
            a3 = allServQueue[idxServ]
            #print(f'a1:{a1}, a2:{a2}, a3:{a3}')
            similarity[idxServ] = edge.get_similarity(
                ideal_a1, ideal_a2, ideal_a3, a1, a2, a3)
            if similarity[idxServ] > 0.99:
                candiList.append(s)
        if len(candiList) == 0:
            topIdx = np.argmax(similarity)
            candiList.append(topIdx+1)
        # in similarity array, find the top 3 serv
        dictUserServ[idxUser+1] = candiList


    return dictUserServ


def compute_fitness(evo_cls, arg_vec):
    config = evo_cls.query.config
    tmpScene = copy.deepcopy(evo_cls.scene)
    #evo_cls.cur_fit_idx += 1
    

    newScene = edge.updateCurrentScene(config, tmpScene,arg_vec)
    cmni,cmpt,miga = edge.computeScene(config, newScene)
    fitVal = edge.statisticLatency(cmni,cmpt,miga)

    #evo_cls.result.add_all_fit_result(evo_cls.cur_fit_idx, fitVal, arg_vec)

    return fitVal

    

def run_Single_MA_CDLS(query):
    config = query.config
    ga = Single_MA_CDLS(query)
    #print('begin')

    # result = globalVar.Result()
    #start = util.sta_show_now_time_msg(msg='evo run')
    ga.run()
    #util.sta_show_used_time_msg(start, msg='evo run end,')

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


    bestEngy = edge.computeEnergyWithPhen(config, ga.result.best_phen)
    #arrayGen = np.array(ga.listGen).reshape((-1,1))
    #arrayGenBest = np.array(ga.listGenBest).reshape((-1,1))
    #table = np.hstack((arrayGen,arrayGenBest))

    table = ga.result.get_dict_as_table(ga.result.dict_gen)
    fileName = ga.query.get_file_name()
    title = ['gen_idx', 'best_val']
    edge.save_table_to_csv(config, table, title, fileName, prefix='log', surfix='gen')

    table2 = ga.result.get_dict_as_table(ga.result.dict_fit)
    fileName2 = ga.query.get_file_name()
    title2 = ['fit_idx', 'best_val']
    edge.save_table_to_csv(config, table2, title2, fileName2, prefix='log', surfix='fit')
    #print('end')

    return query  # bestGen, bestCall, bestVal, bestEngy, bestPhen

