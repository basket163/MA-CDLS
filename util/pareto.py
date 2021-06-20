import numpy as np
import copy

class Phen(object):
    def __init__(self, f1, f2, phen):
        self.f1 = f1
        self.f2 = f2
        self.phen = phen

class Pareto(object):
    '''get Pareto and knee point'''
    def __init__(self):
        self.phen_cls_list = []
        '''
        self.f1_list = []
        self.f2_list = []
        self.phen_matrix = None
        self.pareto = {}
        self.knee_point = None
        '''

    def add_phen(self, f1, f2, phen):
        p = Phen(f1, f2, phen)
        self.phen_cls_list.append(p)
        '''
        self.f1_list.append(f1)
        self.f2_list.append(f2)
        if self.phen_matrix == None:
            self.phen_matrix = phen
        else:
            np.vstack((self.phen_matrix, phen))
        '''

    def update_phen(self, f1_old, f2_old, phen_old, f1_new, f2_new, phen_new):
        p_new = Phen(f1_new, f2_new, phen_new)
        for i, p in enumerate(self.phen_cls_list):
            if p.f1 == f1_old and p.f2 == f2_old and (p.phen == phen_old).all():
                self.phen_cls_list[i].f1 = f1_new
                self.phen_cls_list[i].f2 = f2_new
                self.phen_cls_list[i].phen = phen_new
        '''
        self.f1_list.append(f1)
        self.f2_list.append(f2)
        if self.phen_matrix == None:
            self.phen_matrix = phen
        else:
            np.vstack((self.phen_matrix, phen))
        '''

    def add_pop(self, f1_list, f2_list, pop):
        for i in np.arange(len(f1_list)):
            #print(f'add {f1}, {f2_list[i]}, {pop[i, :]}')
            p = Phen(f1_list[i], f2_list[i], pop[i,:])
            self.phen_cls_list.append(p)
        '''
        self.f1_list.extend(f1_list)
        self.f2_list.extend(f2_list)
        #self.phen_list.extend(phen_list)
        if self.phen_matrix == None:
            self.phen_matrix = pop
        else:
            np.vstack((self.phen_matrix, pop))
        '''

    def get_pareto_front_list(self):
        dict_pareto = self.obtain_pareto_front_dict()
        pareto_f1, pareto_f2, phen_pareto = self.convert_pareto_front_to_list(dict_pareto)
        return pareto_f1, pareto_f2, phen_pareto

    def obtain_pareto_front_dict(self):
        dict_pareto = {}
        '''
        print(f'len cls: {len(self.phen_cls_list)}')
        for p in self.phen_cls_list:
            print(f'{p.f1} {p.f2}')
        print('end')
        '''
        #input('get_pareto_front')
        for p in self.phen_cls_list:
            if p.f1 not in dict_pareto.keys():
                dict_pareto[p.f1] = p
            else:
                if p.f2 < dict_pareto[p.f1].f2:
                    dict_pareto[p.f1] = p


        '''
        print('after pareto')
        for k,v in dict_pareto.items():
            print(f'{v.f1} {v.f2}')
        input('pause')
        '''
        return dict_pareto

    def get_pareto_front_distinct_list(self):
        dict_pareto = self.obtain_pareto_front_dict_distinct()
        pareto_f1, pareto_f2, phen_pareto = self.convert_pareto_front_to_list(dict_pareto)
        return pareto_f1, pareto_f2, phen_pareto

    def obtain_pareto_front_dict_distinct(self):
        # select distinct f1
        dict_pareto = {}
        for p in self.phen_cls_list:
            if p.f1 not in dict_pareto.keys():
                dict_pareto[p.f1] = p
            else:
                if p.f2 < dict_pareto[p.f1].f2:
                    dict_pareto[p.f1] = p

        # select distinct f2
        dict_pareto_f2 = {}
        for k,p in dict_pareto.items():
            if p.f2 not in dict_pareto_f2.keys():
                dict_pareto_f2[p.f2] = p
            else:
                if p.f1 < dict_pareto_f2[p.f2].f1:
                    dict_pareto_f2[p.f2] = p

        # change dict key from f2 to f1
        dict_pareto_last = {}
        list_f1 = []
        list_f2 = []
        for k,p in dict_pareto_f2.items():
            dict_pareto_last[p.f1] = p
            list_f1.append(p.f1)
            list_f2.append(p.f2)
        len_pareto = len(list_f1)

        # select dominant point
        dict_dominant = {}
        for k,p in dict_pareto_last.items():
            for idx in range(len_pareto):
                if list_f1[idx] < p.f1 and list_f2[idx] < p.f2:
                    dict_dominant[k] = p
                    break

        # select non-dominant point
        dict_non_dominant = {}
        for k, p in dict_pareto_last.items():
            if k not in dict_dominant.keys():
                dict_non_dominant[k] = p
        return dict_non_dominant

    def convert_pareto_front_to_list(self, dict_pareto):
        # convert dict_pareto
        pareto_f1 = []
        pareto_f2 = []
        phen_pareto = None
        for k,p in dict_pareto.items():
            pareto_f1.append(p.f1)
            pareto_f2.append(p.f2)
            if phen_pareto is None:
                phen_pareto = p.phen
            else:
                phen_pareto = np.vstack((phen_pareto, p.phen))
        #print('convert_pareto_front_to_list')
        #print(pareto_f1)
        #print(pareto_f2)
        return pareto_f1, pareto_f2, phen_pareto

    def get_knee_point(self):
        dict_pareto = self.obtain_pareto_front_dict()
        pareto_f1, pareto_f2, phen_matrix = self.convert_pareto_front_to_list(dict_pareto)
        idx_array = np.arange(len(pareto_f1))
        #print(idx_array)
        #print(pareto_f1)
        #print(pareto_f2)
        #input()
        distance_arr, rank_score = self.rank_distance(idx_array, pareto_f1, pareto_f2)
        r = 0
        idx = np.where(rank_score == r)[0][0]  # + 1
        first_f1 = pareto_f1[idx]
        first_f2 = pareto_f2[idx]
        first_phen = phen_matrix[idx]
        return first_f1, first_f2, first_phen

    def rank_distance(self, idx_array, f1_array, f2_array):
        distance_list = []
        point1, point2 = self.select_extreme_points(f1_array, f2_array)
        for i in idx_array:
            point = np.array([f1_array[i], f2_array[i]])
            distance_list.append(self.point_distance_line(point, point1, point2))
        index_score = np.argsort(-np.array(distance_list))
        rank_score = np.argsort(index_score)  # +1
        distance_arr = np.array(distance_list)
        return distance_arr, rank_score

    def point_distance_line(self, point, line_point1, line_point2):
        # compute the distance from a point to a line
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance

    def select_extreme_points(self, f1_array, f2_array):
        if len(f1_array) < 3:
            #print('select_extreme_points, array num < 3.')
            pass
        # select f1 max
        max_f1_idx = np.argmax(f1_array)
        max_f1 = f1_array[max_f1_idx]
        max_f1_same_list = []
        for i, x in enumerate(f1_array):
            if x == max_f1:
                max_f1_same_list.append(f2_array[i])
        #print(f'max_f1 {max_f1}: {max_f1_same_list}')
        # when f1 is max, select min f2


        # select f2 min
        min_f2_idx = np.argmin(f2_array)
        min_f2 = f2_array[min_f2_idx]
        min_f2_same_list = []
        for i, x in enumerate(f2_array):
            if x == min_f2:
                min_f2_same_list.append(f1_array[i])
        # when f2 is min, select max f1
        #print(f'min_f2 {min_f2}: {min_f2_same_list}')

        point1 = np.array([min(min_f2_same_list), min_f2])
        point2 = np.array([max_f1, max(max_f1_same_list)])
        #print(point1)
        #print(point2)
        #input()
        '''
        new_point_flag = False
        if far_f1_idx == far_f2_idx:
            print(f'\nextreme points are the same!')
            # combine = np.hstack((np.arange(len(f1_array)).reshape(-1,1),f1_array.reshape(-1,1),f2_array.reshape(-1,1)))
            # print(f'the same index: {far_f1_idx}\n{combine}')
            tmp_f1 = copy.deepcopy(f1_array)
            tmp_f2 = copy.deepcopy(f2_array)
            allow = 5
            while (allow):
                allow = allow - 1
                if far_f1_idx != far_f2_idx:
                    break
                if allow == 1:
                    new_point_flag = True
                    break

                tmp_f1[far_f1_idx] = min(tmp_f1) - 1
                tmp_f2[far_f1_idx] = min(tmp_f2) - 1
                far_f1_idx = np.argmax(tmp_f1)
                far_f2_idx = np.argmax(tmp_f2)

                print(f'new {far_f1_idx} {far_f2_idx} ')
            # input()
        '''
        #point1 = np.array([f1_array[far_f1_idx], f2_array[far_f1_idx]])
        #point2 = np.array([f1_array[far_f2_idx], f2_array[far_f2_idx]])
        '''
        if new_point_flag:
            sigle_point = (f1_array[far_f1_idx], f2_array[far_f2_idx])
            zero_point = (0, 0)
        '''
        return point1, point2

    def get_top_pareto_pop_then_resize(self, size):
        ret_phen_list = self.obtain_copied_pop_with_size(size)
        list_f1, list_f2, list_phen = self.convert_pareto_pop_to_list(ret_phen_list)
        # important
        self.phen_cls_list = ret_phen_list
        return list_f1, list_f2, list_phen

    def obtain_copied_pop_with_size(self, size):
        ori_phen_list = copy.deepcopy(self.phen_cls_list)
        ret_phen_list = []
        #print(f'ori phen len: {len(ori_phen_list)} ret: {len(ret_phen_list)}, pop_size {size}')

        ret_phen_list = self.recursive_get_pareto(ori_phen_list,ret_phen_list,size)
        #print(f'after recursive {len(ret_phen_list)}')
        #for p in ret_phen_list:
        #    print(f'{p.f1} {p.f2}: {p.phen}')
        return ret_phen_list

    def convert_pareto_pop_to_list(self, ret_phen_list):
        list_f1 = []
        list_f2 = []
        list_phen = None
        for p in ret_phen_list:
            list_f1.append(p.f1)
            list_f2.append(p.f2)
            if list_phen is None:
                list_phen = p.phen
            else:
                list_phen = np.vstack((list_phen, p.phen))
        return list_f1, list_f2, list_phen


    def recursive_get_pareto(self, phen_list, ret_phen, ret_size):
        dict_pareto = {}
        for i, p in enumerate(phen_list):
            if p.f1 not in dict_pareto.keys():
                dict_pareto[p.f1] = p
            else:
                if p.f2 > dict_pareto[p.f1].f2:
                    dict_pareto[p.f1] = p
        for k,v in dict_pareto.items():
            phen_list.remove(v)
            ret_phen.append(v)
            if len(ret_phen) >= ret_size:
                return ret_phen
        return self.recursive_get_pareto(phen_list, ret_phen, ret_size)

def test():
    print('test pareto')
    pareto = Pareto()
    '''
    p1 = np.array([0,0,1])
    p2 = np.array([0,1,0])
    p3 = np.array([0,1,1])
    p4 = np.array([1,1,1])
    pareto.add_phen(5,10, p1)
    pareto.add_phen(6,14, p2)
    pareto.add_phen(7,13, p3)
    pareto.add_phen(8,12, p4)
    
    
    
    dict_pareto = pareto.get_pareto_front()
    pareto_f1, pareto_f2, phen_matrix = pareto.convert_dict_pareto(dict_pareto)
    col_f1 = np.array(pareto_f1).reshape(-1,1)
    col_f2 = np.array(pareto_f2).reshape(-1,1)
    print(np.hstack((col_f1,col_f2)))
    '''

    # 2
    f1_list = [5,6,7,8]
    f2_list = [9,8,9,11]
    matrix = np.zeros((4,3)).reshape(4,3)
    pareto.add_pop(f1_list,f2_list,matrix)

    p1 = np.array([0,0,1])
    p2 = np.array([0,1,0])
    p3 = np.array([0,1,1])
    p4 = np.array([1,1,1])
    pareto.add_phen(5,10, p1)
    pareto.add_phen(6,14, p2)
    pareto.add_phen(7,8, p3)
    pareto.add_phen(8,12, p4)

    print_pareto_knee(pareto)

    # pareto.obtain_copied_pop_with_size(3)
    # pareto.obtain_copied_pop_with_size(4)
    # pareto.obtain_copied_pop_with_size(5)

    # answer: 6, 14: 0,1,0
    p2_old = np.array([0, 1, 0])
    p2_new = np.array([1, 1, 0])
    pareto.update_phen(6, 14, p2_old, 6, 15, p2_new)

    print('updated a phen')
    print_pareto_knee(pareto)


def print_pareto_knee(pareto):
    dict_pareto = pareto.obtain_pareto_front_dict()
    pareto_f1, pareto_f2, phen_matrix = pareto.convert_pareto_front_to_list(dict_pareto)
    col_f1 = np.array(pareto_f1).reshape(-1,1)
    col_f2 = np.array(pareto_f2).reshape(-1,1)
    print(np.hstack((col_f1,col_f2, phen_matrix)))
    print()

    first_f1, first_f2, first_phen = pareto.get_knee_point()
    print(f'{first_f1}, {first_f2}: {first_phen}')



#test()