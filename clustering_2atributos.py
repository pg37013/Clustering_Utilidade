import pandas as pd
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from Clusters import *

class Dim_2:

    def total_eventos(self,dataset):
        sum = 0
        for j in range(len(dataset)):
            sum += dataset.loc[j, 'N']  # número total de indíviduos
        return sum

    def rep_pontual(self,dataset):
        # Calcular o representante de um conjunto, conjunto este que tem o atributo e o respetivo peso associado
        sum = self.total_eventos(dataset)
        rep1 = 0
        for i in range(len(dataset)):
            rep1 += (dataset.loc[i,'a1'] * dataset.loc[i,'N'])
        rep1 = rep1 / sum
        rep2 = 0
        for i in range(len(dataset)):
            rep2 += (dataset.loc[i, 'a2'] * dataset.loc[i, 'N'])
        rep2 = rep2 / sum
        return rep1, rep2

    def util_gross(self, dataset, fun_util):
        rep1, rep2 = self.rep_pontual(dataset)
        if fun_util=="soma_quad":
            util = math.pow(rep1,2)+math.pow(rep2,2)
        elif fun_util=="soma_lin_quad":
            util = rep1 + math.pow(rep2, 2)
        elif fun_util == "soma_quad_lin":
            util = math.pow(rep1, 2) + rep2
        elif fun_util == "soma_lin_raiz":
            util = rep1 + math.sqrt(rep2)
        elif fun_util == "soma_raiz_lin":
            util =  math.sqrt(rep1) + rep2
        elif fun_util == 'soma_raiz':
            util = math.sqrt(rep1) + math.sqrt(rep2)
        return util

    def util_fina(self, dataset, fun_util):
        sum = self.total_eventos(dataset)
        util = 0
        for i in range(len(dataset)):
            if fun_util=="soma_quad":
                util += (math.pow(dataset.loc[i, 'a1'],2)+math.pow(dataset.loc[i, 'a2'],2))*(dataset.loc[i,'N'])
            elif fun_util=="soma_lin_quad":
                util += (dataset.loc[i, 'a1'] + math.pow(dataset.loc[i, 'a2'], 2))*(dataset.loc[i,'N'])
            elif fun_util == "soma_quad_lin":
                util += (math.pow(dataset.loc[i, 'a1'], 2) + dataset.loc[i, 'a2'])*(dataset.loc[i,'N'])
            elif fun_util == "soma_lin_raiz":
                util += (dataset.loc[i, 'a1'] + math.sqrt(dataset.loc[i, 'a2']))*(dataset.loc[i,'N'])
            elif fun_util == "soma_raiz_lin":
                util += (math.sqrt(dataset.loc[i, 'a1']) + dataset.loc[i, 'a2'])*(dataset.loc[i,'N'])
            elif fun_util == 'soma_raiz':
                util += (math.sqrt(dataset.loc[i, 'a1']) + math.sqrt(dataset.loc[i, 'a2'])) * (dataset.loc[i, 'N'])
        return util/sum

    def utilidade(self, dataset_orig, particao, fun_util):
        sum = self.total_eventos(dataset_orig)
        # particao vem na forma de lista
        util = 0
        for i in range(len(particao)):
            part  = pd.DataFrame(particao[i], columns=['a1', 'a2', 'N'])
            rep1, rep2 = self.rep_pontual(part)
            part_sum = self.total_eventos(part)
            if fun_util == "soma_quad":
                util += ((math.pow(rep1, 2) + math.pow(rep2, 2)) * part_sum)
            elif fun_util == "soma_lin_quad":
                util += ((rep1 + math.pow(rep2, 2)) * part_sum)
            elif fun_util == "soma_quad_lin":
                util += ((math.pow(rep1, 2) + rep2) * part_sum)
            elif fun_util == "soma_lin_raiz":
                util += ((rep1 + math.sqrt(rep2)) * part_sum)
            elif fun_util == "soma_raiz_lin":
                util += ((math.sqrt(rep1) + rep2) * part_sum)
            elif fun_util == 'soma_raiz':
                util += ((math.sqrt(rep1) + math.sqrt(rep2)) * part_sum)
        return util/sum

    def particoes(self,dataset, k):
        #Função que retorna uma lista com todas as partições possíveis do 'dataset'
        n = len(dataset)
        u = list(itertools.product([i for i in range(1, n)], repeat=k))
        particao = []
        for e in u:
            if sum(e) == n:
                out = []
                b = dataset
                for i in e:
                    out.append(b[:i])
                    b = b[i:]
                particao.append(out)
        return particao

    def recuperar_info(self, dataset, particao, tipo):
        if tipo == 'horizontal':
            atributo ='a2'
        elif tipo == 'vertical':
            atributo = 'a1'
        else:
            print('Qual o tipo de corte?')
        new = []
        for i in particao:
            a = []
            for j in i:
                u = []
                for t in range(len(j)):
                    PART_H = []
                    for l in range(len(dataset)):
                        if j[t][0] == dataset.loc[l, atributo]:
                            PART_H.append([dataset.loc[l, 'a1'], dataset.loc[l, 'a2'], dataset.loc[l, 'N']])
                    u.extend(PART_H)
                a.append(u)
            new.append(a)
        return new

    def loss(self,dataset, particao, fun_util):
        fina=self.util_fina(dataset, fun_util)
        util_part=self.utilidade(dataset, particao, fun_util)
        loss=np.abs(fina-util_part)
        return loss

    def all_parts(self,lst, n):
        result = []
        for k in range(int(math.pow(n, len(lst)))):
            # initialize result
            res = []
            for i in range(n):
                subsublist = []
                sublist = [[]] * i
                res.append(sublist)
            k2 = k
            for i in range(len(lst)):
                res[int(k2 % n)].append(lst[i])
                k2 /= n
            result.append(res)
        result = self.remove_lst(result)
        result.sort()
        result = list(result for result, _ in itertools.groupby(result))
        return result

    def remove_lst(self,lst):
        '''
        Remover a ocurrência de listas vazias
        '''
        if not isinstance(lst, list):
            return lst
        else:
            return [x for x in map(self.remove_lst, lst) if (x != [] and x != '')]

    def part_opt_old(self,dataset, P, k, f):
        '''
        Função que retorna a partição ótima
        '''
        for fun_util in f:
            u = []
            for i in P:
                util = dim2.utilidade(dataset, i, fun_util)
                u.append(util)
            if fun_util == 'soma_quad' or fun_util == 'soma_lin_quad' or fun_util == 'soma_quad_lin':
                util_opt = max(u)
                part_opt = P[u.index(util_opt)]
            elif fun_util == 'soma_lin_raiz' or fun_util == 'soma_raiz_lin' or fun_util == 'soma_raiz':
                util_opt = min(u)
                part_opt = P[u.index(util_opt)]
            print('000',part_opt)
        return part_opt

    def part_opt(self,dataset, P, fun_util):
        '''
        Função que retorna a partição ótima
        '''
        u = []
        for i in P:
            util = dim2.utilidade(dataset, i, fun_util)
            u.append(util)
        if fun_util == 'soma_quad' or fun_util == 'soma_lin_quad' or fun_util == 'soma_quad_lin':
            util_opt = max(u)
            part_opt = P[u.index(util_opt)]
        elif fun_util == 'soma_lin_raiz' or fun_util == 'soma_raiz_lin' or fun_util == 'soma_raiz':
            util_opt = min(u)
            part_opt = P[u.index(util_opt)]
        return part_opt

    def particoes(self, dataset, k):
        # Função que retorna uma lista com todas as partições possíveis do 'dataset'
        n = len(dataset)
        u = list(itertools.product([i for i in range(1, n)], repeat=k))
        particao = []
        for e in u:
            if sum(e) == n:
                out = []
                b = dataset
                for i in e:
                    out.append(b[:i])
                    b = b[i:]
                particao.append(out)
        return particao

    def recuperar_info(self, dataset, particao, tipo):
        if tipo == 'horizontal':
            atributo = 'a2'
        elif tipo == 'vertical':
            atributo = 'a1'
        else:
            print('Qual o tipo de corte?')
        new = []
        for i in particao:
            a = []
            for j in i:
                u = []
                for t in range(len(j)):
                    PART_H = []
                    for l in range(len(dataset)):
                        if j[t][0] == dataset.loc[l, atributo]:
                            PART_H.append([dataset.loc[l, 'a1'], dataset.loc[l, 'a2'], dataset.loc[l, 'N']])
                    u.extend(PART_H)
                a.append(u)
            new.append(a)
        return new

    def rho(self, C, fun_util):
        c = pd.DataFrame(C, columns=['a1', 'a2', 'N'])
        sum = self.total_eventos(c)
        rep1, rep2 = self.rep_pontual(c)
        r = 0
        for a in C:
            if fun_util == "soma_quad":
                r+=((math.pow(a[0],2)+math.pow(a[1],2))-(math.pow(rep1,2)+math.pow(rep2,2)))
            elif fun_util == "soma_lin_quad":
                r += ((a[0] + math.pow(a[1], 2)) - (rep1 + math.pow(rep2, 2)))
            elif fun_util == "soma_quad_lin":
                r += ((math.pow(a[0], 2) + a[1]) - (math.pow(rep1, 2) + rep2))
            elif fun_util == "soma_lin_raiz":
                r += ((a[0] + math.sqrt(a[1])) - (rep1 + math.sqrt(rep2)))
            elif fun_util == "soma_raiz_lin":
                r += ((math.sqrt(a[0]) + a[1]) - (math.sqrt(rep1) + rep2))
            elif fun_util == 'soma_raiz':
                r += ((math.sqrt(a[0]) + math.sqrt(a[1])) - (math.sqrt(rep1) + math.sqrt(rep2)))
        return r/sum

    def delta_loss(self, data_orig, C, C_menos, C_mais, fun_util):
        c = pd.DataFrame(C, columns=['a1', 'a2', 'N'])
        c_menos = pd.DataFrame(C_menos, columns=['a1', 'a2', 'N'])
        c_mais = pd.DataFrame(C_mais, columns=['a1', 'a2', 'N'])
        sum = self.total_eventos(data_orig)
        sum_c = self.total_eventos(c)
        sum_menos = self.total_eventos(c_menos)
        sum_mais = self.total_eventos(c_mais)
        DL = (sum_c/sum)*np.abs(self.rho(C, fun_util))-(sum_menos/sum)*np.abs(self.rho(C_menos, fun_util))-(sum_mais/sum)*np.abs(self.rho(C_mais, fun_util))
        return DL

    def ganho(self, dataset, particao, fun_util):
        fina = self.util_fina(dataset, fun_util)
        util_part = self.utilidade(dataset, particao, fun_util)
        ganho = np.abs(fina - util_part)
        return ganho

    # ------------------------- CART -------------------------

    def remove_duplicates(self,l):
        return list(set(l))

    def make_cut(self, d, fun_util):
        dataset = pd.DataFrame(d, columns=['a1', 'a2', 'N'])
        a1 = []
        for i in range(len(dataset)):
            a1.append([dataset.loc[i, 'a1'], dataset.loc[i, 'N']])
        l1 = []
        for i in a1:
            l1.append(i[0])
        L1 = self.remove_duplicates(l1)

        A1 = []
        for i in L1:
            n = 0
            for j in a1:
                if i == j[0]:
                    n += j[1]
            A1.append([i,n])

        a2 = []
        for i in range(len(dataset)):
            a2.append([dataset.loc[i, 'a2'], dataset.loc[i, 'N']])

        l2 = []
        for i in a2:
            l2.append(i[0])
        L2 = self.remove_duplicates(l2)

        A2 = []
        for i in L2:
            n = 0
            for j in a2:
                if i == j[0]:
                    n += j[1]
            A2.append([i, n])

        if len(A1) <= 1:
            min_V = math.inf
        else:
            # Corte Vertical
            part_V = self.particoes(A1, 2)
            new_V = self.recuperar_info(dataset, part_V, 'vertical')
            ganho_V = []
            for i in range(len(new_V)):
                g = self.ganho(dataset, new_V[i], fun_util)
                ganho_V.append(g)
            min_V = min(ganho_V)
            particao_v = new_V[ganho_V.index(min_V)]

        if len(A2) <= 1:
            min_H = math.inf
        else:
            # Corte Horizontal
            part_H = self.particoes(A2, 2)
            new_H = self.recuperar_info(dataset, part_H, 'horizontal')
            ganho_H = []
            for i in range(len(new_H)):
                g = self.ganho(dataset, new_H[i], fun_util)
                ganho_H.append(g)
            min_H = min(ganho_H)
            particao_h = new_H[ganho_H.index(min_H)]
        if min_H <= min_V:
            part_opt = particao_h
        else:
            part_opt = particao_v
        return part_opt

    def cart(self, data_orig, dataset, fun_util, k):
        return self.cart_aux(data_orig, dataset, fun_util, k)

    def cart_aux(self, data_orig, dataset, fun_util, k):
        dataset_orig_pd = pd.DataFrame(data_orig, columns=['a1', 'a2', 'N'])
        if k == 1:
            return dataset
        elif k > 1:
            P = []
            for i in range(len(dataset)):
                d = dataset.copy()
                p = []
                if len(dataset[i]) == 1:
                    # Não faz sentido dividir em dois, quando o conjunto tem apenas 1 ponto
                    continue
                particao = self.make_cut(dataset[i], fun_util)
                d.pop(i)
                p.extend(particao)
                p.extend(d)
                P.append(p)
            G = []
            for j in range(len(P)):
                particao = P[j]
                G.append(self.ganho(dataset_orig_pd, particao, fun_util))
            min_g = min(G)
            part_opt = P[G.index(min_g)]
            return self.cart(data_orig, part_opt, fun_util, k - 1)