import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import itertools
from itertools import permutations
import pandas as pd
import random

class Dim_1:

    def tot_subconjunto(self,part):
        sum = 0
        for j in range(len(part)):
            sum += part[j][1]
        return sum

    def rep_pontual(self,conjunto):
        # Calcular o representante de um conjunto, conjunto este que tem o atributo e o respetivo peso associado
        sum = 0
        rep = 0
        for j in range(len(conjunto)):
            sum += conjunto[j][1]  # número total de indíviduos
        for i in range(len(conjunto)):
            rep += conjunto[i][0] * conjunto[i][1]
        rep = rep / sum
        return rep

    def extra(self,dataset):
        extra = 0
        e=self.e(dataset)
        for x in range(len(dataset)):
            extra += ((dataset.loc[x,'a']-e)*(dataset.loc[x,'a']-e)*(dataset.loc[x,'a']-e))
        return extra

    def e(self,dataset):
        sum = 0
        for j in range(len(dataset)):
            sum += dataset.loc[j, 'N']
        e=0
        for i in range(len(dataset)):
            e += (dataset.loc[i,'a'] * (dataset.loc[i,'N']/sum))
        print('Soma: ', sum)
        return e

    def var(self,dataset):
        e = self.e(dataset)
        sum = 0
        for j in range(len(dataset)):
            sum += dataset.loc[j, 'N']
        var=0
        for x in range(len(dataset)):
            var += ((dataset.loc[x, 'a']-e)*(dataset.loc[x, 'a']-e)*dataset.loc[x,'N'])
        var=var/sum
        return var

    def skew(self, dataset):
        e = self.e(dataset)
        var = self.var(dataset)
        sum = 0
        for j in range(len(dataset)):
            sum += dataset.loc[j, 'N']
        skewness = 0
        for x in range(len(dataset)):
            skewness += ((dataset.loc[x, 'a'] - e) * (dataset.loc[x, 'a'] - e) * (dataset.loc[x, 'a'] - e) * dataset.loc[x, 'N'])
        skewness = skewness / sum
        skewness=(skewness/(np.sqrt(var*var*var)))
        return skewness

    def utilidade_Fina(self,dataset,fun_util):
        sum = 0
        for j in range(len(dataset)):
            sum += dataset.loc[j, 'N']
        util=0
        for i in range(len(dataset)):
            if fun_util == 'quad':
                util+=(dataset.loc[i,'N']*dataset.loc[i,'a']*dataset.loc[i,'a'])
            elif fun_util == 'cub':
                util += (dataset.loc[i, 'N'] * dataset.loc[i, 'a'] * dataset.loc[i, 'a']* dataset.loc[i, 'a'])
            elif fun_util == 'raiz':
                util += (dataset.loc[i, 'N'] * math.sqrt(dataset.loc[i, 'a']))
            elif fun_util == 'ent':
                util += (dataset.loc[i, 'N']* dataset.loc[i, 'a']*math.log(dataset.loc[i, 'a'],2))
            elif fun_util == 'pol':
                util += (dataset.loc[i, 'N'] * dataset.loc[i, 'a'] * (dataset.loc[i, 'a']-1))
            elif fun_util == 'pol1':
                util += (dataset.loc[i, 'N'] * dataset.loc[i, 'a'] * (dataset.loc[i, 'a']+1))
            elif fun_util == 'pol2':
                util += (dataset.loc[i, 'N'] * dataset.loc[i, 'a'] * (dataset.loc[i, 'a']-10))
            elif fun_util == 'pol3':
                util += (dataset.loc[i, 'N'] * dataset.loc[i, 'a'] * (dataset.loc[i, 'a']+10))
            elif fun_util=='exp':
                util += (dataset.loc[i, 'N']*math.exp(dataset.loc[i, 'a']))
            elif fun_util=='exp2':
                util += (dataset.loc[i, 'N']*math.exp(dataset.loc[i, 'a']+1))
            elif fun_util=='exp1':
                util += (dataset.loc[i, 'N']*math.exp(dataset.loc[i, 'a']-1))

            elif fun_util == 'exp_1':
                util += (dataset.loc[i, 'N'] * (-math.exp(-dataset.loc[i, 'a']/9105)))
            elif fun_util == 'exp_2':
                util += (dataset.loc[i, 'N'] * (-2*math.exp(-2*dataset.loc[i, 'a'])))
            elif fun_util == 'exp_4':
                util += (dataset.loc[i, 'N'] * (-4*math.exp(-4*dataset.loc[i, 'a'])))

            else:
                print('Erro: Função Inválida!')
        return util/sum

    def utilidade_Grosseira(self,dataset,fun_util):
        rep=self.rep_pontual(dataset)
        if fun_util=='quad':
            util=rep*rep
        elif fun_util=='cub':
            util=rep*rep*rep
        elif fun_util=='raiz':
            util=math.sqrt(rep)
        elif fun_util=='ent':
            util=rep*math.log(rep,2)
        elif fun_util == 'pol':
            util=rep*(rep-1)
        elif fun_util == 'pol1':
            util=rep*(rep+1)
        elif fun_util == 'pol2':
            util=rep*(rep-10)
        elif fun_util == 'pol3':
            util=rep*(rep+10)
        elif fun_util == 'exp':
            util=math.exp(rep)
        elif fun_util == 'exp1':
            util=math.exp(rep-1)
        elif fun_util == 'exp2':
            util=math.exp(rep+1)

        elif fun_util=='exp_1':
            util = -math.exp(-rep/9105)
        elif fun_util == 'exp_2':
            util = -2*math.exp(-2*rep)
        elif fun_util == 'exp_4':
            util = -4 * math.exp(-4 * rep)

        else:
            print('Erro: Função Inválida!')
        return util

    def utilidade_CART(self,particao,fun_util):
        util=0
        T=0
        for part in particao:
            rep=self.rep_pontual(part)
            tot=0
            for i in range(len(part)):
                tot+=int(part[i][1])
                T+=int(part[i][1])
            if fun_util=='quad':
                util += (rep * rep * tot)
            elif fun_util=='cub':
                util += (rep * rep * rep * tot)
            elif fun_util=='raiz':
                util += (math.sqrt(rep) * tot)
            elif fun_util == 'ent':
                util += (rep * math.log(rep, 2) * tot)
            elif fun_util == 'pol':
                util += (rep * (rep-1) * tot)
            elif fun_util == 'pol1':
                util += (rep * (rep+1) * tot)
            elif fun_util == 'pol2':
                util += (rep * (rep-10) * tot)
            elif fun_util == 'pol3':
                util += (rep * (rep+10) * tot)
            elif fun_util == 'exp':
                util += (math.exp(rep)*tot)
            elif fun_util == 'exp1':
                util += (math.exp(rep-1)*tot)
            elif fun_util == 'exp2':
                util += (math.exp(rep+1)*tot)

            elif fun_util == 'exp_1':
                util += ((-math.exp(-rep/9105))*tot)
            elif fun_util == 'exp_2':
                util = ((-2 * math.exp(-2 * rep))*tot)
            elif fun_util == 'exp_4':
                util = ((-4 * math.exp(-4 * rep))*tot)

            else:
                print('Erro: Função Inválida!')
        return util/T

    def utilidade(self,particao,fun_util):
        U=[] #lista que guarda a utilidade de cada partição
        P=[] #lista com as partições cuja utilidade estão no vetor U
        for c in range(len(particao)):
            P.append(particao[c])
            util = 0
            tot = 0 #número total de indivíduos do dataset
            for i in range(len(particao[c])):
                rep = self.rep_pontual(particao[c][i])
                tot_sub=0
                for j in range(len(particao[c][i])):
                    tot += float(particao[c][i][j][1])
                    tot_sub+=float(particao[c][i][j][1])
                if fun_util=='quad':
                    util += (rep * rep * tot_sub)
                elif fun_util=='cub':
                    util += (rep * rep * rep * tot_sub)
                elif fun_util == 'raiz':
                    util += (math.sqrt(rep) * tot_sub)
                elif fun_util == 'ent':
                    util += (rep * math.log(rep, 2) * tot_sub)
                elif fun_util == 'pol':
                    util += (rep * (rep - 1) * tot_sub)
                elif fun_util == 'pol1':
                    util += (rep * (rep + 1) * tot_sub)
                elif fun_util == 'pol2':
                    util += (rep * (rep - 10) * tot_sub)
                elif fun_util == 'pol3':
                    util += (rep * (rep + 10) * tot_sub)
                elif fun_util == 'exp':
                    util += (math.exp(rep) * tot_sub)
                elif fun_util == 'exp1':
                    util += (math.exp(rep-1) * tot_sub)
                elif fun_util == 'exp2':
                    util += (math.exp(rep+1) * tot_sub)

                elif fun_util == 'exp_1':
                    util += ((-math.exp(-rep/9105)) * tot_sub)
                elif fun_util == 'exp_2':
                    util = ((-2 * math.exp(-2 * rep)) * tot_sub)
                elif fun_util == 'exp_4':
                    util = ((-4 * math.exp(-4 * rep)) * tot_sub)

                else:
                    print('Erro: Função Inválida!')
            util=(util/tot)
            U.append(util)
        return U, P

    def ganho(self,dataset,utilidade,fun_util):
        rep = self.rep_pontual(dataset)
        if fun_util=='quad':
            ganho = (rep * rep) - utilidade
        elif fun_util=='cub':
            ganho = (rep * rep * rep) - utilidade
        elif fun_util=='raiz':
            ganho = (math.sqrt(rep)) - utilidade
        elif fun_util=='ent':
            ganho = (rep*math.log(rep,2)) - utilidade
        elif fun_util == 'pol':
            ganho = (rep * (rep-1)) - utilidade
        elif fun_util == 'pol1':
            ganho = (rep * (rep+1)) - utilidade
        elif fun_util == 'pol2':
            ganho = (rep * (rep-10)) - utilidade
        elif fun_util == 'pol3':
            ganho = (rep * (rep+10)) - utilidade
        elif fun_util == 'exp':
            ganho = math.exp(rep) - utilidade
        elif fun_util == 'exp1':
            ganho = math.exp(rep-1) - utilidade
        elif fun_util == 'exp2':
            ganho = math.exp(rep+1) - utilidade

        elif fun_util=='exp_1':
            ganho = -math.exp(-rep/9105) - utilidade
        elif fun_util == 'exp_2':
            ganho = -2*math.exp(-2*rep) - utilidade
        elif fun_util == 'exp_4':
            ganho = -4 * math.exp(-4 * rep) - utilidade

        else:
            print('Erro: Função Inválida!')
        return ganho

    def particoes(self,dataset, k):
        #Funçaõ que retorna uma lista com todas as partições possíveis do 'dataset'
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

    def brute_force_inc(self,dataset,k,fun_util):
        #'k' é o parâmetro relativo ao número de partições que se pretende
        rep=self.rep_pontual(dataset) #representante da base de dados
        #Cálculo da utilidade de todas as partições
        #Ciclo que nos dá todas as partições do dataset
        U, P=self.utilidade(self.particoes(dataset, k),fun_util)
        G=[] #lista com o ganho de cada partição
        for i in range(len(U)):
            if fun_util=='quad':
                ganho=(rep*rep)-U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'cub':
                ganho = (rep * rep * rep) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'ent':
                ganho = (rep * math.log(rep, 2)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'pol':
                ganho = (rep * (rep - 1)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'pol1':
                ganho = (rep * (rep + 1)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'pol2':
                ganho = (rep * (rep - 10)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'pol3':
                ganho = (rep * (rep + 10)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util=='exp':
                ganho = math.exp(rep) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util=='exp1':
                ganho = math.exp(rep-1) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util=='exp2':
                ganho = math.exp(rep+1) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'raiz':
                ganho = (math.sqrt(rep)) - U[i]
                G.append(ganho)
                opt_g = max(G)
                part_opt = P[G.index(max(G))]
                opt_u = U[G.index(max(G))]

            elif fun_util == 'exp_1':
                ganho = -math.exp(-rep/9105) - U[i]
                G.append(ganho)
                opt_g = max(G)
                part_opt = P[G.index(max(G))]
                opt_u = U[G.index(max(G))]
            elif fun_util == 'exp_2':
                ganho = -2 * math.exp(-2 * rep) - U[i]
                G.append(ganho)
                opt_g = max(G)
                part_opt = P[G.index(max(G))]
                opt_u = U[G.index(max(G))]
            elif fun_util == 'exp_4':
                ganho = -4 * math.exp(-4 * rep) - U[i]
                G.append(ganho)
                opt_g = max(G)
                part_opt = P[G.index(max(G))]
                opt_u = U[G.index(max(G))]

            else:
                print('Erro: Função Inválida!')
        return opt_g,part_opt,opt_u

    def all_part(self,dataset,k):
        perm = permutations(dataset)
        P=[]
        for i in list(perm):
            p = self.particoes(list(i), k)
            P.append(p)
        return P[0]

    def brute_force_vdd(self,dataset,k,fun_util):
        #'k' é o parâmetro relativo ao número de partições que se pretende
        rep=self.rep_pontual(dataset) #representante da base de dados
        #Cálculo da utilidade de todas as partições
        #Ciclo que nos dá todas as partições do dataset
        U, P=self.utilidade(self.all_part(dataset, k),fun_util)
        G=[] #lista com o ganho de cada partição
        for i in range(len(U)):
            if fun_util=='quad':
                ganho=(rep*rep)-U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'cub':
                ganho = (rep * rep * rep) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'ent':
                ganho = (rep * math.log(rep, 2)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'pol':
                ganho = (rep * (rep - 1)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'pol1':
                ganho = (rep * (rep + 1)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'pol2':
                ganho = (rep * (rep - 10)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'pol3':
                ganho = (rep * (rep + 10)) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util=='exp':
                ganho = math.exp(rep) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util=='exp1':
                ganho = math.exp(rep-1) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util=='exp2':
                ganho = math.exp(rep+1) - U[i]
                G.append(ganho)
                opt_g = min(G)
                part_opt = P[G.index(min(G))]
                opt_u = U[G.index(min(G))]
            elif fun_util == 'raiz':
                ganho = (math.sqrt(rep)) - U[i]
                G.append(ganho)
                opt_g = max(G)
                part_opt = P[G.index(max(G))]
                opt_u = U[G.index(max(G))]

            elif fun_util == 'exp_1':
                ganho = -math.exp(-rep/9105) - U[i]
                G.append(ganho)
                opt_g = max(G)
                part_opt = P[G.index(max(G))]
                opt_u = U[G.index(max(G))]
            elif fun_util == 'exp_2':
                ganho = -2 * math.exp(-2 * rep) - U[i]
                G.append(ganho)
                opt_g = max(G)
                part_opt = P[G.index(max(G))]
                opt_u = U[G.index(max(G))]
            elif fun_util == 'exp_4':
                ganho = -4 * math.exp(-4 * rep) - U[i]
                G.append(ganho)
                opt_g = max(G)
                part_opt = P[G.index(max(G))]
                opt_u = U[G.index(max(G))]

            else:
                print('Erro: Função Inválida!')
        return opt_g,part_opt,opt_u

    def ro(self,part,fun_util):
        rep=self.rep_pontual(part)
        ro=0
        for i in part:
            if fun_util=='quad':
                ro+=(i[0]*i[0]-(rep*rep))
            elif fun_util=='cub':
                ro += (i[0] * i[0]*i[0] - (rep * rep*rep))
            elif fun_util=='raiz':
                ro+=(math.sqrt(i[0])-math.sqrt(rep))

            elif fun_util == 'exp_1':
                ro += (-math.exp(-i[0]/9105) - (-math.exp(-rep/9105)))
            elif fun_util == 'exp_2':
                ro += (-2 * math.exp(-2 * i[0]) - (-2 * math.exp(-2 * rep)))
            elif fun_util == 'exp_4':
                ro += (-4 * math.exp(-4 * i[0]) - (-4 * math.exp(-4 * rep)))

        ro=ro/self.tot_subconjunto(part)
        return ro

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

    def u_Ck(self,Ck,fun_util):
        #utilidade de um conjunto Ck
        rep=self.rep_pontual(Ck)
        if fun_util=='quad':
            util=rep*rep
        elif fun_util=='cub':
            util=rep*rep*rep
        elif fun_util=='raiz':
            util=math.sqrt(rep)
        elif fun_util=='ent':
            util=rep * math.log(rep, 2)
        elif fun_util=='pol':
            util = rep*(rep-1)
        elif fun_util=='pol1':
            util = rep*(rep+1)

        elif fun_util=='exp_1':
            util = -math.exp(-rep/9105)

        return util

    def find_max_gain(self,Ck,fun_util):
        val = -1
        P = self.particoes(Ck, 2)
        part_opt=[]
        for i in range(len(P)):
            Ck_menos=P[i][0]
            Ck_mais = P[i][1]
            # Calcular o ganho da partição binária
            valk = np.abs(self.u_Ck(Ck,fun_util)-
                          np.divide(1,self.tot_subconjunto(Ck))*
                          (self.u_Ck(Ck_menos,fun_util)*self.tot_subconjunto(Ck_menos)+
                           self.u_Ck(Ck_mais,fun_util)*self.tot_subconjunto(Ck_mais)))
            if valk>val:
                val=valk
                menos=Ck_menos
                mais=Ck_mais
        part_opt.append(menos)
        part_opt.append(mais)
        return val, part_opt

    def cart(self, dataset, fun_util, k):
        return self.cart_aux(dataset, fun_util, k)

    def cart_aux(self,dataset, fun_util, k):
        if k == 1:
            dataset.sort()
            return dataset
        elif k > 1:
            valmax=0
            for j in range(len(dataset)):
                if len(dataset[j])==1:
                    val=0
                else:
                    val, part_opt=self.find_max_gain(dataset[j],fun_util)
                    print('val:', val)
                    if val>valmax:
                        valmax=val
                        partk=part_opt
                        kmax=j
            dataset.pop(kmax)
            for i in partk:
                dataset.append(i)
            dataset.sort()
            return self.cart_aux(dataset, fun_util, k - 1)
