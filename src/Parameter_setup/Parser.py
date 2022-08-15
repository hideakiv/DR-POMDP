#Parse POMDP file
import numpy as np
import re


class Parser:
    def __init__(self):
        self.discount = 1.0         # discount factor
        self.values = 1.0           # reward maximization->1.0/ cost minimization->-1.0
        self.states = []            # set of states
        self.actions = []           # set of actions
        self.observations = []      # set of observations

        self.start = []             # initial probability

        self.T = []                 # Transition probability
        self.O = []                 # Observation probability
        self.R = []                 # Reward/cost

    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def parse(self,filename):
        N = 0 #number of states
        M = 0 #number of observations
        L = 0 #number of actions
        with open(filename,'r') as f:
            lines = f.readlines()
        l = 0
        lmax = len(lines)
        preambledone = 0

        while l < lmax:
            row = lines[l].strip()
            foo = list(filter(None,re.split(" |: ",row)))
            if len(foo)>0:
                foo0 = foo[0]
            else:
                foo0 = None

            if foo0 == 'discount':
                self.discount = float(foo[1])

            elif foo0 == 'values':
                if foo[1] == 'reward':
                    self.values = 1.0
                elif foo[1] == 'cost':
                    self.values = -1.0

            elif foo0 == 'states':
                if foo[1].isdigit():
                    for i in range(int(foo[1])):
                        self.states.append(str(i))
                else:
                    for s in foo[1:]:
                        self.states.append(s)
                preambledone += 1

            elif foo0 == 'actions':
                if foo[1].isdigit():
                    for i in range(int(foo[1])):
                        self.actions.append(str(i))
                else:
                    for s in foo[1:]:
                        self.actions.append(s)
                preambledone += 1

            elif foo0 == 'observations':
                if foo[1].isdigit():
                    for i in range(int(foo[1])):
                        self.observations.append(str(i))
                else:
                    for s in foo[1:]:
                        self.observations.append(s)
                preambledone += 1

            elif foo0 == 'start':
                self.start = np.zeros(N)
                if self.is_number(foo[1]) and len(foo) == N+1:
                    self.start = np.asarray([float(foo[i+1]) for i in range(N)])
                else:
                    if foo[1] == 'uniform':
                        self.start = np.asarray([1/N for i in range(N)])
                    elif foo[1] == 'include':#correctbelow
                        ss = foo[2:]
                        m = len(ss)
                        for s in ss:
                            i = self.states.index(s)
                            self.start[i] = 1/m
                    elif foo[1] == 'exclude':
                        ss = foo[2:]
                        m = len(ss)
                        for i in range(N):
                            s = self.states[i]
                            if s not in ss:
                                self.start[i] = 1/(N-m)
                    else:
                        i = self.states.index(foo[1])
                        self.start[i] = 1

            elif foo0 == 'T':
                h = len(foo)
                if foo[1] == '*':
                    aa = range(L)
                else:
                    aa = [self.actions.index(foo[1])]

                if h == 5:
                    if foo[2] == '*':
                        ii = range(N)
                    else:
                        ii = [self.states.index(foo[2])]
                    if foo[3] == '*':
                        jj = range(N)
                    else:
                        jj = [self.states.index(foo[3])]
                    for a in aa:
                        for i in ii:
                            for j in jj:
                                self.T[a,i,j] = float(foo[4])
                elif h == 3:
                    if foo[2] == '*':
                        ii = range(N)
                    else:
                        ii = [self.states.index(foo[2])]
                    l += 1
                    row = lines[l]
                    bar = row.strip().split()
                    for j in range(N):
                        for a in aa:
                            for i in ii:
                                self.T[a,i,j]=float(bar[j])
                elif h == 2:
                    for i in range(N):
                        l +=1
                        row = lines[l]
                        bar = row.strip().split()
                        for j in range(N):
                            for a in aa:
                                self.T[a,i,j]=float(bar[j])

            elif foo0 == 'O':
                h = len(foo)
                if foo[1] == '*':
                    aa = range(L)
                else:
                    aa = [self.actions.index(foo[1])]

                if h == 5:
                    if foo[2] == '*':
                        jj = range(N)
                    else:
                        jj = [self.states.index(foo[2])]
                    if foo[3] == '*':
                        kk = range(M)
                    else:
                        kk = [self.observations.index(foo[3])]
                    for a in aa:
                        for j in jj:
                            for k in kk:
                                self.O[a,j,k] = float(foo[4])
                elif h == 3:
                    if foo[2] == '*':
                        jj = range(N)
                    else:
                        jj = [self.states.index(foo[2])]
                    l += 1
                    row = lines[l]
                    bar = row.strip().split()
                    for k in range(M):
                        for a in aa:
                            for j in jj:
                                self.O[a,j,k]=float(bar[k])
                elif h == 2:
                    for j in range(N):
                        l = l+1
                        row = lines[l]
                        bar = row.strip().split()
                        for k in range(M):
                            for a in aa:
                                self.O[a,j,k]=float(bar[k])
            elif foo0 == 'R':
                h=len(foo)
                if foo[1] == '*':
                    aa = range(L)
                else:
                    aa = [self.actions.index(foo[1])]

                if foo[2] == '*':
                    ii = range(N)
                else:
                    ii = [self.states.index(foo[2])]
                if h == 6:
                    if foo[3] == '*':
                        jj = range(N)
                    else:
                        jj = [self.states.index(foo[3])]
                    if foo[4] == '*':
                        kk = range(M)
                    else:
                        kk = [self.observations.index(foo[4])]
                    for a in aa:
                        for i in ii:
                            for j in jj:
                                for k in kk:
                                    self.R[a,i,j,k] = self.values*float(foo[5])
                if h == 4:
                    if foo[3] == '*':
                        jj = range(N)
                    else:
                        jj = [self.states.index(foo[3])]
                    l = l+1
                    row = lines[l]
                    bar = row.strip().split()
                    for k in range(M):
                        for a in aa:
                            for i in ii:
                                for j in jj:
                                    self.R[a,i,j,k] = self.values*float(bar[k])
                if h==3:
                    for j in range(N):
                        l = l+1
                        row = lines[l]
                        bar = row.strip().split()
                        for k in range(M):
                            for a in aa:
                                for i in ii:
                                    self.R[a,i,j,k] = self.values*float(bar[k])

            else:
                pass


            l += 1

            if preambledone == 3:
                N = len(self.states)
                M = len(self.observations)
                L = len(self.actions)

                self.T = np.zeros((L,N,N))
                self.O = np.zeros((L,N,M))
                self.R = np.zeros((L,N,N,M))
                preambledone = 0

if __name__ == '__main__':
    model = Parser()
    model.parse('../../Models/ejs/ejs2.POMDP')
