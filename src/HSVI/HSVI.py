import numpy as np
from Parameter_setup import Params_Standard, Params_DR
import time
import csv

"""
Heursitic Search Value Iteration algorithm
"""

class TimeKeeper():
    def __init__(self,b0,folder,name):
        self.time_start = time.clock()
        self.b0 = b0
        self.f = open(folder+name+'_time_gap.csv', 'w')

        self.writer = csv.writer(self.f, lineterminator='\n')
        self.limit = 3600

    def write_data(self,prm):
        UB = prm.getUB(self.b0)
        LB = prm.getLB(self.b0)
        foo = [self.getTime(),UB,LB]
        self.writer.writerow(foo)

    def getTime(self):
        return time.clock()-self.time_start

def HSVI(b0,epsilon,prm,folder=None):
    if folder is not None:
        keeper = TimeKeeper(b0,folder,prm.name)
    else:
        keeper = None
    while getGap(b0,0,epsilon,prm) > 0:
        BoundUncertaintyExplore(b0,0,epsilon,prm,keeper)

        #prm.save(folder)
        #prm.load(folder)
        #pdb.set_trace()
        
        print('')
        if keeper is not None and keeper.getTime()>keeper.limit:
            break
    if keeper is not None:
        keeper.f.close()

def BoundUncertaintyExplore(b,t,epsilon,prm,keeper,g=None):
    maxa, Pa = getMax_a(b,prm)
    maxz, Gap = getMax_z(b,Pa,t,epsilon,prm)
    if Gap > 0:
        if t % 50 == 0:
            print('depth:',t+1,'Gap:',Gap)
        BoundUncertaintyExplore(updateb(b,Pa,maxz),t+1,epsilon,prm,keeper,g=Gap)
    #updateLB
    backup(b,prm)
    #updateUB
    forward(b,prm)

    prm.popLB()
    prm.popUB()

    if keeper is not None:
        keeper.write_data(prm)

def forward(b,prm):
    maxa = -1
    maxQ = -np.inf
    Qs = np.zeros(prm.nA)
    for a in range(prm.nA):
        Q = getQ_UB(b,a,prm)
        Qs[a] = Q
        if Q > maxQ:
            maxQ = Q
            maxa = a
    prm.addUB(b,maxQ)

def backup(b,prm):
    maxa = -1
    lb = -np.inf
    g = None
    for a in range(prm.nA):
        if isinstance(prm,Params_Standard.Params_Standard):
            ga = np.copy(prm.eR[a,:])
            for z in range(prm.nZ):
                maxval = -np.inf
                maxgrad = None
                newb = updateb(b,prm.eP[a,:,:,:],z)
                if newb is None:
                    continue
                for ii,lbg in prm.LB.items():

                    val = lbg @ newb
                    if val > maxval:
                        maxval = val
                        maxgrad = lbg
                temp = np.zeros(prm.nS)
                for i in range(prm.nS):
                    for j in range(prm.nS):
                        temp[i] += prm.beta * prm.eP[a,i,j,z]*maxgrad[j]
                ga += temp
            val = ga @ b
        elif isinstance(prm,Params_DR.Params_DR):
            ga = prm.LBprob[a].backup(b,prm)
            val = ga @ b

        if val > lb:
            lb = val
            g = ga
            maxa = a
    prm.addLB(g,maxa)

def getMax_a(b,prm):
    maxa = -1
    maxQ = -np.inf
    for a in range(prm.nA):
        Q = getQ_UB(b,a,prm)
        if Q > maxQ:
            maxQ = Q
            maxa = a
    if isinstance(prm,Params_Standard.Params_Standard):
        Pa = prm.eP[maxa,:,:,:]
    elif isinstance(prm,Params_DR.Params_DR):
        Pa = prm.LBprob[maxa].getdist(b,prm)
    return maxa, Pa

def getQ_UB(b,a,prm):
    if isinstance(prm,Params_Standard.Params_Standard):
        Q = prm.eR[a,:] @ b
        for z in range(prm.nZ):
            Pa = prm.eP[a,:,:,:]
            newb = updateb(b,Pa,z)
            if newb is None:
                continue
            Q += prm.beta * getPz(b,Pa,z) * prm.getUB(newb)
        return Q
    elif isinstance(prm,Params_DR.Params_DR):
        Q = prm.UBprob[a].solve(a,b,prm)
        return Q


def getPz(b,Pa,z):
    nS = len(b)
    Pz = 0
    for i in range(nS):
        for j in range(nS):
            Pz += Pa[i,j,z] * b[i]
    return Pz

def updateb(b,Pa,z):
    newb = np.zeros(len(b))
    for i in range(len(b)):
        newb += Pa[i,:,z] * b[i]
    if np.sum(newb)>1e-9:
        newb = newb / np.sum(newb)
        for i in range(len(b)):
            if newb[i]<1e-9:
                newb[i]=0
        return newb / np.sum(newb)
    else:
        return None

def getMax_z(b,Pa,t,epsilon,prm):
    maxz = -1
    maxPGap = -np.inf
    maxGap = None
    for z in range(prm.nZ):
        newb = updateb(b,Pa,z)
        if newb is None:
            continue
        Gap = getGap(newb,t+1,epsilon,prm)
        PGap = getPz(b,Pa,z) * Gap
        if PGap > maxPGap:
            maxPGap = PGap
            maxz = z
            maxGap = Gap
    return maxz, maxGap

def getGap(b,t,epsilon,prm):
    UB = prm.getUB(b)
    LB = prm.getLB(b)
    Gap = UB - LB - epsilon * prm.beta**(-t)
    return Gap
