import numpy as np
import scipy.optimize
import scipy.linalg
from . import Params
from . import ParamsSample
from HSVI import UpperBoundLP as UBLP
from HSVI import Elimination

"""
Parameters for Standard POMDP
"""


class Params_Standard(Params.Params):
    def __init__(self,name,parser,folder=None):
        Params.Params.__init__(self,name,parser)

        self.eP = np.zeros((self.nA,self.nS,self.nS,self.nZ))
        self.eR = np.zeros((self.nA,self.nS))

        if folder is None:
            self.eP = self.P
            for a in range(self.nA):
                for i in range(self.nS):
                    for j in range(self.nS):
                        for z in range(self.nZ):
                            self.eR[a,i] += parser.R[a,i,j,z]*self.P[a,i,j,z]
        else:
            for a in range(self.nA):
                for i in range(self.nS):
                    Pai = ParamsSample.readSample(folder,self,a,i)
                    if Pai is None:
                        self.eP[a,i,:,:] = self.P[a,i,:,:]
                    else:
                        self.eP[a,i,:,:] = np.average(Pai,axis=0)

                    for j in range(self.nS):
                        for z in range(self.nZ):
                            self.eR[a,i] += parser.R[a,i,j,z]*self.eP[a,i,j,z]

        self.LBcount = 0
        self.LBAction = {}
        self.LB = self.initializeLB()
        self.LBthresh = 10
        self.UBcount = 0
        self.UB = self.initializeUB(parser)
        self.UBthresh = self.nS+10

    def initializeLB(self):
        rlb = -np.inf
        for a in range(self.nA):
            ra = np.inf
            for i in range(self.nS):
                if self.eR[a,i] < ra:
                    ra = self.eR[a,i]
            ra = ra / (1-self.beta)
            if ra > rlb:
                rlb = ra
        LB = np.ones(self.nS) * rlb
        self.LBcount += 1
        self.LBAction[0] = np.random.choice(self.nA)
        return {0:LB}

    def initializeUB(self,parser):
        c = np.ones(self.nS)
        A = np.zeros((self.nA*self.nS,self.nS))
        b = np.zeros(self.nA*self.nS)
        for a in range(self.nA):
            for i in range(self.nS):
                for j in range(self.nS):
                    A[a*self.nS+i,j] = self.beta * parser.T[a,i,j]
                A[a*self.nS+i,i] += -1.0
                b[a*self.nS+i] = -self.eR[a,i]
        res = scipy.optimize.linprog(c,A_ub=A,b_ub=b)
        v = res.x
        ub = []
        for i in range(self.nS):
            belief = np.zeros(self.nS)
            belief[i] = 1.0
            ub.append((belief,v[i]))
            self.UBcount += 1
        return UBLP.UpperBoundLP(self.nS,UB=ub)

    def getLB(self,b):
        v = -np.inf
        for ii,lbg in self.LB.items():
            aa = lbg @ b
            if aa > v:
                v = aa
        return v

    def getUB(self,b):
        return self.UB.getV(b)

    def addLB(self,lb,a):
        self.LB[self.LBcount] = lb
        self.LBAction[self.LBcount] = a
        self.LBcount += 1

    def addUB(self,b,v):
        self.UB.addPoint(b,v)
        self.UBcount += 1

    def popLB(self):
        if len(self.LB) > self.LBthresh * 1.1:
            popidx = Elimination.filter(self.LB,self.nS)
            for idx in popidx:
                del self.LB[idx]
                del self.LBAction[idx]
            self.LBthresh = len(self.LB)

    def popUB(self):
        if self.UB.getNum() > self.UBthresh * 1.1:
            popidx = Elimination.delPoints(self.UB,self.nS)
            for idx in popidx:
                self.UB.popConstr(idx)
            self.UBthresh = self.UB.getNum()

    def getLBAction(self,b):
        v = -np.inf
        idx = None
        for ii,lbg in self.LB.items():
            aa = lbg @ b
            if aa > v:
                v = aa
                idx = ii
        return self.LBAction[idx]


    def save(self,folder):
        filename = folder + 'pprmUB.lp'
        self.UB.save(filename)
        del self.UB

    def load(self,folder):
        filename = folder + 'pprmUB.lp'
        self.UB = UBLP.UpperBoundLP(self.nS,filename=filename,count=self.UBcount)
