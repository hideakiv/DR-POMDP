import numpy as np
from . import Params
from . import ParamsSample
import scipy.optimize
import scipy.linalg
from HSVI import UpperBoundLP as UBLP
from HSVI import Elimination
from HSVI import MomentLPLB
from HSVI import MomentLPUB
from HSVI import MomentLPUBinit
from Sampling import Normalization

"""
Parameters for Moment-based DR-POMDP
"""

class Params_DR(Params.Params):
    def __init__(self,name,parser,folder=None,pbar=None,cap=None,Aeq=None,beq=None,Robust=False,scale=1):
        Params.Params.__init__(self,name,parser)
        self.pbar = np.zeros((self.nA,self.nS,self.nS*self.nZ))
        self.R = parser.R
        self.cap = np.zeros((self.nA,self.nS,self.nS*self.nZ))

        if folder is None:
            if pbar is None:
                for a in range(self.nA):
                    for i in range(self.nS):
                        for j in range(self.nS):
                            for z in range(self.nZ):
                                self.pbar[a,i,j*self.nZ+z] = self.P[a,i,j,z]
            else:
                self.pbar = pbar
            if cap is None:
                self.cap = np.ones((self.nA,self.nS,self.nS*self.nZ)) * 0.05
            else:
                self.cap = cap
        else:
            for a in range(self.nA):
                for i in range(self.nS):
                    Pai = ParamsSample.readSample(folder,self,a,i)
                    if Pai is None:
                        for j in range(self.nS):
                            for z in range(self.nZ):
                                self.pbar[a,i,j*self.nZ+z] = self.P[a,i,j,z]
                        self.cap[a,i,:] = np.zeros(self.nS*self.nZ)
                    else:
                        for j in range(self.nS):
                            for z in range(self.nZ):
                                self.pbar[a,i,j*self.nZ+z] = np.average(Pai[:,j,z])
                        #How do we set the mad value?
                        if Robust:
                            Maxdev = maxdev(Pai,self.pbar[a,i,:],self.nS,self.nZ)
                            for j in range(self.nS):
                                for z in range(self.nZ):
                                    self.cap[a,i,j*self.nZ+z] = scale * Maxdev
                        else:
                            MAD = mad(Pai,self.pbar[a,i,:],self.nS,self.nZ)
                            for j in range(self.nS):
                                for z in range(self.nZ):
                                    self.cap[a,i,j*self.nZ+z] = scale * MAD


        self.Aeq = [[np.ones((1,self.nS*self.nZ)) for i in range(self.nS)] for a in range(self.nA)]
        self.beq = [[np.ones(1) for i in range(self.nS)] for a in range(self.nA)]
        if Aeq is not None and beq is not None:
            for a in range(self.nA):
                for i in range(self.nS):
                    if Aeq[a][i] is not None and beq[a][i] is not None:
                        self.Aeq[a][i] = np.vstack((self.Aeq[a][i],Aeq[a][i]))
                        self.beq[a][i] = np.vstack((self.beq[a][i],beq[a][i]))
        self.q = [[ [] for i in range(self.nS)] for a in range(self.nA)]#base for equality constraint
        for a in range(self.nA):
            for i in range(self.nS):
                self.q[a][i] = Normalization.getbase(self.Aeq[a][i],self.beq[a][i])

        self.LBprob = [MomentLPLB.MomentLPLB(a,self) for a in range(self.nA)]
        self.LBcount = 0
        self.LBAction = {}
        self.LB = self.initializeLB()
        self.LBthresh = 10

        self.UBprob = [MomentLPUB.MomentLPUB(a,self) for a in range(self.nA)]
        self.UBcount = 0
        self.UB = self.initializeUB()
        self.UBthresh = self.nS+10

    def initializeLB(self):
        rlb = -np.inf
        for a in range(self.nA):
            ra = self.LBprob[a].initializeLB(self)
            minra = np.min(ra) / (1-self.beta)
            if minra > rlb:
                rlb = minra
        LB = np.ones(self.nS) * rlb
        for a in range(self.nA):
            self.LBprob[a].addgrad(self)
        self.LBcount += 1
        self.LBAction[0] = np.random.choice(self.nA)
        return {0: LB}

    def initializeUB(self):
        foo = MomentLPUBinit.MomentLPUBinit(self)
        v= foo.solve(self)
        ub = []
        for i in range(self.nS):
            belief = np.zeros(self.nS)
            belief[i] = 1.0
            ub.append((belief,v[i]))
            for a in range(self.nA):
                self.UBprob[a].addPoint(belief,v[i],self)
            self.UBcount += 1
        return UBLP.UpperBoundLP(self.nS,ub)

    def getLB(self,b):
        v = -np.inf
        for ii, lb in self.LB.items():
            aa = lb @ b
            if aa > v:
                v = aa
        return v

    def getUB(self,b):
        return self.UB.getV(b)

    def addLB(self,lb,ac):
        self.LB[self.LBcount] = lb
        for a in range(self.nA):
            self.LBprob[a].addgrad(self)
        self.LBAction[self.LBcount] = ac
        self.LBcount += 1

    def addUB(self,b,v):
        self.UB.addPoint(b,v)
        for a in range(self.nA):
            self.UBprob[a].addPoint(b,v,self)
        self.UBcount += 1

    def popLB(self):
        if len(self.LB) > self.LBthresh * 1.1:
            popidx = Elimination.filter(self.LB,self.nS)
            for idx in popidx:
                del self.LB[idx]
                del self.LBAction[idx]
                for a in range(self.nA):
                    self.LBprob[a].popConstr(idx,self.nZ)
            self.LBthresh = len(self.LB)

    def popUB(self):
        if self.UB.getNum() > self.UBthresh * 1.1:
            popidx = Elimination.delPoints(self.UB,self.nS)
            for idx in popidx:
                self.UB.popConstr(idx)
                for a in range(self.nA):
                    self.UBprob[a].popConstr(idx,self.nZ)
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
        filename = folder + 'mprmUB.lp'
        self.UB.save(filename)
        del self.UB

        for a in range(self.nA):
            filename = folder + 'mprmLBprob_'+str(a)+'.lp'
            self.LBprob[a].save(filename)
        del self.LBprob

        for a in range(self.nA):
            filename = folder + 'mprmUBprob_'+str(a)+'.lp'
            self.UBprob[a].save(filename)
        del self.UBprob

    def load(self,folder):
        filename = folder + 'mprmUB.lp'
        self.UB = UBLP.UpperBoundLP(self.nS,filename=filename,count=self.UBcount)

        self.LBprob = []
        for a in range(self.nA):
            filename = folder + 'mprmLBprob_'+str(a)+'.lp'
            self.LBprob.append(MomentLPLB.MomentLPLB(a,self,filename=filename,count=self.LBcount))

        self.UBprob = []
        for a in range(self.nA):
            filename = folder + 'mprmUBprob_'+str(a)+'.lp'
            self.UBprob.append(MomentLPUB.MomentLPUB(a,self,filename=filename,count=self.UBcount))


def mad(Pai,meanval,nS,nZ):
    n = Pai.shape[0]
    ret = 0
    for i in range(n):
        absval = 0
        for j in range(nS):
            for z in range(nZ):
                absval += np.abs(Pai[i,j,z] - meanval[j*nZ+z])

        ret += absval/n

    return ret

def maxdev(Pai,meanval,nS,nZ):
    n = Pai.shape[0]
    ret = 0
    for i in range(n):
        absval = 0
        for j in range(nS):
            for z in range(nZ):
                absval += np.abs(Pai[i,j,z] - meanval[j*nZ+z])
        ret = max(absval,ret)

    return ret
