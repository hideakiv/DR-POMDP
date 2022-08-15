import numpy as np
from Parameter_setup import Params_DR
from Sampling import Sample as us
from HSVI import HSVI

"""
environments - helper for identifying the policy's assumption of P and R
"""

class Env():
    def __init__(self,prm):
        self.prm = prm

    def getPaRa(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

class RegularEnv(Env):
    def __init__(self,prm,epsilon):
        Env.__init__(self,prm)
        self.epsilon = epsilon
        rmax = -np.inf
        rmin = -np.inf
        for a in range(self.prm.nA):
            rmina =  np.inf
            for i in range(self.prm.nS):
                for j in range(self.prm.nS):
                    for z in range(self.prm.nZ):
                        if self.prm.R[a,i,j,z] > rmax:
                            rmax = self.prm.R[a,i,j,z]
                        if self.prm.R[a,i,j,z] < rmina:
                            rmina = self.prm.R[a,i,j,z]
            if rmina > rmin:
                rmin = rmina
        self.thresh = (rmax-rmin)/(1-self.prm.beta)

    def getPaRa(self,b,a):
        Pa = self.prm.P[a,:,:,:]
        Ra = self.prm.R[a,:,:,:]
        return Pa, Pa, Ra

    def done(self,discount):
        if self.thresh*discount < self.epsilon:
            return True
        else:
            return False

class RegularEnv2(Env):
    def __init__(self,prm,epsilon):
        Env.__init__(self,prm)
        self.epsilon = epsilon
        rmax = -np.inf
        rmin = -np.inf
        for a in range(self.prm.nA):
            rmina =  np.inf
            for i in range(self.prm.nS):
                for j in range(self.prm.nS):
                    for z in range(self.prm.nZ):
                        if self.prm.R[a,i,j,z] > rmax:
                            rmax = self.prm.R[a,i,j,z]
                        if self.prm.R[a,i,j,z] < rmina:
                            rmina = self.prm.R[a,i,j,z]
            if rmina > rmin:
                rmin = rmina
        self.thresh = (rmax-rmin)/(1-self.prm.beta)

    def getPaRa(self,b,a):
        Pa = self.prm.eP[a,:,:,:]####<-this is the difference
        Ra = self.prm.R[a,:,:,:]
        return Pa, Pa, Ra

    def done(self,discount):
        if self.thresh*discount < self.epsilon:
            return True
        else:
            return False

class WorstCaseEnv(Env):
    def __init__(self,prm,epsilon):
        assert isinstance(prm,Params_DR.Params_DR)
        Env.__init__(self,prm)
        self.epsilon = epsilon
        rmax = -np.inf
        rmin = -np.inf
        for a in range(self.prm.nA):
            rmina =  np.inf
            for i in range(self.prm.nS):
                for j in range(self.prm.nS):
                    for z in range(self.prm.nZ):
                        if self.prm.R[a,i,j,z] > rmax:
                            rmax = self.prm.R[a,i,j,z]
                        if self.prm.R[a,i,j,z] < rmina:
                            rmina = self.prm.R[a,i,j,z]
            if rmina > rmin:
                rmin = rmina
        self.thresh = (rmax-rmin)/(1-self.prm.beta)

    def getPaRa(self,b,a):
        HSVI.HSVI(b,self.epsilon,self.prm)
        Pa = self.prm.LBprob[a].getdist(b,self.prm)
        Ra = self.prm.R[a,:,:,:]
        return Pa, Pa, Ra

    def done(self,discount):
        if self.thresh*discount < self.epsilon:
            return True
        else:
            return False

class WorstCaseWithErrorEnv(Env):
    def __init__(self,prm,epsilon,radius):
        assert isinstance(prm, Params_DR.Params_DR)
        Env.__init__(self,prm)
        self.epsilon = epsilon
        self.radius = radius
        rmax = -np.inf
        rmin = -np.inf
        for a in range(self.prm.nA):
            rmina =  np.inf
            for i in range(self.prm.nS):
                for j in range(self.prm.nS):
                    for z in range(self.prm.nZ):
                        if self.prm.R[a,i,j,z] > rmax:
                            rmax = self.prm.R[a,i,j,z]
                        if self.prm.R[a,i,j,z] < rmina:
                            rmina = self.prm.R[a,i,j,z]
            if rmina > rmin:
                rmin = rmina
        self.thresh = (rmax-rmin)/(1-self.prm.beta)

    def getPaRa(self,b,a):
        HSVI.HSVI(b,self.epsilon,self.prm)
        Pa = self.prm.LBprob[a].getdist(b,self.prm)
        Ra = self.prm.R[a,:,:,:]
        Pap = np.zeros((self.prm.nS,self.prm.nS,self.prm.nZ))
        for i in range(self.prm.nS):
            prob = []
            for j in range(self.prm.nS):
                for z in range(self.prm.nZ):
                    prob.append(Pa[i,j,z])
            foo = us.sampleP(prob,self.radius,self.prm.q[a][i])
            for j in range(self.prm.nS):
                for z in range(self.prm.nZ):
                    Pap[i,j,z] = foo[j*self.prm.nZ+z]
        return Pa, Pap, Ra

    def done(self,discount):
        if self.thresh*discount < self.epsilon:
            return True
        else:
            return False
