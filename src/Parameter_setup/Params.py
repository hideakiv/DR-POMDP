import numpy as np

"""
Params: from Parser, converts Transition-Observation probability to joint probability
"""
class Params:
    def __init__(self,name,parser):
        self.name = name
        self.S = parser.states
        self.Z = parser.observations
        self.A = parser.actions

        self.nS = len(parser.states)
        self.nZ = len(parser.observations)
        self.nA = len(parser.actions)

        #only focus on transition-observation probability
        self.P = np.zeros((self.nA,self.nS,self.nS,self.nZ))
        for a in range(self.nA):
            for i in range(self.nS):
                for j in range(self.nS):
                    for z in range(self.nZ):
                        self.P[a,i,j,z] = parser.T[a,i,j]*parser.O[a,j,z]
                        #print('%d,%d,%d,%d: %f' % (a,i,j,z,P[a,i,j,z]))

        
        self.R = parser.R

        self.beta = parser.discount
