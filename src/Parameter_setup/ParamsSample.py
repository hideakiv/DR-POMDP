import numpy as np
from Sampling import Sample

"""
randomly samples from a given param and then saves to a folder
-stdevs: standard deviation dim nA * nS
-nsamples: dim nA * nS
"""

def normalSample(folder,params,stdevs,nsamples):
    #sample probabilities
    for a in range(params.nA):
        for i in range(params.nS):
            fileP = folder + "/Psamples_" + str(a) + "_" + str(i) + ".csv"
            #fileR = folder + "/Rsamples_" + str(a) + "_" + str(i) + ".csv"

            #if stdev is 0, do nothing
            if stdevs[a][i] == 0 or nsamples[a][i] == 0:
                #np.savetxt(fileP, params.P[a,i,:,:], delimiter=",")
                #np.savetxt(fileR, params.R[a,i], delimiter=",")
                pass
            else:
                n = nsamples[a][i]
                center = np.reshape(params.P[a,i,:,:], params.nS * params.nZ)
                retP = Sample.sampleNormalOnSimplex(params.nS*params.nZ,center,stdevs[a][i],n)
                np.savetxt(fileP, retP, delimiter=",")
                #np.savetxt(fileP, retP.reshape(n,params.nA,params.nS), delimiter=",")
                #np.savetxt(fileR, params.R[a,i], delimiter=",")

def readSample(folder,params,a,i):
    fileP = folder + "/Psamples_" + str(a) + "_" + str(i) + ".csv"
    try:
        Pai = np.loadtxt(fileP, delimiter=",")
        n,nn = Pai.shape
        Pai=Pai.reshape((n,params.nS,params.nZ))
    except:
        Pai = None

    return Pai

if __name__ == '__main__':
    model = Parser.Parser()
    model.parse('../../Models/ejs/ejs2.POMDP')
    params = Params.Params("ejs2",model)

    folder = '../Experiments/test'
    stdevs = [[0.3,0.2],[0,0.1]]
    nsamples = [[4,4],[4,4]]
    ParamsSample.normalSample(folder,params,stdevs,nsamples)
