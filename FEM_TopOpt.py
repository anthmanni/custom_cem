'''Topology Optimzation Code for Electromagnetism

Designs a 2D metalens with relative permittivity eps_r
capable of monochromatic focusing of TE-polarized light
at a point in space.

Equation solved: nabla dot (nabla Ez) + k^2 A Ez = F
with First Order absorbing boundary condition:
n dot (nabla Ez) = - i k Ez on boundaries
and incident plane wave propagating from bottom to top.

The equation is solved in a rectangular domain, discretized using
quadrilateral bi-linear finite elements.

Started Jan 2024 from MATLAB to Python, original author Rasmus E. Christansen, April 2021
Validated against MATLAB code on August 3, 2024.!'''
import numpy as np
import scipy.sparse
from scipy.signal import convolve2d as conv2
from numpy import tile, array,tanh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# from matspy import spy  # function for viewing sparse matrix sparsity patterns

''' QUICK HELPER FUNCTION TO MIMIC arr(:) IN MATLAB '''
def flat(arr):
    return arr.reshape((arr.shape[0]*arr.shape[1],1),order='F')

''' ELEMENT MATRICES '''
def element_matrices(scaling):
    # First order quadrilateral elements
    aa = scaling/2
    bb = scaling/2 # element size scaling
    k1 = (aa**2 + bb**2)/(aa*bb)
    k2 = (aa**2 - 2*bb**2)/(aa*bb)
    k3 = (bb**2 - 2*aa**2)/(aa*bb)
    LaplaceElementMatrix = array([[k1/3, k2/6, -k1/6, k3/6],
                                     [k2/6, k1/3, k3/6, -k1/6],
                                     [-k1/6,k3/6,k1/3,k2/6],
                                     [k3/6,-k1/6,k2/6,k1/3]])
    MassElementMatrix = aa*bb*array([[4/9,2/9,1/9,2/9],
                                        [2/9,4/9,2/9,1/9],
                                        [1/9,2/9,4/9,2/9],
                                        [2/9,1/9,2/9,4/9]])
    return LaplaceElementMatrix, MassElementMatrix

''' THRESHOLDING '''
def threshold(xIn, beta, eta):
    xOut = (tanh(beta*eta)+tanh(beta*(xIn-eta)))/(tanh(beta*eta)+tanh(beta*(1-eta)))
    return xOut
def derivative_of_threshold(xIn,beta,eta):
    xOut = (1-tanh(beta*(xIn-eta))**2)*beta/(tanh(beta*eta)+tanh(beta*(1-eta)))
    return xOut

''' DENSITY FILTER '''
def density_filter(filterKernel, filterScalingA, filterScalingB, x, func):
    xS = conv2(x*func/filterScalingA,filterKernel,mode='same')/filterScalingB
    return xS
def density_filter_setup(fR, nElx, nEly):
    [dy,dx] = np.meshgrid(np.arange(-np.ceil(fR)+1,np.ceil(fR)),np.arange(-np.ceil(fR)+1,np.ceil(fR)))
    Kernel = np.maximum(0,fR-np.sqrt(dx**2+dy**2)) # cone filter kernel
    Scaling = conv2(np.ones((nEly,nElx)),Kernel,'same') # filter scaling
    return Kernel, Scaling

''' MATERIAL PARAMETER INTERPOLATION '''
def material_interpolation(eps_r,x,alpha_i):
    A = 1+x*(eps_r-1)-1j*alpha_i*x*(1-x) # interpolation
    dAdx = (eps_r-1)*(1+0*x)-1j*alpha_i*(1-2*x) # derivative of interpolation
    return A, dAdx

''' DISCRETIZATION CLASS BECAUSE MATLAB USES STRUCTS '''
class discretization:
    def __init__(self, nelx, nely, telmidx, dvelmidx):
        self.nelx = nelx
        self.nely = nely
        self.tElmIdx = telmidx
        self.dvelmidx = dvelmidx

''' CONNECTIVITY AND INDEX SETS '''
def index_sets_sparse(dis):
    '''INDEX SETS FOR SYSTEM MATRIX
    NOTE: MATLAB RESHAPE DEFAULTS TO FORTRAN ORDERING (DOWN COLUMNS)
    NUMPY RESHAPE DEFAULTS TO C ORDERING (DOWN ROWS)'''
    nex = dis.nelx
    ney = dis.nely # extracting number of elements
    nodenrs = np.arange(1,(nex+1)*(ney+1)+1).reshape((1+ney,1+nex),order='F') # node numbers
    dis.nodenrs = nodenrs
    print('nodenrs: ',nodenrs[:5])
    print(nodenrs[-5:])
    dis.edofvec_ = nodenrs[:-1,:-1] + 1
    dis.edofvec = dis.edofvec_.reshape((nex*ney,1),order='F') # first dof in element, ORDER = 'F' IN RESHAPE ALWAYS
    print('edofvec: ',dis.edofvec[:5])
    print(dis.edofvec[-5:])
    dis.edofmat = array(tile(dis.edofvec,(1,4)))+array(tile([0,ney+1,ney,-1],(nex*ney,1)))
    print('edofmat shape:',dis.edofmat.shape)
    dis.iS = np.kron(dis.edofmat,np.ones((4,1))).T.reshape((16*nex*ney,1),order='F')
    dis.jS = np.kron(dis.edofmat,np.ones((1,4))).T.reshape((16*nex*ney,1),order='F')
    dis.idxDSdx = dis.edofmat.T.reshape((1,4*nex*ney),order='F')
    print(dis.edofmat[:5])
    print(dis.edofmat[-5:])
    '''INDEX SETS FOR BOUNDARY CONDITIONS'''
    TMP = tile(np.concatenate(([np.arange(1,ney+1)],[np.arange(2,ney+2)]),axis=0),(2,1))
    # dis.TMP1 = TMP
    dis.iB1 = TMP.reshape((4*ney,1),order='F') # row indices
    dis.jB1 = np.concatenate(([TMP[1]],[TMP[0]],[TMP[2]],[TMP[3]]),axis=0).reshape((4*ney,1),order='F') # column indices

    TMP = np.concatenate(([np.arange(1,(ney+1)*nex,ney+1)],[np.arange(ney+2,(ney+1)*(nex+1)+1,ney+1)]),axis=0)
    TMP = tile(TMP,(2,1))
    # dis.TMP2 = TMP
    dis.iB2 = TMP.reshape((4*nex,1),order='F')
    dis.jB2 = np.concatenate(([TMP[1]],[TMP[0]],[TMP[2]],[TMP[3]]),axis=0).reshape((4*nex,1),order='F')

    TMP = np.concatenate(([np.arange((ney+1)*nex+1,(ney+1)*(nex+1))],[np.arange((ney+1)*nex+2,(ney+1)*(nex+1)+1)]),axis=0)
    TMP = tile(TMP,(2,1))
    # dis.TMP3 = TMP
    dis.iB3 = TMP.reshape((4*ney,1),order='F')
    dis.jB3 = np.concatenate(([TMP[1]],[TMP[0]],[TMP[2]],[TMP[3]]),axis=0).reshape((4*ney,1),order='F')

    TMP = np.concatenate(([np.arange(2*(ney+1),(ney+1)*(nex+2),ney+1)],[np.arange((ney+1),(ney+1)*(nex+1),ney+1)]),axis=0)
    TMP = tile(TMP,(2,1))
    dis.iB4 = TMP.reshape((4*nex,1),order='F')
    dis.jB4 = np.concatenate(([TMP[1]],[TMP[0]],[TMP[2]],[TMP[3]]),axis=0).reshape((4*nex,1),order='F')
    dis.iRHS = TMP
    print('iRHS shape:',dis.iRHS.shape)

    '''INDEX SETS FOR INTEGRATION OF ALL ELEMENTS'''
    ima0 = tile([1,2,3,4]*4,(1,nex*ney)).T
    # ima0 = flat(ima0)
    jma0 = tile([1]*4+[2]*4+[3]*4+[4]*4,(1,nex*ney)).T
    # jma0 = flat(jma0)
    addTMP = tile(4*np.arange(nex*ney),(16,1))
    addTMP = flat(addTMP)
    dis.ielfull = ima0 + addTMP
    dis.jelfull = jma0 + addTMP
    print('ielfull, jelfull shapes:',dis.ielfull.shape,dis.jelfull.shape)

    '''INDEX SETS FOR SENSITIVITY COMPUTATIONS'''
    dis.ielsens = np.arange(1,4*nex*ney+1).T
    jelsens = tile(np.arange(1,nex*ney+1),(4,1))
    dis.jelsens = flat(jelsens)
    print('ielsens, jelsens shapes:',dis.ielsens.shape,dis.jelsens.shape)

''' ABSORBING BOUNDARY CONDITIONS AND RIGHT HAND SIDE '''
def boundary_conditions_rhs(waveVector, dis, scaling):
    AbsBCMatEdgeValues = 1j*waveVector*scaling*array([[1/6],[1/6],[1/3],[1/3]])
    ''' ALL BOUNDARIES HAVE ABSORBING BOUNDARY CONDITIONS'''
    dis.iBC = np.concatenate((dis.iB1,dis.iB2,dis.iB3,dis.iB4),axis=0)
    dis.jBC = np.concatenate((dis.jB1, dis.jB2, dis.jB3, dis.jB4), axis=0)
    dis.vBC = tile(AbsBCMatEdgeValues,(2*(dis.nelx+dis.nely),1))
    print('Boundary Condition shapes i, j, v: ',dis.iBC.shape, dis.jBC.shape, dis.vBC.shape)
    ''' BOTTOM BOUNDARY HAS INCIDENT PLANEWAVE '''
    F = np.zeros(((dis.nelx+1)*(dis.nely+1),1),dtype='complex128') # SYSTEM RIGHT HAND SIDE
    print(dis.iRHS)
    F[dis.iRHS[0]-1] += 1j * waveVector
    F[dis.iRHS[1]-1] += 1j * waveVector
    F = scaling*F
    return F

''' PHYSICS CLASS '''
class phys:
    def __init__(self,scale,eps_r,wlen):
        self.scale = scale
        self.eps_r = eps_r
        self.k = 2*np.pi/(wlen*self.scale)

''' FILTER THRESHOLD CLASS '''
class filthresh:
    def __init__(self,beta,eta):
        self.beta = beta
        self.eta = eta

''' SYSTEM SOLVING, OBJECTIVE FUNCTION AND GRADIENT EVALUATION '''
def objective_grad(dVs,dis,phy,filThr):
    global fig, ax1, ax2
    dFP = np.zeros((dis.nely,dis.nelx))
    dFP[np.arange(dis.nely-1, int(np.ceil(dis.nely * 9 / 10))-1, -1)] = 1
    dFP = dFP.flatten('F') # design field in physics, 0: air
    dFP[dis.dvelmidx.flatten(order='F')] = dVs
    dFP = dFP.reshape((dis.nely, dis.nelx), order='F')  # design field in physics, 0: air
    ''' FILTERING THE DESIGN FIELD AND COMPUTE MATERIAL FIELD '''
    dFPS = density_filter(filThr.filKer, np.ones((dis.nely,dis.nelx)),
                          filThr.filSca,dFP,np.ones((dis.nely,dis.nelx)))
    dis.dFPST = threshold(dFPS,filThr.beta,filThr.eta)
    A,dAdx = material_interpolation(phy.eps_r,dis.dFPST,1.0) # MATERIAL FIELD
    dis.dFP = dFP.copy()

    ''' CONSTRUCT THE SYSTEM MATRIX '''
    F = boundary_conditions_rhs(phy.k,dis,phy.scale)
    # print('F',np.nonzero(F)[0])

    tmp = flat(dis.LEM) - phy.k**2*flat(dis.MEM) @ flat(A).T
    dis.vS = tmp.reshape((16*dis.nelx*dis.nely,1),order='F')
    # print(vars(dis))
    S = csc_matrix((dis.vS.flatten('F'),(dis.iS.flatten('F')-1,dis.jS.flatten('F')-1)),
                    shape=(dis.nodenrs[-1,-1],dis.nodenrs[-1,-1]))
    S1 = csc_matrix((dis.vBC.flatten('F'),(dis.iBC.flatten('F')-1,dis.jBC.flatten('F')-1)),
                    shape=(dis.nodenrs[-1,-1],dis.nodenrs[-1,-1]))
    S += S1
    print('Sysmat shape:',S.shape,'\n',len(S.nonzero()[0]),' nonzero entries')

    ''' SOLVING THE STATE SYSTEM: S @ Ez = F '''
    lu = splu(S) # LU FACTORIZATION
    Ez = lu.solve(F)
    print('Sparse Ez?:' , scipy.sparse.issparse(Ez))
    dis.Ez = Ez.copy()
    print('Ez shape: ', Ez.shape)
    print('Ez: ',Ez[:5])

    '''FIGURE OF MERIT SELECTION MATRIX'''
    P = csc_matrix((0.25*np.ones(4),(dis.edofmat[dis.tElmIdx-1],dis.edofmat[dis.tElmIdx-1])),
                    shape = ((dis.nelx+1)*(dis.nely+1),(dis.nelx+1)*(dis.nely+1))) # Weighting matrix
    print('tElmIdx: ', dis.tElmIdx)
    print('P shape: ',P.shape,np.nonzero(P))
    # spy(P)

    '''GET SOLUTION IN TARGET ELEMENT (INTENSITY AT POINT)'''
    FOM = Ez.T.conj().dot(P.dot(Ez))
    print('FOM: ',FOM)

    '''ADJOINT RIGHT HAND SIDE'''
    AdjRHS = P.T*(2*Ez.real-2*1j*Ez.imag)
    print('AdjRHS: ',AdjRHS[np.nonzero(AdjRHS)])
    # print(AdjRHS[:5])

    '''SOLVING THE ADJOINT SYSTEM: S.T @ AdjLambda = -AdjRHS/2 -- S IS SYMMETRIC, REUSE LU(S)'''
    AdjLambda = lu.solve(-AdjRHS/2)
    print('AdjLambda Shape:',AdjLambda.shape)
    print('Solved Adjoint System')
    print('AdjLambda: ',AdjLambda[:5])
    # AdjLambda = Q1.T @ inv(L.T) @ U.T @ Q2.T @ (-AdjRHS/2) # Solving

    '''COMPUTING SENSITIVITIES'''
    TMP = -phy.k**2 * flat(dis.MEM) @ flat(dAdx).T
    dis.vDS = TMP.reshape(16*dis.nelx*dis.nely,1,order='F')

    '''CONSTRUCT dS/dx'''
    DSdx = csc_matrix((dis.vDS.flatten('F'),
    (dis.ielfull.flatten('F')-1,dis.jelfull.flatten('F')-1)))
    DSdxMulV = DSdx @ Ez[dis.idxDSdx.flatten('F')-1]
    print('DSdxMulV shape:', DSdxMulV.shape)
    DsdxMulV = csc_matrix((DSdxMulV.flatten('F'),
                           (dis.ielsens.flatten('F')-1,dis.jelsens.flatten('F')-1)))
    sens = 2*np.real(AdjLambda[dis.idxDSdx.flatten('F')-1].T @ DsdxMulV) # Computing sensitivities
    print('sens: ',sens[:5])
    sens = sens.reshape((dis.nely,dis.nelx),order='F')
    print('vDS: ',dis.vDS[:5])

    '''FILTERING SENSITIVITIES'''
    DdFSTDFS = derivative_of_threshold(dFPS,filThr.beta,filThr.eta)
    sensFOM = density_filter(filThr.filKer,filThr.filSca,np.ones((dis.nely,dis.nelx)),sens,DdFSTDFS)

    '''EXTRACTING SENSITIVITIES FOR DESIGNABLE REGION'''
    sensFOM = sensFOM.flatten('F')[dis.dvelmidx.flatten('F')-1]
    # sensFOM = sens.flatten('F')[dis.dvelmidx.flatten('F')-1]

    '''SCIPY MINIMIZE, MAKE FOM NEGATIVE TO GET A MAXIMIZATION'''
    FOM = -np.abs(FOM[0][0])
    sensFOM = -sensFOM # may need to .flatten() instead
    print('FOM Sensitivities = ',sensFOM,'\n','FOM = ',FOM)
    fig.suptitle(f'FOM = {abs(FOM):.1f}')
    ax1.imshow(dis.dFP.reshape((dis.nely, dis.nelx), order='F'))
    ax1.set_title('Material Distribution')
    ax2.imshow(np.reshape(np.abs(dis.Ez)**2,(dis.nely + 1, dis.nelx + 1), order='F'))
    ax2.set_title(f'|E$_z$|$^2$')
    fig.canvas.draw()
    plt.pause(.02)
    # plt.savefig(f'{abs(FOM):0.3f}-FOM, metalens-35.png',bbox_inches='tight')
    return (FOM,list(sensFOM))


def topopt(targetXY,dVElmIdx,nElX,nElY,dVini,eps_r,wlen,fR,maxItr,**kwargs):
    '''SET UP PHYSICS PARAMETERS'''
    phy = phys(scale=1e-9,eps_r=eps_r,wlen=wlen)

    '''SET UP ALL INDEX SETS, ELEMENT MATRICES AND RELATED QUANTITIES'''
    tElmIdx = (targetXY[0]-1)*nElY+targetXY[1] # target index
    dis = discretization(nElX,nElY,tElmIdx,dVElmIdx)
    dis.LEM,dis.MEM = element_matrices(phy.scale)
    index_sets_sparse(dis)

    '''SET UP FILTER AND THRESHOLDING PARAMETERS'''
    filthr = filthresh(5,0.5)
    filthr.filKer,filthr.filSca = density_filter_setup(fR,nElX,nElY)

    '''INITIALIZE DESIGN VARIABLES, BOUNDS AND OPTIMIZER OPTIONS'''
    dVs = np.zeros(len(dis.dvelmidx.flatten('F')))
    dVs[np.arange(len(dis.dvelmidx.flatten('F')))] = dVini
    LBdVs = np.zeros((len(dVs+1),1)) # lower bound on design variables
    UBdVs = np.ones((len(dVs+1),1)) # upper bound " "
    opt_bounds = ((i,j) for i,j in zip(LBdVs,UBdVs))

    # plt.imshow(dVs.reshape((dVElmIdx.shape[0],dVElmIdx.shape[1])))
    # plt.colorbar()
    # plt.show()
    '''SOLVE DESIGN PROBLEM USING SCIPY OPTIMIZER'''
    FOM = lambda design_variables: objective_grad(design_variables,dis,phy,filthr)
    plt.ion()
    global fig, ax1, ax2
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.tight_layout()
    opt_dVs = minimize(FOM,dVs,jac=True,method='L-BFGS-B',
                       bounds = opt_bounds)
                       # ,maxiter=maxItr, maxfun=maxItr)
    plt.ioff()
    plt.show()

    '''FINALIZE BINARIZED DESIGN EVALUATION'''
    filthr.beta = 1000
    print('Black/white design evaluation:')
    FOM_,_ = FOM(opt_dVs.x)
    #
    # plt.figure(1)
    # plt.imshow(dis.dFPST.reshape((dis.nely, dis.nelx), order='F'))
    # plt.figure(2)
    # plt.imshow(np.reshape(np.abs(dis.Ez) ** 2, (dis.nely + 1, dis.nelx + 1), order='F'))
    # plt.show()

    return dVs, FOM

if __name__ == '__main__':
    DomainElementsX = 400
    DomainElementsY = 200
    DesignThicknessElements = 15
    DDIdx = tile(np.arange(0, DomainElementsX * DomainElementsY, DomainElementsY), (DesignThicknessElements, 1)) + tile(
        np.arange(165, 165 + DesignThicknessElements), (DomainElementsX, 1)).T
    DVs,obj = topopt([200,80],DDIdx,DomainElementsX,DomainElementsY,0.5,3.0,35,6.0,200)
    plt.ioff()
    plt.show()