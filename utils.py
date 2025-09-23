import itertools
import functools
import numpy as np
import torch
import opt_einsum
import numqi
import operator
import scipy.linalg

def depolarizing_channel(rho, p):
    dim = rho.shape[0]
    return p * rho + (1-p) * np.eye(dim) / dim

def inner_product(matA, matB):
    return np.trace(matA.conj().T @ matB)

def hilbert_schmidt_norm(matA):
    return np.sqrt(inner_product(matA, matA)).real

def density_matrix_fidelity(rho, sigma):
    """Compute the fidelity between two density matrices."""
    sqrt_rho = scipy.linalg.sqrtm(rho)
    middle_matrix = sqrt_rho @ sigma @ sqrt_rho
    sqrt_middle = scipy.linalg.sqrtm(middle_matrix)
    tmp = np.trace(sqrt_middle)
    return np.real(tmp)**2

class HilbertSchmidtMeasure(torch.nn.Module):
    def __init__(self, dim_list, rank, num_ensemble, dtype=torch.complex128):
        super().__init__()
        assert dtype in {torch.float64, torch.complex128}
        # schmidt/tensor rank
        self.rank = rank
        self.num_ensemble = num_ensemble
        dim_list = tuple(int(x) for x in dim_list)
        self.dim_list = dim_list
        self.manifold_psi = torch.nn.ModuleList([numqi.manifold.Sphere(x,
                batch_size=num_ensemble*rank, dtype=dtype) for x in dim_list])
        self.target_rho = None
        N0 = len(dim_list)
        self.manifold_ensemble_coeff = numqi.manifold.DiscreteProbability(num_ensemble, dtype=torch.float64)
        if rank>1:
            self.manifold_psi_coeff = numqi.manifold.PositiveReal(num_ensemble*rank, dtype=torch.float64)
            # contract_expression for norm computation
            # coeff_p,coeff_p,psi_conj,psi
            shape_list = [(num_ensemble,rank),(num_ensemble,rank)] + [(num_ensemble,rank,x) for x in dim_list] + [(num_ensemble,rank,x) for x in dim_list]
            index_list = [(N0+2,N0),(N0+2,N0+1)] + [(N0+2,N0,x) for x in range(N0)] + [(N0+2,N0+1,x) for x in range(N0)]
            self.contract_psi_psi = opt_einsum.contract_expression(*[y for x in zip(shape_list,index_list) for y in x], [N0+2])
            # construct psi list
            shape_list = [(num_ensemble,rank)] + [(num_ensemble,rank,x) for x in dim_list]
            index_list = [(N0+1,N0)] + [(N0+1,N0,x) for x in range(N0)]
            self.contract_psi = opt_einsum.contract_expression(*[y for x in zip(shape_list,index_list) for y in x], [N0+1]+list(range(N0)))
        else:
            shape_list = [(num_ensemble,x) for x in dim_list]
            index_list = [(N0+1,x) for x in range(N0)]
            # construct psi list
            self.contract_psi = opt_einsum.contract_expression(*[y for x in zip(shape_list,index_list) for y in x], [N0+1]+list(range(N0)))

    def set_target_rho(self, np0):
        assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and (np0.shape[0]==np.prod(self.dim_list))
        assert np.abs(np0-np0.T.conj()).max() < 1e-10
        self.target_rho = torch.tensor(np0, dtype=torch.complex128)

    def forward(self, return_info=False):
        coeff_q = self.manifold_ensemble_coeff().to(torch.complex128)
        num_ensemble = coeff_q.shape[0]
        if hasattr(self, 'manifold_psi_coeff'):
            psi_coeff = self.manifold_psi_coeff().to(torch.complex128).reshape(num_ensemble, -1)
            tmp0 = [x() for x in self.manifold_psi]
            psi_list = [x.reshape(num_ensemble,-1,x.shape[1]) for x in tmp0]
            psi_conj_list = [x.conj() for x in psi_list]
            norm_list = self.contract_psi_psi(psi_coeff, psi_coeff, *psi_conj_list, *psi_list)
            # normalization
            psi_coeff = psi_coeff / torch.sqrt(norm_list.real.reshape(-1,1))
            psi = self.contract_psi(psi_coeff, *psi_list).reshape(num_ensemble, -1)
        else:
            psi_list = [x().to(torch.complex128) for x in self.manifold_psi]
            psi = self.contract_psi(*psi_list).reshape(num_ensemble, -1)
        sigma = (psi.T * coeff_q) @ psi.conj()
        tmp0 = (sigma - self.target_rho).reshape(-1)
        # compute the square of distance
        loss = torch.vdot(tmp0, tmp0).real
        ret = loss
        if return_info:
            sigma = sigma.detach().numpy()
            rho = self.target_rho.numpy()
            witness = ((sigma-rho)-inner_product(sigma, sigma-rho)*np.identity(sigma.shape[0]))/hilbert_schmidt_norm(sigma-rho)
            distance = np.sqrt(loss.detach().numpy())
            info = dict(distance=distance, sigma=sigma, witness=witness)
            ret = loss, info
        return ret

def generate_bipartitions(n):
    systems = list(range(n))
    bipartitions = set()
    for i in range(1, n // 2 + 1):
        for group1 in itertools.combinations(systems, i):
            group1 = list(group1)
            group2 = [x for x in systems if x not in group1]
            group1, group2 = sorted(group1), sorted(group2)
            bipartitions.add(frozenset([tuple(group1), tuple(group2)]))
    result = [sorted(map(list, partition), key=lambda x: x[0]) for partition in bipartitions]
    return sorted(result, key=lambda x: x[0])

class GenuineHilbertSchmidtMeasure(torch.nn.Module):
    def __init__(self, dim_list, num_ensemble, distance:str='hs', dtype=torch.complex128):
        super().__init__()
        assert distance in {'hs','trace','fidelity'}
        self.dim_list = dim_list
        self.num_ensemble = num_ensemble
        self.target_rho = None
        self.distance = distance
        N0 = len(dim_list)
        bipartition_list = generate_bipartitions(N0)
        self.bipartition_list = bipartition_list
        manifold_psiA_list = torch.nn.ModuleList()
        manifold_psiB_list = torch.nn.ModuleList()
        for bipartition in bipartition_list:
            dimA = functools.reduce(operator.mul, (dim_list[i] for i in bipartition[0]))
            dimB = functools.reduce(operator.mul, (dim_list[i] for i in bipartition[1]))
            manifold_psiA_list.append(numqi.manifold.Sphere(dimA, batch_size=num_ensemble, dtype=dtype))
            manifold_psiB_list.append(numqi.manifold.Sphere(dimB, batch_size=num_ensemble, dtype=dtype))
        self.manifold_psiA_list = manifold_psiA_list
        self.manifold_psiB_list = manifold_psiB_list
        self.manifold_ensemble_coeff = numqi.manifold.DiscreteProbability(num_ensemble*len(bipartition_list), dtype=torch.float64)

        contract_psi = []
        for i in range(len(bipartition_list)):
            shapeA = [dim_list[j] for j in bipartition_list[i][0]]
            shapeB = [dim_list[j] for j in bipartition_list[i][1]]
            shape_list = [(num_ensemble, *shapeA), (num_ensemble, *shapeB)]
            indexA = tuple([N0] + bipartition_list[i][0])
            indexB = tuple([N0] + bipartition_list[i][1])
            index_list = [indexA, indexB]
            contract_psi.append(opt_einsum.contract_expression(*[y for x in zip(shape_list,index_list) for y in x], [N0]+list(range(N0))))

        self.contract_psi = contract_psi

    def set_target_rho(self, np0, zero_eps:float=1e-10):
        assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and (np0.shape[0]==np.prod(self.dim_list))
        assert np.abs(np0-np0.T.conj()).max() < zero_eps
        assert abs(np.trace(np0)-1) < zero_eps
        EVL,EVC = np.linalg.eigh(np0)
        assert EVL[0]>-zero_eps
        tmp0 = EVL>zero_eps
        self._sqrt_rho = torch.tensor(np.sqrt(EVL[tmp0]).reshape(-1,1)*EVC.T.conj()[tmp0], dtype=torch.complex128)
        self.target_rho = torch.tensor(np0, dtype=torch.complex128)

    def forward(self,return_info=False):
        coeff_q = self.manifold_ensemble_coeff().to(torch.complex128)
        psi_A_list = [x().to(torch.complex128) for x in self.manifold_psiA_list]
        psi_B_list = [x().to(torch.complex128) for x in self.manifold_psiB_list]
        psi_AB_list = []
        for i in range(len(psi_A_list)):
            shapeA = [self.dim_list[j] for j in self.bipartition_list[i][0]]
            shapeB = [self.dim_list[j] for j in self.bipartition_list[i][1]]
            psi_A = psi_A_list[i].reshape(self.num_ensemble,*shapeA)
            psi_B = psi_B_list[i].reshape(self.num_ensemble,*shapeB)
            psi_AB_list.append(self.contract_psi[i](psi_A, psi_B).reshape(self.num_ensemble, -1))
        psi_AB = torch.cat(psi_AB_list, dim=0)
        sigma = (psi_AB.T * coeff_q) @ psi_AB.conj()
        if self.distance == 'hs':
            tmp0 = (sigma - self.target_rho).reshape(-1)
            ret = torch.vdot(tmp0, tmp0).real
        elif self.distance == 'trace':
            ret = 0.5*torch.linalg.svdvals(sigma - self.target_rho).sum()
        else:
            ret = 1-torch.sqrt(torch.linalg.svdvals(self._sqrt_rho @ sigma @ self._sqrt_rho.T.conj())).sum()**2
        if return_info:
            sigma = sigma.detach().numpy()
            rho = self.target_rho.numpy()
            witness = ((sigma-rho)-inner_product(sigma, sigma-rho)*np.identity(sigma.shape[0]))/hilbert_schmidt_norm(sigma-rho)
            tmp0 = np.maximum(ret.item(),0)
            distance = np.sqrt(np.sqrt(tmp0) if (self.distance=='hs') else tmp0)
            info = dict(distance=distance, sigma=sigma, witness=witness)
            ret = ret, info
        return ret


def get_grid_state(edge_list:list[list[tuple[int,int]]], dimA:int|None=None, dimB:int|None=None)->np.ndarray:
    '''create a grid state with given edge_list

    reference: High Schmidt number concentration in quantum bound entangled states
    [arxiv-link](https://arxiv.org/abs/2402.12966)

    Parameters:
        edge_list (list[list[tuple[int,int]]): list of edges, each element is a list of tuples (i,j) where i is the index of system A and j is the index of system B
        dimA (int,None): dimension of system A, if None, it is set to the maximum index in edge_list
        dimB (int,None): dimension of system B, if None, it is set to the maximum index in edge_list

    Returns:
        rho (np.ndarray): density matrix of the grid state
    '''
    if dimA is None:
        dimA = max(y[0] for x in edge_list for y in x) + 1
    if dimB is None:
        dimB = max(y[1] for x in edge_list for y in x) + 1
    tmp0 = np.zeros((len(edge_list), dimA, dimB), dtype=np.float64)
    ind0 = np.array([(i,y0,y1) for i,x0 in enumerate(edge_list) for y0,y1 in x0])
    tmp0[ind0[:,0], ind0[:,1], ind0[:,2]] = 1
    tmp1 = tmp0.reshape(-1, dimA*dimB)
    ret = tmp1.T @ tmp1.conj()
    ret /= np.trace(ret)
    return ret

def get_bound_entangled_state_Marcus2018(dA1:int, dA2:int):
    '''create a bound entangled state from Marcus2018

    reference: High-Dimensional Entanglement in States with Positive Partial Transposition
    [arxiv-link](https://arxiv.org/abs/1802.04975)

    Parameters:
        dA1 (int): dimension of system A1
        dA2 (int): dimension of system A2

    Returns:
        rho (np.ndarray): density matrix of shape `(dA1*dA1*dA2*dA2, dA1*dA1*dA2*dA2)`
    '''
    assert (dA1>=2) and (dA2>=2)
    # numqi.state.maximally_entangled_state(dA1)
    tmp0 = np.eye(dA1).reshape(-1)/np.sqrt(dA1) #maximally entangled state
    omegaA1B1 = tmp0.reshape(-1,1)*tmp0.conj()
    tmp0 = np.eye(dA2).reshape(-1)/np.sqrt(dA2) #maximally entangled state
    omegaA2B2 = tmp0.reshape(-1,1)*tmp0.conj()
    tmp0 = (np.eye(dA1*dA1) - omegaA1B1).reshape(dA1,dA1,dA1,dA1)
    tmp1 = (np.eye(dA2*dA2) - omegaA2B2).reshape(dA2,dA2,dA2,dA2)
    tmp2 = omegaA1B1.reshape(dA1,dA1,dA1,dA1)
    tmp3 = (dA2+1) * omegaA2B2.reshape(dA2,dA2,dA2,dA2)
    ret = (np.einsum(tmp0, [0,1,2,3], tmp1, [4,5,6,7], [0,4,1,5,2,6,3,7], optimize=True)
          + np.einsum(tmp2, [0,1,2,3], tmp3, [4,5,6,7], [0,4,1,5,2,6,3,7], optimize=True)).reshape(dA1*dA1*dA2*dA2, -1)
    ret /= np.trace(ret)
    # SN(ret) >= dA2//dA1
    return ret

def get_bound_entangled_state(key:str, **kwargs):
    if key=='grid-553':
        edge_list = [[(0,0)], [(0,1)], [(1,0)], [(2,3)], [(2,3)], [(3,2)], [(3,2)],
                    [(1,4)], [(4,1)], [(2,1),(4,3)], [(2,2),(3,3)], [(1,2),(3,4)], [(0,2),(1,1),(2,0)]]
        ret = get_grid_state(edge_list, dimA=5, dimB=5)
    elif key=='Marcus2018':
        dA1 = kwargs.get('dA1', 2)
        dB1 = kwargs.get('dB1', 4)
        ret = get_bound_entangled_state_Marcus2018(dA1, dB1)
    else:
        raise ValueError(f'key={key} is not supported')
    return ret

state_01 = np.kron(np.eye(3)[0], np.eye(3)[1])  # |01>
state_12 = np.kron(np.eye(3)[1], np.eye(3)[2])  # |12>
state_20 = np.kron(np.eye(3)[2], np.eye(3)[0])  # |20>

proj_01 = np.outer(state_01, state_01)  # |01><01|
proj_12 = np.outer(state_12, state_12)  # |12><12|
proj_20 = np.outer(state_20, state_20)  # |20><20|

sigma_plus = (proj_01 + proj_12 + proj_20) / 3

state_10 = np.kron(np.eye(3)[1], np.eye(3)[0])  # |10>
state_21 = np.kron(np.eye(3)[2], np.eye(3)[1])  # |21>
state_02 = np.kron(np.eye(3)[0], np.eye(3)[2])  # |02>

proj_10 = np.outer(state_10, state_10)  # |10><10|
proj_21 = np.outer(state_21, state_21)  # |21><21|
proj_02 = np.outer(state_02, state_02)  # |02><02|

sigma_minus = (proj_10 + proj_21 + proj_02) / 3

def horodecki_state(b):
    max_entangled_state = numqi.state.maximally_entangled_state(3)
    tmp0 = (2/7) * np.outer(max_entangled_state, max_entangled_state) + (b/7) * sigma_plus + (5-b)/7 * sigma_minus
    return tmp0
