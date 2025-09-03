import numpy as np
import torch 
import opt_einsum
import numqi

def inner_product(matA, matB):
    return np.trace(matA.conj().T @ matB)

def hilbert_schmidt_norm(matA):
    return np.sqrt(inner_product(matA, matA)).real

class Hilbert_Schmidt_Measure(torch.nn.Module):
    def __init__(self, dim_list, rank, num_ensemble):
        super().__init__()
        # schmidt/tensor rank
        self.rank = rank
        self.num_ensemble = num_ensemble
        dim_list = tuple(int(x) for x in dim_list)
        self.dim_list = dim_list
        self.manifold_psi = torch.nn.ModuleList([numqi.manifold.Sphere(x,
                batch_size=num_ensemble*rank, dtype=torch.complex128) for x in dim_list])
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
            psi_list = [x() for x in self.manifold_psi]
            psi = self.contract_psi(*psi_list).reshape(num_ensemble, -1)
        sigma = (psi.T * coeff_q) @ psi.conj()
        tmp0 = (sigma - self.target_rho).reshape(-1)
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
