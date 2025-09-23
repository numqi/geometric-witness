import numpy as np
import torch
import opt_einsum
import numqi

from utils import HilbertSchmidtMeasure

np_rng = np.random.default_rng()


def hf_weighted_softmax(x0:torch.Tensor, weight:list[int]):
    assert (x0.ndim==1) and (len(weight)==x0.shape[0])
    x0 = x0 - x0.max()
    x1 = torch.exp(x0)
    x1 = torch.concat([x1[i]*torch.ones(weight[i],dtype=x0.dtype) for i in range(x0.shape[0])], axis=0)
    x1 = x1 / x1.sum()
    return x1


def sort_veca_vecb(vecab):
    veca,vecb = vecab
    if veca[1]<0:
        veca = -veca
    if vecb[1]<0:
        vecb = -vecb
    if (veca[0]<0) and (vecb[0]>0):
        ret = np.stack([vecb,veca[::-1]], axis=0)
    elif (veca[0]<0) and (vecb[0]<0):
        ret = np.stack([veca[::-1],vecb[::-1]], axis=0)
    elif (veca[0]>0) and (vecb[0]<0):
        ret = np.stack([vecb[::-1],veca], axis=0)
    elif (veca[2]<0) and (vecb[2]>0):
        ret = np.stack([vecb[::-1], veca], axis=0)
    else:
        ret = np.stack([veca,vecb], axis=0)
    return ret

def sort_vece_vecf(vecef):
    vece,vecf = vecef
    if vece[0]<0:
        vece = -vece
    if vecf[0]<0:
        vecf = -vecf
    if vece[1]<0:
        ret = np.stack([vecf,vece], axis=0)
    else:
        ret = np.stack([vece,vecf], axis=0)
    return ret

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim_list = (3,3)
        num_ensemble = 10
        dtype = torch.float64
        self.num_ensemble = num_ensemble
        dim_list = tuple(int(x) for x in dim_list)
        self.dim_list = dim_list
        self.target_rho = None
        N0 = len(dim_list)
        shape_list = [(num_ensemble,x) for x in dim_list]
        index_list = [(N0+1,x) for x in range(N0)]
        self.contract_psi = opt_einsum.contract_expression(*[y for x in zip(shape_list,index_list) for y in x], [N0+1]+list(range(N0)))
        self.softmax_weight = [4,4,2]
        self.theta_coeff_p = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
        self.manifold_psiAB = numqi.manifold.Sphere(3, batch_size=4, dtype=dtype)
        self.theta_vec_e_f = torch.nn.Parameter(torch.randn(2, dtype=torch.float64))

    def set_target_rho(self, np0):
        assert (np0.ndim==2) and (np0.shape[0]==np0.shape[1]) and (np0.shape[0]==np.prod(self.dim_list))
        assert np.abs(np0-np0.T.conj()).max() < 1e-10
        self.target_rho = torch.tensor(np0, dtype=torch.float64)

    def forward(self, return_info=False):
        hff = lambda x: torch.flip(x, dims=[0])
        coeff_q = hf_weighted_softmax(self.theta_coeff_p, self.softmax_weight)
        tmp0 = self.manifold_psiAB()
        tmp1 = torch.cos(self.theta_vec_e_f)
        tmp2 = torch.sin(self.theta_vec_e_f)/np.sqrt(2)
        vece,vecf = torch.stack([tmp2,tmp1,tmp2], axis=1)
        veca,vecb,vecc,vecd = self.manifold_psiAB()
        vecAB = torch.stack([veca, vecb, vecc, vecd, vece, vecf])
        psiA = torch.stack([veca, vecb, hff(vecb), hff(veca), vecc, vecd, hff(vecd), hff(vecc), vece, vecf])
        psiB = torch.stack([vecb,hff(veca),veca,hff(vecb),vecd,hff(vecc),vecc,hff(vecd),vecf,vece])
        psi_list = psiA,psiB

        # coeff_q = self.manifold_ensemble_coeff()
        # psi_list = [self.manifold_psiA(), self.manifold_psiB()]

        num_ensemble = coeff_q.shape[0]
        psi = self.contract_psi(*psi_list).reshape(num_ensemble, -1)
        sigma = (psi.T * coeff_q) @ psi.conj()
        tmp0 = (sigma - self.target_rho).reshape(-1)
        loss = torch.vdot(tmp0, tmp0).real
        ret = loss
        if return_info:
            tmp0 = coeff_q.detach().numpy()
            ind0 = np.argsort(tmp0)
            coeff_p = tmp0[ind0]
            coeff_psiA = psi_list[0].detach().numpy()[ind0]
            coeff_psiB = psi_list[1].detach().numpy()[ind0]
            vecAB = vecAB.detach().numpy()
            vecp = coeff_q.detach().numpy()[[0,4,8]]
            if vecp[0]>vecp[1]:
                vecp = vecp[[1,0,2]]
                vecAB = vecAB[[2,3,0,1,4,5]]
            vecAB[:2] = sort_veca_vecb(vecAB[:2])
            vecAB[2:4] = sort_veca_vecb(vecAB[2:4])
            vecAB[4:] = sort_vece_vecf(vecAB[4:])
            info = dict(distance=np.sqrt(loss.detach().numpy()), coeff_p=coeff_p, coeff_psiA=coeff_psiA, coeff_psiB=coeff_psiB,
                        vecAB=vecAB, vecp=vecp)
            ret = loss, info
        return ret

rho_bes = numqi.entangle.load_upb('tiles', return_bes=True)[1]
alpha = 1

model = HilbertSchmidtMeasure(dim_list=[3,3], num_ensemble=18, rank=1, dtype=torch.float64)
model.set_target_rho(numqi.utils.hf_interpolate_dm(rho_bes, alpha=alpha))
theta_optim = numqi.optimize.minimize(model, num_repeat=10, tol=1e-14, print_every_round=0)
info_best = model(return_info=True)[1]

model = DummyModel()
model.set_target_rho(numqi.utils.hf_interpolate_dm(rho_bes, alpha=alpha))
theta_optim = numqi.optimize.minimize(model, num_repeat=30, tol=1e-14, print_every_round=1)
# optim_fun = numqi.optimize.minimize_adam(model, num_step=int(1e5), theta0='uniform', optim_args=('adam',0.001))
# optim_fun = numqi.optimize.minimize_adam(model, num_step=int(1e5), theta0='no-init', optim_args=('adam',0.0001))
info = model(return_info=True)[1]
print(f'p={alpha}', abs(info['distance']-info_best['distance']))
if abs(info['distance']-info_best['distance'])< 1e-11:
    theta_optim = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-25, print_freq=10)
    coeff_p = info['coeff_p']
    coeff_psiA = info['coeff_psiA']
    coeff_psiB = info['coeff_psiB']
    coeff_psi_pretty = np.concat([coeff_psiA, 0*coeff_psiA[:,:1], coeff_psiB], axis=1)
    # print(coeff_p, coeff_psi_pretty, sep='\n')

    sigma = np.einsum(coeff_p, [0], coeff_psiA, [0,1], coeff_psiA.conj(), [0,3], coeff_psiB, [0,2], coeff_psiB.conj(), [0,4], [1,2,3,4], optimize=True).reshape(9,9)
    x0 = np.linalg.norm(sigma - numqi.utils.hf_interpolate_dm(rho_bes, alpha=alpha), ord='fro')
    assert abs(x0 - info_best['distance']) < 1e-10

    vecAB = info['vecAB']
    vecp = info['vecp']
    print(vecp, vecAB, sep='\n')

    veca,vecb,vecc,vecd,vece,vecf = vecAB
    tmp0 = np.array([vecp[0]]*4 + [vecp[1]]*4 + [vecp[2]]*2)
    tmp1 = np.stack([veca, vecb, vecb[::-1], veca[::-1], vecc, vecd, vecd[::-1], vecc[::-1], vece, vecf])
    tmp2 = np.stack([vecb,veca[::-1],veca,vecb[::-1],vecd,vecc[::-1],vecc,vecd[::-1],vecf,vece])
    sigma = np.einsum(tmp0, [0], tmp1, [0,1], tmp1, [0,3], tmp2, [0,2], tmp2, [0,4], [1,2,3,4], optimize=True).reshape(9,9)
    x0 = np.linalg.norm(sigma - numqi.utils.hf_interpolate_dm(rho_bes, alpha=alpha), ord='fro')
    assert abs(x0 - info_best['distance']) < 1e-10


'''
[0.0757194012 0.1304440966 0.0876730043]
[[ 0.8229256465  0.5311820089  0.2015913035]
 [ 0.4803624773  0.3840457767 -0.7885180605]
 [ 0.7069252161  0.7070734526 -0.0174319096]
 [ 0.1899794768  0.4370482698 -0.8791453852]
 [ 0.0217552097  0.9995265988  0.0217552097]
 [ 0.2725407668 -0.9227367235  0.2725407668]]

[0.0757193265 0.1304441659 0.0876730153]
[[ 0.8229256796  0.5311819888  0.2015912217]
 [ 0.4803623965  0.3840458538 -0.7885180722]
 [ 0.7069253535  0.7070733206 -0.017431693 ]
 [ 0.189979603   0.4370481659 -0.8791454095]
 [ 0.0217551412  0.9995266018  0.0217551412]
 [ 0.2725407463 -0.9227367356  0.2725407463]]

[0.0757195714 0.1304439247 0.0876730079]
[[ 0.8229255764  0.5311822803  0.2015908749]
 [ 0.4803619711  0.3840460136 -0.7885182536]
 [ 0.7069251725  0.7070734956 -0.0174319332]
 [ 0.1899792951  0.4370482207 -0.8791454489]
 [ 0.0217551263  0.9995266024  0.0217551263]
 [ 0.2725407417 -0.9227367383  0.2725407417]]

[[ a0  a1  a2  0.         b0  b1  b2]
 [ b0  b1  b2  0.         a2  a1  a0]
 [ b2  b1  b0  0.         a0  a1  a2]
 [ a2  a1  a0  0.         b2  b1  b0]

 [ c0  c1  c2  0.          d0  d1  d2]
 [ d0  d1  d2  0.          c2  c1  c0]
 [ d2  d1  d0  0.          c0  c1  c2]
 [ c2  c1  c0  0.          d2  d1  d0]

 [ e0  e1  e2  0.          f0  f1  f2]
 [ f0  f1  f2  0.          e0  e1  e2]]
'''
