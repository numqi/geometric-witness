import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numqi

from utils import HilbertSchmidtMeasure

np_rng = np.random.default_rng()


rho_bes = numqi.entangle.load_upb('tiles', return_bes=True)[1]
alpha = 0.9

model = HilbertSchmidtMeasure(dim_list=[3,3], rank=1, num_ensemble=18, dtype=torch.float64)
model.set_target_rho(numqi.utils.hf_interpolate_dm(rho_bes, alpha=alpha))
theta_optim = numqi.optimize.minimize(model, num_repeat=30, tol=1e-14, print_every_round=0)
info_best = model(return_info=True)[1]

model = HilbertSchmidtMeasure(dim_list=[3,3], rank=1, num_ensemble=10, dtype=torch.float64)
model.set_target_rho(numqi.utils.hf_interpolate_dm(rho_bes, alpha=alpha))
theta_optim = numqi.optimize.minimize(model, num_repeat=30, tol=1e-14, print_every_round=0)
info = model(return_info=True)[1]
print(f'p={alpha}')
if abs(info['distance']-info_best['distance'])< 1e-11:
    # theta_optim = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-20, print_freq=10)
    tmp0 = model.manifold_ensemble_coeff().detach().numpy()
    tmp1 = [x().detach().numpy() for x in model.manifold_psi]
    tmp1 = [x*np.sign(x[:,:1]) for x in tmp1]
    ind0 = np.argsort(tmp0)
    coeff_p = tmp0[ind0]
    coeff_psi = [x[ind0] for x in tmp1]
    coeff_psi_pretty = np.concat([coeff_psi[0], 0*coeff_psi[0][:,:1], coeff_psi[1]], axis=1)
    print(coeff_p, coeff_psi_pretty, sep='\n')
else:
    print(abs(info['distance']-info_best['distance']))


# tmp0 = info['sigma']
# threshold = 1e-7
# sigma = tmp0.real*(np.abs(tmp0.real)>=threshold) + 1j*tmp0.imag*(np.abs(tmp0.imag)>=threshold)
# if np.abs(sigma.imag).max() < threshold:
#     sigma = sigma.real.copy()

dlist = []
sigma_list = []
p_list = np.linspace(0.9, 1, 30)
for p in tqdm(p_list):
    model.set_target_rho(numqi.utils.hf_interpolate_dm(rho_bes, alpha=p))
    theta_optim = numqi.optimize.minimize(model, num_repeat=10, tol=1e-14, print_every_round=0)
    info = model(return_info=True)[1]
    dlist.append(info['distance'])
    sigma_list.append(info['sigma'])
sigma_list = np.stack(sigma_list)
if np.abs(sigma_list.imag).max() < 1e-10:
    sigma_list = sigma_list.real.copy()
dlist = np.array(dlist)

dlist_ = [0.01096270264308223, 0.014493836455404138, 0.018050664015030294, 0.021629962177489453, 0.025229073995343528,
        0.02884577448422723, 0.03247818879445587, 0.036124726148011564, 0.0397840265739638, 0.04345491342229373]

assert np.abs(sigma_list.transpose(0,2,1) - sigma_list).max() < 1e-12
assert np.abs(sigma_list[:,::-1,::-1] - sigma_list).max() < 1e-6

# zc0 = (sigma_list - sigma_list.mean(axis=0)).reshape(sigma_list.shape[0], -1)

INDEX = np.array([[1,3,4,5,6,7,4,7,9], [3,2,5,6,8,6,7,10,7], [4,5,1,7,6,3,9,7,4], [5,6,7,2,8,10,3,6,7],
    [6,8,6,8,11,8,6,8,6], [7,6,3,10,8,2,7,6,5], [4,7,9,3,6,7,1,5,4], [7,10,7,6,8,6,5,2,3],
    [9,7,4,7,6,5,4,3,1]], dtype=np.int64)
assert np.all(INDEX[::-1,::-1]==INDEX) and np.all(INDEX.T==INDEX)
for x in range(1,11):
    ind1 = np.nonzero(INDEX.reshape(-1)==x)[0]
    tmp0 = sigma_list.reshape(-1, 81)[:,ind1]
    assert np.max(tmp0 - tmp0.mean(axis=1,keepdims=True)).max() < 1e-7
INDEXa = np.array([np.nonzero(INDEX.reshape(-1)==x)[0][0] for x in range(1,12)])
zc0 = sigma_list.reshape(-1, 81)[:,INDEXa]
U,S,V = np.linalg.svd(zc0[:,:-1], full_matrices=False)
assert np.abs((U*S) @ V - zc0[:,:-1]).max() < 1e-12
mask = S>1e-6
assert np.abs((U*S)[:,mask] @ V[mask] - zc0[:,:-1]).max() < 1e-6

np.abs(zc0[:,:-1] @ V[np.logical_not(mask)].T).max()



zc0 = sigma_list.reshape(-1, 81)[:,INDEXa]

[10,8,3,7,5,4,6]
ind0 = [0,1,2,9]
for x in ind0:
    tmp0 = np.linalg.svd(zc0[:, sorted(set(ind0) - {x})], compute_uv=False)
    tmp0[tmp0<1e-6] = 0
    print(x, tmp0)
np.linalg.svd(zc0[:,[0,1,2,3,4,5,6,7,8,9]], compute_uv=False)

zc0[:,[0,1,2,9]]
zc0[:,10]
print('x_k = a*x_1 + b*x_2 + c*x_3 + d*x_10')
print('k,   [a,b,c,d]')
for x in [10,8,3,7,5,4,6]:
    coeff, residuals, rank, s = np.linalg.lstsq(zc0[:,[0,1,2,9]], zc0[:,x], rcond=None)
    assert residuals<1e-12
    print(x+1, coeff)


print("coeff:", coeff)

fig,ax = plt.subplots()
tmp0 = (zc0 - zc0[-1]).T
for k in [2]:
    ax.plot(p_list, tmp0[k], label=f'x{k+1} (shifted)')
ax.legend()
ax.set_xlabel('p')
ax.set_ylabel('x_k')
ax.grid()
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
