"""C initial geometry only: no labels, GP parameters, champions or test imports."""
import numpy as np
from scipy.spatial.distance import pdist,squareform
from .common import digest

def positive_median(distances):
    d=np.asarray(distances,dtype=np.float64);positive=d[d>0]
    if not np.isfinite(d).all():raise FloatingPointError('Nonfinite unlabelled distances')
    return {'value':float(np.median(positive)) if len(positive) else 1.,'degenerate':not len(positive),
         'positive_pairs':len(positive),'zero_pairs':int(np.count_nonzero(d==0))}

def signal_kernel(public):
    psi=np.asarray(public['wl'],dtype=np.float64);hp=np.asarray(public['x_raw'],dtype=np.float64)[:,-4:]
    if psi.ndim!=2 or hp.shape!=(len(psi),4) or not np.isfinite(psi).all() or not np.isfinite(hp).all():raise ValueError('Invalid frozen unlabelled pool')
    mean=hp.mean(0);scale=hp.std(0,ddof=0);constant=np.ptp(hp,axis=0)==0;scale[constant]=1.
    standardized=(hp-mean)/scale
    dw=pdist(psi);dh=pdist(standardized);sw=positive_median(dw);sh=positive_median(dh)
    def matern(d):
        t=np.sqrt(5.)*d;return (1+t+t*t/3.)*np.exp(-t)
    kw=matern(squareform(dw)/sw['value']);kh=matern(squareform(dh)/sh['value'])
    k=.25*kw+.25*kh+.5*kw*kh
    transform={'HP_pool_mean':mean.tolist(),'HP_pool_std_ddof0':scale.tolist(),'constant_columns':np.flatnonzero(constant).tolist(),
       'dW':sw,'dH':sh,'psi_column_zscore':False,'scale_source':'all768 unlabelled pool only','ellW':1.,'ellH':1.,'sf2':1.,
       'weights':[.25,.25,.5],'epsilon':1e-6,'dtype':'float64','performance_labels_used':False}
    return k,transform

def select(public,first,k=50):
    ids=public['ids'];matrix,transform=signal_kernel(public)
    if first not in ids or len(ids)!=len(set(ids)):raise ValueError('Missing shared first point or duplicate IDs')
    selected=[ids.index(first)];remaining=[i for i in range(len(ids)) if i!=selected[0]]
    steps=[{'step':1,'candidate_id':first,'conditional_variance':float(matrix[selected[0],selected[0]]),'criterion':'B_shared_first_point'}]
    while len(selected)<k:
        kr=matrix[np.ix_(remaining,selected)];ks=matrix[np.ix_(selected,selected)]+np.eye(len(selected))*1e-6
        l=np.linalg.cholesky(ks);v=np.diag(matrix)[remaining]-np.sum(np.linalg.solve(l,kr.T)**2,axis=0)
        if not np.isfinite(v).all() or v.min()<-1e-10:raise FloatingPointError('Invalid C Schur variance')
        pos=min(range(len(remaining)),key=lambda i:(-float(v[i]),ids[remaining[i]]));picked=remaining.pop(pos)
        steps.append({'step':len(selected)+1,'candidate_id':ids[picked],'conditional_variance':float(v[pos]),'criterion':'WI_signal_Schur'})
        selected.append(picked)
    return {'method':'C','first_point':first,'ids':[ids[i] for i in selected],'selection':steps,'transform':transform,
         'kernel':'fixed label-free WI Matern5/2 interaction','pool_identity':digest({k:public[k] for k in ('ids','x_raw','wl')}),'labels_used':False}
