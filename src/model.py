"""Frozen chamber-specific structural model; no per-year recipe selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,warnings
import numpy as np,pandas as pd,joblib
from scipy.special import logit,expit
from scipy.stats import t
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
BASE=Path(__file__).resolve().parents[1]

def safe_logit(x):return logit(np.clip(np.asarray(x,dtype=float),.005,.995))

def fit_all():
    config=json.loads((BASE/'model_spec.json').read_text())
    d=pd.read_csv(BASE/'inputs/training_features.csv');raw=pd.read_csv(BASE/'inputs/calibration_raw.csv');audit=[]
    for ch,recipes in config['chambers'].items():
        tr=d[d.chamber.eq(ch)&d.Year.between(2002,2024)].copy()
        for r in recipes:
            pipe=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),MLPRegressor(hidden_layer_sizes=tuple(r['layers']),activation='tanh',solver='lbfgs',alpha=r['alpha'],max_iter=2000,max_fun=50000,tol=1e-4,random_state=184,early_stopping=False))
            with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                warnings.simplefilter('always',ConvergenceWarning)
                pipe.fit(tr[r['columns']],safe_logit(tr.observed_vote)-safe_logit(tr.lagged_pres_vote.fillna(.5)))
            nw=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
            if nw:raise RuntimeError('Model failed convergence check')
            joblib.dump({'pipeline':pipe,'columns':r['columns'],'weight':r['weight']},BASE/'models'/f'{r["name"]}.joblib')
            audit.append(dict(chamber=ch,recipe=r['name'],train_start=2002,train_end=2024,n=len(tr),convergence_warnings=nw))
        g=raw[raw.chamber.eq(ch)&raw.Year.between(2014,2024)].copy()
        # Calibration matches the compatible-round cohort used in V7/V9.
        g=g[~((g.chamber.eq('senate')&g.state.eq('GA'))|(g.chamber.eq('house')&((g.state.eq('ME')&g.Year.ge(2018))|(g.state.eq('AK')&g.Year.ge(2022)))))]
        g['err']=safe_logit(g.observed_vote)-g.raw_logit;by=g.groupby('Year').err.mean();bias=float(by.mean())
        var=float(g.assign(e2=(g.err-bias)**2).groupby('Year').e2.mean().mean()+by.var(ddof=1)/len(by))
        config.setdefault('calibration_parameters',{})[ch]=dict(bias=bias,sd=float(np.sqrt(max(var,.0025))),df=5,start=2014,end=2024)
    (BASE/'models/calibration.json').write_text(json.dumps(config['calibration_parameters'],indent=2));pd.DataFrame(audit).to_csv(BASE/'results/training_audit.csv',index=False)

def predict(d):
    config=json.loads((BASE/'model_spec.json').read_text());cal=json.loads((BASE/'models/calibration.json').read_text());parts=[]
    if not d.Year.eq(2026).all():raise ValueError('2026 features required')
    for ch,g in d.groupby('chamber'):
        mu=np.zeros(len(g))
        for r in config['chambers'][ch]:
            saved=joblib.load(BASE/'models'/f'{r["name"]}.joblib')
            with threadpool_limits(limits=1):mu+=r['weight']*(saved['pipeline'].predict(g[r['columns']])+safe_logit(g.lagged_pres_vote))
        c=cal[ch];center=mu+c['bias'];scale=c['sd']*np.sqrt(3/5)
        q=g[['race_id','chamber','democrat','republican']].copy();q['raw_logit']=mu;q['predicted_dem_two_party_share']=expit(center)
        q['p_dem_two_party']=np.clip(t.cdf(center/scale,5),1e-8,1-1e-8);q['lo95']=expit(center+t.ppf(.025,5)*scale);q['hi95']=expit(center+t.ppf(.975,5)*scale);parts.append(q)
    return pd.concat(parts,ignore_index=True)

if __name__=='__main__':fit_all()
