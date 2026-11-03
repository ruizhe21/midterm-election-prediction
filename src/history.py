from pathlib import Path
import json,re,unicodedata
import pandas as pd,numpy as np
RAW=Path(__file__).resolve().parents[1]/"inputs"

def normalize(name):
    name=unicodedata.normalize('NFKD',str(name)).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z ]','',name).strip()

def histories(d):
    """No candidate history or map lookup can inspect an outcome at t or later.

    House raw district lags reset on recorded boundary changes. Supplied lagged
    presidential and legislative features are retained with source provenance.
    Senate exact 6/12/18-year lags proxy seat history, not a Senate-class registry.
    """
    lines=json.loads((RAW/'map_change_years.json').read_text())['lines']
    history={}; candidates={}; national={}; records=[]
    for year, group in d.groupby('Year',sort=True):
        for row in group.itertuples():
            key=(row.chamber,row.race)
            previous=history.get(key,[])
            changes=[x for x in lines.get(row.race,[]) if x<=year] if row.chamber=='house' else []
            boundary=max(changes,default=1946)
            valid=[r for r in previous if r.Year>=boundary]
            f={'map_changed':int(row.chamber=='house' and boundary==year),
               'inc_president_alignment':row.inc*(2*row.president_D-1),
               'midterm_inc':row.midterm*row.inc,
               'midterm_lean':row.midterm*(row.lagged_pres_vote-.5),
               'president_lean':(2*row.president_D-1)*(row.lagged_pres_vote-.5),
               'history_count':len(valid)}
            step=2 if row.chamber=='house' else 6
            byyear={r.Year:r for r in valid}
            for j in [1,2,3]:
                r=byyear.get(year-step*j)
                f[f'own_lag_{j}']=r.observed_vote if r is not None else np.nan
                f[f'overperformance_lag_{j}']=r.observed_vote-r.lagged_pres_vote if r is not None else np.nan
            for half in [4,8,16]:
                w=np.array([2**(-(year-r.Year)/half) for r in valid])
                f[f'ewm_vote_{half}']=np.average([r.observed_vote for r in valid],weights=w) if len(w) else np.nan
                f[f'ewm_overperformance_{half}']=np.average([r.observed_vote-r.lagged_pres_vote for r in valid],weights=w) if len(w) else np.nan
            for party, name in [('D',row.democrat),('R',row.republican)]:
                # State and chamber avoid conflating people with the same name.
                prior=candidates.get((row.state,row.chamber,party,normalize(name)),[])
                if prior:
                    w=np.array([2**(-(year-y)/8) for y,v in prior])
                    f[f'candidate_{party}_past']=np.average([v for y,v in prior],weights=w)*len(prior)/(len(prior)+3)
                else: f[f'candidate_{party}_past']=0.
                f[f'candidate_{party}_n']=len(prior)
            for lag in [2,4,6]:
                f[f'previous_national_residual_{lag}']=national.get(year-lag,np.nan)
            records.append(f)
        # Updating after the whole cycle prevents same-year cross-race leakage.
        for row in group.itertuples():
            history.setdefault((row.chamber,row.race),[]).append(row)
            for party,name,sign in [('D',row.democrat,1),('R',row.republican,-1)]:
                candidates.setdefault((row.state,row.chamber,party,normalize(name)),[]).append((year,sign*(row.observed_vote-row.lagged_pres_vote)))
        house=group[group.chamber=='house']
        national[year]=(house.observed_vote-house.lagged_pres_vote).mean()
    f=pd.DataFrame(records,index=d.index)
    return pd.concat([d,f],axis=1),list(f.columns)+['midterm']
