"""Isolate 2026 candidate-history linkage corrections; frozen model parameters.

This is a deployment input sensitivity, not a new historical accuracy claim.
Aliases below were reviewed with current ballot names, stable member IDs and
the existing legislative membership snapshot. No winner outcome is inspected.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from history import normalize
from model import predict
BASE=Path(__file__).resolve().parents[1]
ALIASES=[
 ('CA','D','Amerish Bera','Ami Bera','B001287'),
 ('CA','D','Rohit Khanna','Ro Khanna','K000389'),
 ('CA','D','Gilbert Cisneros','Gil Cisneros','C001123'),
 ('GA','D','Henry Johnson','Hank Johnson','J000288'),
 ('FL','R','Maria Salazar','María Elvira Salazar','S000168'),
 ('FL','R','Carlos Gimenez','Carlos A. Giménez','G000593'),
 ('IN','D','Frank Mrvan','Frank J. Mrvan','M001214'),
 ('KY','R','Harold Rogers','Hal Rogers','R000395'),
 ('NJ','R','Tom Kean Jr.','Thomas Kean Jr.','K000398'),
 ('NM','D','Gabriel Vasquez','Gabe Vasquez','V000136'),
 ('NY','D','Joe Morelle','Joseph Morelle','M001206'),
 ('NY','D','Timothy Kennedy','Tim Kennedy','K000402'),
 ('RI','D','Gabriel Amo','Gabe Amo','A000380'),
 ('VT','D','Rebecca Balint','Becca Balint','B001318'),
]

def main():
    d=pd.read_csv(BASE/'inputs/features_2026_PROVISIONAL.csv');h=pd.read_csv(BASE/'inputs/history_panel.csv');new=d.copy();log=[]
    assert h.Year.max()==2024 and d.Year.eq(2026).all()
    for state,party,old,name,bio in ALIASES:
        col='democrat' if party=='D' else 'republican';sgn=1 if party=='D' else -1
        mask=d.chamber.eq('house')&d.state.eq(state)&d[col].map(normalize).eq(normalize(old))
        prior=h[h.chamber.eq('house')&h.state.eq(state)&h[col].map(normalize).eq(normalize(name))&h.Year.lt(2026)]
        if not mask.any():continue
        if prior.empty:raise ValueError('Historical alias not found: '+state+' '+name)
        w=2.**(-(2026-prior.Year.to_numpy())/8);v=sgn*(prior.observed_vote-prior.lagged_pres_vote)
        n=len(prior);past=float(np.average(v,weights=w)*n/(n+3))
        new.loc[mask,f'candidate_{party}_n']=n;new.loc[mask,f'candidate_{party}_past']=past
        for i in d.index[mask]:
            log.append(dict(race_id=d.loc[i,'race_id'],party=party,current_name=old,historical_name=name,bioguide_id=bio,old_history_n=int(d.loc[i,f'candidate_{party}_n']),corrected_history_n=n,old_past=float(d.loc[i,f'candidate_{party}_past']),corrected_past=past,earliest_history_year=int(prior.Year.min()),latest_history_year=int(prior.Year.max()),official_member_url=f'https://bioguide.congress.gov/search/bio/{bio}',status='reviewed_alias_deployment_sensitivity'))
    changed=[c for c in d if not d[c].equals(new[c])];assert set(changed)<=set(['candidate_D_n','candidate_D_past','candidate_R_n','candidate_R_past'])
    before=predict(d);after=predict(new)
    delta=before[['race_id','p_dem_two_party','raw_logit']].merge(after[['race_id','p_dem_two_party','raw_logit']],on='race_id',suffixes=('_before','_aligned'),validate='one_to_one')
    delta['probability_change_pp']=100*(delta.p_dem_two_party_aligned-delta.p_dem_two_party_before)
    delta['winner_changed']=(delta.p_dem_two_party_before>.5)!=(delta.p_dem_two_party_aligned>.5)
    affected=delta[delta.race_id.isin([r['race_id'] for r in log])].copy()
    pd.DataFrame(log).to_csv(BASE/'results/candidate_history_alias_corrections.csv',index=False)
    affected.to_csv(BASE/'results/candidate_identity_sensitivity.csv',index=False)
    new.to_csv(BASE/'inputs/features_2026_IDENTITY_SENSITIVITY.csv',index=False)
    after.to_csv(BASE/'results/structural_details_IDENTITY_SENSITIVITY.csv',index=False)
    summary=dict(corrected_races=len(log),old_zero_history=int(sum(r['old_history_n']==0 for r in log)),winner_changes=int(affected.winner_changed.sum()),max_absolute_probability_change_pp=float(affected.probability_change_pp.abs().max()),median_absolute_probability_change_pp=float(affected.probability_change_pp.abs().median()),changed_features=changed,model_refit=False,historical_scores_changed=False,operational_default_changed=False,limitation='Input-only 2026 sensitivity. It does not validate full history linkage, missing lags, joint uncertainty, or unsupported election types.')
    (BASE/'results/candidate_identity_sensitivity_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
