"""Combine disclosed identity repair and domain sensitivity for review only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]

def main():
    out=BASE/'results/review_draft';out.mkdir(exist_ok=True)
    base=pd.read_csv(BASE/'results/structural_details_PROVISIONAL.csv').set_index('race_id')
    extended=pd.read_csv(BASE/'results/domain_sensitivity/structural_details_PROVISIONAL.csv').set_index('race_id')
    np.testing.assert_allclose(base.p_dem_two_party,extended.loc[base.index,'p_dem_two_party'],atol=1e-13,rtol=0)
    aligned=pd.read_csv(BASE/'results/structural_details_IDENTITY_SENSITIVITY.csv').set_index('race_id')
    for col in ['raw_logit','predicted_dem_two_party_share','p_dem_two_party','lo95','hi95']:
        extended.loc[aligned.index,col]=aligned[col]
    extended.reset_index().to_csv(out/'conditional_model_details.csv',index=False)
    aliases=pd.read_csv(BASE/'results/ballot_name_crosswalk.csv').set_index('race_id')
    registry=pd.read_csv(BASE/'results/ballot_candidates.csv');pred=[]
    for key,r in extended.iterrows():
        party='D' if r.p_dem_two_party>.5 else 'R';prob=r.p_dem_two_party if party=='D' else 1-r.p_dem_two_party
        if key in aliases.index:
            a=aliases.loc[key];assert a.party==party and pd.notna(a.ballot_name);name=a.ballot_name
        else:
            g=registry[registry.race_id.eq(key)&registry.party_code.eq(party)&registry.write_in.eq(0)]
            if len(g)!=1:raise ValueError('Cannot infer individual from party: '+key)
            name=g.iloc[0].candidate_name
        pred.append(dict(race_id=key,predicted_winner_party=party,candidate_name=name,win_probability=prob))
    p=pd.DataFrame(pred)
    for prefix in ['full_congress','senate']:
        t=pd.read_csv(BASE/'inputs'/f'{prefix}_template.csv')
        t[['race_id','chamber']].merge(p,on='race_id',how='left',validate='one_to_one').to_csv(out/f'{prefix}_REVIEW_ONLY.csv',index=False)
    cov=pd.read_csv(BASE/'results/domain_sensitivity/coverage.csv').fillna('')
    alias_races=set(pd.read_csv(BASE/'results/candidate_history_alias_corrections.csv').race_id)
    cov['candidate_history_corrected']=cov.race_id.isin(alias_races)
    cov.to_csv(out/'coverage_and_limitations.csv',index=False)
    summary=dict(submission_ready=False,computed_rows=len(p),uncomputed_rows=len(cov)-len(p),house_computed=int(extended.chamber.eq('house').sum()),senate_computed=int(extended.chamber.eq('senate').sum()),domain_extension_races=sorted(set(extended.index)-set(base.index)),identity_corrected_races=sorted(alias_races),operational_model_changed=False,scope='Conditional D/R structural probabilities only. Six extra races are unvalidated extrapolations. No new historical accuracy claim. No joint seat or control forecast. No organizer market snapshot.')
    (out/'readiness.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
