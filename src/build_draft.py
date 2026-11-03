"""Provisional structural outputs, explicitly withholding unsupported contest types.

This is not a complete or submission-ready 470/35-race forecast. Export only
the supported D/R conditional calculation, with every missing input disclosed.
"""
from pathlib import Path
import sys,json,io
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1]
from model import predict,fit_all
from history import histories

def race_key(code,template):
    if code in template:return code
    if code.endswith('-01') and code[:2]+'-AL' in template:return code[:2]+'-AL'
    return code

def main(extend_domain=False):
    out=BASE/'results'/('domain_sensitivity' if extend_domain else '')
    out.mkdir(parents=True,exist_ok=True)
    t=pd.read_csv(BASE/'inputs/full_congress_template.csv');valid=set(t.race_id)
    roster=pd.read_csv(BASE/'inputs/electindex_races.csv').fillna('');roster['race_id']=roster.race_code.map(lambda c:race_key(c,valid));roster=roster[roster.race_id.isin(valid)].copy()
    assert len(roster)==470 and not roster.race_id.duplicated().any()
    pres=pd.read_csv(BASE/'inputs/downballot_pres2024_maps2026.csv',header=None).iloc[4:].copy();pres['race_id']=pres[0];pres['state']=pres[0].str[:2]
    for c in [4,5]:pres[c]=pd.to_numeric(pres[c].str.replace(',',''))
    assert set(pres.race_id)==set(t[t.chamber.eq('House')].race_id)
    pres['share']=pres[4]/(pres[4]+pres[5]);pres=pres.set_index('race_id')
    # The sheet's absent 2020 recalculation marks states whose predecessor
    # geography is not confirmed here. Do not pretend all districts changed.
    uncertain_states=set(pres[pres[11].isna()].state)
    state_pres=pres.groupby('state')[[4,5]].sum();state_pres['share']=state_pres[4]/state_pres[[4,5]].sum(axis=1)
    hist=pd.read_csv(BASE/'inputs/history_panel.csv');latest=hist[hist.Year.eq(2024)].set_index(['chamber','race'])
    rows=[];coverage=[];features_provenance=[]
    # Cycles with a new statewide election after the 2024 input snapshot.
    new_gov={'DE','IN','MO','MT','NH','NC','ND','UT','VT','WA','WV','NJ','VA'}
    new_sen=set(hist[hist.Year.eq(2024)&hist.chamber.eq('senate')].state)|{'ME','VT','NE','AK'}
    for r in roster.itertuples():
        ch=r.race_type;key=r.race_id;state=r.state
        oldkey=state if ch=='senate' else (state+'-AL' if key.endswith('-AL') else state+'-'+str(int(key.split('-')[1])))
        reason=[]
        if r.same_party:reason.append('same_party_candidate_model_missing')
        if r.threeway:reason.append('competitive_independent_or_threeway_model_missing')
        if not r.dem_name or r.dem_name.startswith('(') or not r.rep_name or r.rep_name.startswith('('):reason.append('non_DR_or_no_major_party_opponent')
        if r.dem_party!='DEM' or r.rep_party!='REP':reason.append('non_DR_candidate_party')
        # Keep previously excluded electoral-system domains out of the NN release.
        outside=(state=='LA' or (ch=='house' and state in {'AK','ME'}) or (ch=='senate' and state in {'AK','GA'}))
        # A separate, explicitly unvalidated extension: use the SAME fitted
        # model for final D/R outcomes in ME House and GA/LA Senate. LA House
        # and AK remain multiperson/RCV problems; no party-to-person shortcut.
        allow_domain=extend_domain and ((ch=='house' and state=='ME') or (ch=='senate' and state in {'GA','LA'}))
        if outside and not allow_domain:reason.append('electoral_system_outside_historical_evaluation')
        special=key in {'FL-SEN','OH-SEN'}
        if special and not extend_domain:reason.append('special_election_outside_training_domain')
        coverage.append(dict(race_id=key,chamber=ch,democrat=r.dem_name,republican=r.rep_name,independent=r.ind_name,model_status='withheld' if reason else 'provisional_DR_calculation',reason=';'.join(reason),candidate_source='ElectIndex curated roster; not a complete primary-source identity audit',candidate_status_D=r.dem_status,candidate_status_R=r.rep_status,map_status='predecessor_unverified_in_changed_state' if ch=='house' and state in uncertain_states else 'no_change_assumed_from_current_source_coverage',source_url='https://github.com/ElectIndex/26_us_forecast_data/blob/main/races.csv'))
        if extend_domain:coverage[-1]['domain_extension']='unvalidated_same_model_extrapolation' if allow_domain or special else 'original_domain'
        if reason:continue
        inc=1 if r.incumbent_party=='DEM' else (-1 if r.incumbent_party=='REP' else 0)
        x=dict(Year=2026,chamber=ch,state=state,race=oldkey,race_id=key,democrat=r.dem_name,republican=r.rep_name,observed_vote=np.nan,inc=inc,open_seat=int(inc==0),president_D=0,midterm=1,midterm_signed=-1,lagged_pres_vote=pres.loc[key,'share'] if ch=='house' else state_pres.loc[state,'share'],lagged_house_vote=np.nan,lagged_sen_vote=np.nan,lagged_gov_vote=np.nan)
        old=latest.loc[(ch,oldkey)] if (ch,oldkey) in latest.index else None
        mapping_unknown=ch=='house' and state in uncertain_states
        # Conservative missing-input policy; never substitute state returns for
        # district-level Senate/governor results. Median imputation is existing.
        if old is not None and not mapping_unknown:
            if ch=='house':x['lagged_house_vote']=old.observed_vote
            if state not in new_gov:x['lagged_gov_vote']=old.lagged_gov_vote
            if ch=='house' and state not in new_sen:x['lagged_sen_vote']=old.lagged_sen_vote
        if ch=='senate':
            # FL/OH specials fill Class III seats last regularly elected in
            # 2022. Using 2020 would attach the other Senate seat's history.
            prior_year=2022 if special else 2020
            same=hist[hist.chamber.eq('senate')&hist.race.eq(state)&hist.Year.eq(prior_year)]
            if len(same)==1:x['lagged_sen_vote']=same.iloc[0].observed_vote
        for c in ['lagged_pres_vote','lagged_house_vote','lagged_sen_vote','lagged_gov_vote']:
            status='missing_imputed_by_frozen_pipeline' if pd.isna(x[c]) else ('2024_presidential_votes_reconstructed_on_2026_map' if c=='lagged_pres_vote' else 'past_observed_vote_or_no_intervening_event_carryforward')
            features_provenance.append(dict(race_id=key,feature=c,value=x[c],status=status))
        rows.append(x)
    new=pd.DataFrame(rows);allrows=pd.concat([hist,new],ignore_index=True).sort_values(['Year','chamber','race']).reset_index(drop=True)
    d,_=histories(allrows);d=d[d.Year.eq(2026)].copy()
    # Geography is not certified: clear only derived old-district history in
    # this explicitly provisional calculation, not in the frozen training data.
    mask=d.chamber.eq('house')&d.state.isin(uncertain_states)
    cols=[c for c in d if c.startswith(('own_lag_','overperformance_lag_','ewm_'))]
    d.loc[mask,cols]=np.nan;d.loc[mask,'history_count']=0
    # A change indicator is not invented from unknown geometry. Existing source
    # change flags are retained; uncertainty is in the accompanying ledger.
    feature_file=BASE/'inputs'/('features_2026_DOMAIN_SENSITIVITY.csv' if extend_domain else 'features_2026_PROVISIONAL.csv')
    d.to_csv(feature_file,index=False);d=pd.read_csv(feature_file)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False);pd.DataFrame(features_provenance).to_csv(out/'feature_provenance.csv',index=False)
    p=predict(d);p.to_csv(out/'structural_details_PROVISIONAL.csv',index=False)
    w=p.copy();win=w.p_dem_two_party>.5;w['predicted_winner_party']=np.where(win,'D','R');w['candidate_name']=np.where(win,w.democrat,w.republican);w['win_probability']=np.where(win,w.p_dem_two_party,1-w.p_dem_two_party)
    full=t[['race_id','chamber']].merge(w[['race_id','predicted_winner_party','candidate_name','win_probability']],on='race_id',how='left',validate='one_to_one')
    full.to_csv(out/'full_congress_DRAFT_NOT_FOR_SUBMISSION.csv',index=False)
    st=pd.read_csv(BASE/'inputs/senate_template.csv');st[['race_id','chamber']].merge(w[['race_id','predicted_winner_party','candidate_name','win_probability']],on='race_id',how='left',validate='one_to_one').to_csv(out/'senate_DRAFT_NOT_FOR_SUBMISSION.csv',index=False)
    full[full.candidate_name.notna()].to_csv(out/'available_structural_picks_PROVISIONAL.csv',index=False)
    (out/'readiness.json').write_text(json.dumps(dict(submission_ready=False,domain_extension=extend_domain,template_rows=len(full),filled_rows=int(full.candidate_name.notna().sum()),senate_filled=int(full[full.chamber.eq('Senate')].candidate_name.notna().sum()),withheld_rows=int(full.candidate_name.isna().sum()),benchmark_available=False,current_polls_used=False,limitations=['Conditional D/R probabilities do not exhaust multiparty outcomes.','Historical raw lag inputs remain incomplete; fixed median imputation applied.','Geography uncertainty and candidate identity are not fully audited.','Unsupported races left blank intentionally; these are not valid complete submissions.','No new current polling or competition-optimal picks.']+(['New-domain probabilities are unvalidated extrapolations; historical scores do not apply to the enlarged cohort.'] if extend_domain else [])),indent=2))
    print(full.groupby('chamber').candidate_name.agg(['size','count']).to_string())

if __name__=='__main__':main('--extend-domain' in sys.argv)
