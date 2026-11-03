"""Freeze the reviewed model and transparently complete all organizer rows."""
from pathlib import Path
import sys, json, re, hashlib, unicodedata
import numpy as np
import pandas as pd
from scipy.stats import norm
from lxml import html
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / 'model_snapshot'
sys.path.insert(0, str(SNAP / 'src'))
from validate_submission import validate

def normalize(s):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower())

def california_votes():
    d = pd.read_excel(ROOT/'inputs/official/ca_primary_house_2026.xlsx', sheet_name='Working Data', header=None)
    columns = {}; district = None
    for j in range(2, d.shape[1]):
        if pd.notna(d.iloc[0,j]): district = int(d.iloc[0,j])
        columns.setdefault(district, []).append(j)
    totals = {}; district = None
    for i in range(3, d.shape[0]):
        if pd.notna(d.iloc[i,0]): district = int(d.iloc[i,0])
        if d.iloc[i,1] == 'District Totals':
            totals[district] = {normalize(d.iloc[1,j]): int(d.iloc[i,j]) for j in columns[district]}
    return totals

def main():
    policy = json.loads((ROOT/'inputs/completion_policy.json').read_text())
    template = pd.read_csv(SNAP/'inputs/full_congress_template.csv')
    original = pd.read_csv(SNAP/'results/review_draft/full_congress_REVIEW_ONLY.csv')
    candidates = pd.read_csv(SNAP/'results/ballot_candidates.csv').fillna('')
    ready = json.loads((SNAP/'results/review_draft/readiness.json').read_text())
    final = original.copy().set_index('race_id')
    audit = []; distributions = []; evidence = []
    missing = set(original.loc[original.win_probability.isna(),'race_id'])
    assert len(missing) == 31

    def registry(race): return candidates[candidates.race_id.eq(race)].copy()
    def commit(race, name, q, method, reason, source, distribution=None):
        assert race in missing, f'Cannot override reviewed model: {race}'
        g = registry(race); match = g[g.candidate_name.eq(name)]
        assert len(match) == 1, (race, name)
        row = match.iloc[0]
        final.loc[race,['candidate_name','predicted_winner_party','win_probability']] = [name,row.party_code,float(q)]
        audit.append(dict(race_id=race, forecast_method=method, probability_basis='unvalidated_completion_assumption', reason=reason,
                          candidate_source_url=row.source_url, evidence_url=source, candidate_evidence_status=row.evidence_status))
        if distribution is None:
            distribution = {name:float(q), '__all_other_outcomes__':1-float(q)}
        assert abs(sum(distribution.values())-1)<1e-10
        assert q >= max(v for k,v in distribution.items() if k!=name) or '__all_other_outcomes__' in distribution
        for candidate,probability in distribution.items():
            distributions.append(dict(race_id=race,candidate_name=candidate,probability=probability,
                                      meaning='residual aggregate' if candidate.startswith('__') else 'candidate win'))

    # Rules use only already completed 2026 primary/special elections.
    prim = policy['primary_rule']; ca = california_votes()
    ca_url = 'https://elections.cdn.sos.ca.gov/sov/2026-primary/sov/76-us-rep.xlsx'
    def pair_rule(race, counts, source, event_date, residual=0.0):
        ordered = sorted(counts, key=counts.get, reverse=True)
        a,b = ordered; share=counts[a]/sum(counts.values()); margin=2*share-1
        q=min(prim['probability_cap'],float(norm.cdf(prim['margin_shrinkage']*margin/prim['margin_sd'])))
        dist={a:(1-residual)*q,b:(1-residual)*(1-q)}
        if residual: dist['__all_other_outcomes__']=residual
        text=f'Completed {event_date} election: {a} {counts[a]:,}; {b} {counts[b]:,}. Half-margin normal rule; SD 15 pp; cap 95%; other-outcome reserve {residual:.0%}. No transfer model fitted.'
        commit(race,a,dist[a],'completed_election_fallback',text,source,dist)
        for name,votes in counts.items():evidence.append(dict(race_id=race,candidate_name=name,votes=votes,event_date=event_date,source_url=source))
    for district in prim['california_districts']:
        race=f'CA-{district:02d}'; g=registry(race)
        assert len(g)==2
        counts={name:ca[district][normalize(name)] for name in g.candidate_name}
        source=ca_url; event='2026-06-02'
        if district==14:
            tables=pd.read_html(str(ROOT/'inputs/official/ca14_special.html'))
            table=next(t for t in tables if any('Candidates' in str(c) for c in t.columns))
            counts={}
            for name in g.candidate_name:
                row=table[table.iloc[:,0].astype(str).str.startswith(name)].iloc[0]
                counts[name]=int(str(row.iloc[2]).replace(',',''))
            source='https://www.sos.ca.gov/elections/prior-elections/special-elections/2026-cd14/official-canvass';event='2026-08-18'
        pair_rule(race,counts,source,event)
    ak_pdf=PdfReader(ROOT/'inputs/official/ak_primary_summary.pdf')
    for race,page,pairs in [
        ('AK-SEN',0,{'Mary Peltola':'Peltola, Mary','Dan S. Sullivan':'Sullivan, Dan S.'}),
        ('AK-AL',1,{'Nick Begich':'Begich, Nick','Bill Hill':'Hill, Bill'})]:
        text=' '.join(ak_pdf.pages[page].extract_text().split())
        assert '2026' in ' '.join(p.extract_text() for p in ak_pdf.pages[:2])
        counts={}
        for name,ballot in pairs.items():
            match=re.search(re.escape(ballot)+r'\s+(?:DEM|REP|NON)\s+([\d,]+)\s+',text)
            assert match, name
            counts[name]=int(match.group(1).replace(',',''))
        pair_rule(race,counts,'https://www.elections.alaska.gov/enr/26PRIM/results/ElectionSummaryReportRPT.pdf','2026-08-18',prim['alaska_other_winner_mass'])
    for row in policy['judgment_picks']:
        g=registry(row['race_id']); assert g.loc[g.candidate_name.eq(row['candidate_name']),'party_code'].iloc[0]==row['party']
        commit(row['race_id'],row['candidate_name'],row['probability'],row['rule'],row['reason'],g.loc[g.candidate_name.eq(row['candidate_name']),'source_url'].iloc[0])

    # Only two open Louisiana seats use the financial tie-breaking rule.
    money=pd.read_csv(SNAP/'inputs/deepsyte_candidates_2026.csv')
    settings=policy['open_louisiana_rule']; money_log=[]
    for district in [5,6]:
        race=f'LA-{district:02d}'; g=registry(race)
        m=money[money.state.eq('LA')&money.office.eq('house')&money.district.eq(district)].copy()
        m['match_name']=m.candidate.map(normalize)
        dist={}; allocated=0
        for party,mass in [('R',settings['republican_probability']),('D',settings['democratic_probability'])]:
            subgroup=g[g.party_code.eq(party)].copy(); receipts=[]
            for row in subgroup.itertuples():
                matches=m[m.match_name.eq(normalize(row.candidate_name))]
                amount=float(matches.iloc[0].total_receipts_usd) if len(matches)==1 else np.nan
                date=str(matches.iloc[0].fec_report_through) if len(matches)==1 else ''
                if pd.notna(amount):assert pd.Timestamp(date)<=pd.Timestamp(policy['forecast_date'])
                receipts.append(0.0 if pd.isna(amount) else amount)
                money_log.append(dict(race_id=race,candidate_name=row.candidate_name,party=party,receipts_used=receipts[-1],
                                      missing_receipts=bool(pd.isna(amount)),coverage_end_date=date,
                                      source='Saved DeepSyte FEC transcription; finalist totals cross-checked to FEC HTML'))
            weights=np.sqrt(np.array(receipts)+settings['receipt_offset_usd'])
            weights=settings['equal_weight']/len(weights)+(1-settings['equal_weight'])*weights/weights.sum()
            dist.update(dict(zip(subgroup.candidate_name,mass*weights)));allocated+=mass
        other=g[~g.party_code.isin(['R','D'])]
        if len(other):dist.update({n:(1-allocated)/len(other) for n in other.candidate_name})
        else:dist['__all_other_outcomes__']=1-allocated
        winner=max((n for n in dist if not n.startswith('__')),key=dist.get)
        commit(race,winner,dist[winner],'open_multiperson_receipts_rule',settings['description'],
               'https://www.fec.gov/data/candidate/'+('H6LA05242/' if district==5 else 'H6LA05234/'),dist)
    final=final.reset_index()[template.columns]
    errors=validate(final,template); assert not errors,errors
    assert set(a['race_id'] for a in audit)==missing
    available=original.win_probability.notna()
    assert (final.loc[available,'candidate_name'].values==original.loc[available,'candidate_name'].values).all()
    np.testing.assert_array_equal(final.loc[available,'win_probability'],original.loc[available,'win_probability'])
    assert all(((candidates.race_id==row.race_id)&(candidates.candidate_name==row.candidate_name)&(candidates.party_code==row.predicted_winner_party)).any() for row in final.itertuples())
    final.to_csv(ROOT/'full_congress_FINAL.csv',index=False,float_format='%.10f')
    st=pd.read_csv(SNAP/'inputs/senate_template.csv')
    senate=st[['race_id','chamber']].merge(final.drop(columns='chamber'),on='race_id',validate='one_to_one')
    assert not validate(senate,st)
    senate.to_csv(ROOT/'senate_FINAL.csv',index=False,float_format='%.10f')
    print('Complete predictions written: 470 Full Congress rows and 35 Senate rows.')

if __name__=='__main__': main()
