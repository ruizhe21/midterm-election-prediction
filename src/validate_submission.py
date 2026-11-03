"""Fail closed on incomplete or malformed competition CSV files."""
import pandas as pd,numpy as np

def validate(predictions,template):
    errors=[];cols=['race_id','chamber','predicted_winner_party','candidate_name','win_probability']
    if predictions.columns.tolist()!=cols:errors.append('Column names/order do not match the organizer template.')
    if not set(cols).issubset(predictions.columns):return errors
    if predictions.race_id.tolist()!=template.race_id.tolist():errors.append('Race coverage or ordering differs from the template.')
    if predictions.chamber.tolist()!=template.chamber.tolist():errors.append('Chamber labels differ from template.')
    if predictions.race_id.duplicated().any():errors.append('Duplicate race identifiers.')
    missing=predictions[cols[2:]].replace(r'^\s*$' ,np.nan,regex=True).isna().any(axis=1)
    if missing.any():errors.append(f'{int(missing.sum())} races have missing predictions.')
    p=pd.to_numeric(predictions.win_probability,errors='coerce')
    if not (np.isfinite(p)&p.between(0,1)).all():errors.append('Probabilities must be finite numbers in [0,1].')
    return errors

if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser();ap.add_argument('predictions');ap.add_argument('template');a=ap.parse_args()
    errors=validate(pd.read_csv(a.predictions),pd.read_csv(a.template))
    print(json.dumps({'valid':not errors,'errors':errors},indent=2));raise SystemExit(bool(errors))
