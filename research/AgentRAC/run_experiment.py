"""Offline AgentRAC feasibility study. Research prototype, not deployed network control."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rbe import compute_rbe_state, RBE_HALF_WIDTH_DB, RBE_ERROR_THRESHOLD_DB

ROOT = Path(__file__).resolve().parent
FEATURES = ['Temp_C','Pressure_Clean_hPa','RH_pct']
TRAIN_MONTHS = 8  # Jul 2024-Feb 2025; Mar-Jun 2025 untouched holdout

class EvidenceAgent:
    """Deterministic, auditable tool-using decision controller; no LLM is used."""
    def __init__(self, training):
        self.bounds = {f:(float(training[f].min()),float(training[f].max())) for f in FEATURES}
    def check_context(self, row):
        issues=[]
        for f,(lo,hi) in self.bounds.items():
            value=float(row[f])
            if not np.isfinite(value) or value < lo or value > hi:
                issues.append(f'{f}:outside_training_range')
        return issues
    def decide(self, row, reliability):
        # tool 1: check_context; tool 2: reliability state from frozen RBE
        issues=self.check_context(row)
        state=reliability['state']
        if state=='ABSTAIN':
            return 'ABSTAIN','RBE_ABSTAIN',issues
        if issues or state in ('WARM-UP','CAUTION'):
            return 'ESCALATE','CONTEXT_SHIFT_OR_INSUFFICIENT_RELIABILITY',issues
        return 'ACCEPT','RBE_TRUST_AND_CONTEXT_IN_RANGE',issues

def main():
    df=pd.read_csv(ROOT/'rac5g_master_dataset_v1.csv')
    assert df.shape[0]==252
    assert df[FEATURES+['RSSI_dBm']].notna().all().all()
    month_order=list(df['Month_Label'].drop_duplicates())
    train_months=month_order[:TRAIN_MONTHS]
    test_months=month_order[TRAIN_MONTHS:]
    train=df[df.Month_Label.isin(train_months)].copy()
    test=df[df.Month_Label.isin(test_months)].copy()
    # Preserve source order within each representative month; no full timestamps exist.
    model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    model.fit(train[FEATURES],train['RSSI_dBm'])
    agent=EvidenceAgent(train)
    history=[]
    rows=[]
    for _,r in test.iterrows():
        # prospective: decision is made before observed RSSI enters history
        pred=float(model.predict(pd.DataFrame([r[FEATURES].to_dict()]))[0])
        state=compute_rbe_state(history)
        issues=agent.check_context(r)
        agent_action,reason,issues=agent.decide(r,state)
        baseline_action='ACCEPT'
        rbe_action='ACCEPT' if state['state']=='TRUST' else 'ABSTAIN'
        obs=float(r.RSSI_dBm)
        err=abs(obs-pred)
        covered=int(err<=RBE_HALF_WIDTH_DB)
        rows.append(dict(record_id=int(r.Record_ID),month=r.Month_Label,observed_rssi=obs,
                         predicted_rssi=pred,absolute_error_db=err,interval_covered=covered,
                         rbe_state=state['state'],baseline_action=baseline_action,
                         rbe_action=rbe_action,agent_action=agent_action,
                         agent_reason=reason,context_flags=';'.join(issues),
                         rbe_prior_n=int(state['n_prior'])))
        history.append({'Absolute_Error_dB':err,'Interval_Covered':covered})
    result=pd.DataFrame(rows)
    result.to_csv(ROOT/'decision_audit.csv',index=False)
    y=result.observed_rssi.to_numpy(); pred=result.predicted_rssi.to_numpy()
    summary={'status':'offline_exploratory_not_external_validation',
             'train_months':train_months,'test_months':test_months,
             'train_n':len(train),'test_n':len(test),
             'prediction_mae_db':float(mean_absolute_error(y,pred)),
             'prediction_rmse_db':float(np.sqrt(mean_squared_error(y,pred))),
             'prediction_r2':float(r2_score(y,pred)),
             'interval_coverage':float(result.interval_covered.mean()),
             'high_error_definition':f'absolute error > {RBE_ERROR_THRESHOLD_DB} dB',
             'policies':{}}
    for policy in ['baseline','rbe','agent']:
        col=f'{policy}_action'
        accepted=result[result[col]=='ACCEPT']
        summary['policies'][policy]={
            'counts':result[col].value_counts().to_dict(),
            'acceptance_coverage':float(len(accepted)/len(result)),
            'accepted_mae_db':float(accepted.absolute_error_db.mean()) if len(accepted) else None,
            'accepted_high_error_rate':float((accepted.absolute_error_db>RBE_ERROR_THRESHOLD_DB).mean()) if len(accepted) else None,
            'accepted_n':len(accepted)}
    (ROOT/'results_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
