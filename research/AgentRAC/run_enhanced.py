"""Replay the frozen baseline; evaluate safety-policy behavior and stress cases separately."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from rbe import compute_rbe_state, RBE_HALF_WIDTH_DB, RBE_ERROR_THRESHOLD_DB
from agent_controller import AgentController
ROOT=Path(__file__).resolve().parent
FEATURES=['Temp_C','Pressure_Clean_hPa','RH_pct']

def run():
    df=pd.read_csv(ROOT/'rac5g_master_dataset_v1.csv')
    months=list(df.Month_Label.drop_duplicates())
    assert len(months)==12 and len(df)==252
    train=df[df.Month_Label.isin(months[:8])].copy()
    test=df[df.Month_Label.isin(months[8:])].copy()
    model=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(train[FEATURES],train.RSSI_dBm)
    agent=AgentController(train,FEATURES)
    history=[]; audit=[]; stresses=[]
    for _,row in test.iterrows():
        rbe=compute_rbe_state(history) # no current outcome leakage
        prediction=float(model.predict(pd.DataFrame([row[FEATURES].to_dict()]))[0])
        decision=agent.decide(row,rbe)
        error=abs(float(row.RSSI_dBm)-prediction)
        covered=int(error<=RBE_HALF_WIDTH_DB)
        audit.append({'record_id':int(row.Record_ID),'month':row.Month_Label,'predicted_rssi_dbm':prediction,
                      'observed_rssi_dbm':float(row.RSSI_dBm),'absolute_error_db':error,
                      'interval_covered':covered,'rbe_state':rbe['state'],
                      'baseline_action':'ACCEPT',
                      'rbe_action':'ACCEPT' if rbe['state']=='TRUST' else 'ABSTAIN',
                      'agent_action':decision.action,'reason_codes':'|'.join(decision.reason_codes),
                      'approval_required':decision.approval_required,
                      'execution_authorized':decision.execution_authorized,
                      'evidence_json':json.dumps(decision.evidence,sort_keys=True),
                      'next_steps_json':json.dumps(decision.proposed_next_steps)})
        for scenario,kwargs,changed in [
            ('unverified_provenance',{'telemetry_provenance':'unverified_source'},row.copy()),
            ('stale_context',{'observation_age_minutes':180},row.copy()),
            ('out_of_range_temperature',{},row.copy())]:
            if scenario=='out_of_range_temperature':
                changed['Temp_C']=agent.bounds['Temp_C'][1]+10
            d=agent.decide(changed,rbe,**kwargs)
            stresses.append({'record_id':int(row.Record_ID),'scenario':scenario,'action':d.action,
                             'reason_codes':'|'.join(d.reason_codes),'approval_required':d.approval_required,
                             'execution_authorized':d.execution_authorized})
        history.append({'Absolute_Error_dB':error,'Interval_Covered':covered})
    a=pd.DataFrame(audit); s=pd.DataFrame(stresses)
    a.to_csv(ROOT/'enhanced_decision_audit.csv',index=False)
    s.to_csv(ROOT/'policy_stress_tests.csv',index=False)
    policies={}
    for name in ['baseline','rbe','agent']:
        accepted=a[a[f'{name}_action']=='ACCEPT']
        policies[name]={'counts':a[f'{name}_action'].value_counts().to_dict(),
                        'accepted_n':len(accepted),'acceptance_rate':len(accepted)/len(a),
                        'accepted_mae_db':float(accepted.absolute_error_db.mean()) if len(accepted) else None,
                        'accepted_high_error_rate':float((accepted.absolute_error_db>RBE_ERROR_THRESHOLD_DB).mean()) if len(accepted) else None}
    checks={
        'unverified_sources_always_abstain':bool((s[s.scenario=='unverified_provenance'].action=='ABSTAIN').all()),
        'stale_context_always_abstain':bool((s[s.scenario=='stale_context'].action=='ABSTAIN').all()),
        'out_of_range_never_accepted':bool((s[s.scenario=='out_of_range_temperature'].action!='ACCEPT').all()),
        'no_automatic_execution':bool((~a.execution_authorized).all() and (~s.execution_authorized).all()),
        'all_nonaccept_decisions_require_approval':bool(a.loc[a.agent_action!='ACCEPT','approval_required'].all()),
    }
    result={'study_type':'retrospective_offline_replay_and_synthetic_policy_stress_tests',
            'train_months':months[:8],'test_months':months[8:],'train_n':len(train),'test_n':len(test),
            'holdout_r2':float(r2_score(a.observed_rssi_dbm,a.predicted_rssi_dbm)),
            'holdout_mae_db':float(mean_absolute_error(a.observed_rssi_dbm,a.predicted_rssi_dbm)),
            'policies':policies,'stress_scenario_counts':s.groupby(['scenario','action']).size().to_dict().__str__(),
            'policy_checks':checks,
            'limitations':['No independently verified real-time deployment','Representative records lack complete timestamps',
                           'Stress tests validate programmed rules, not improved field safety',
                           'Agent is deterministic tool orchestration, not an LLM',
                           'Fixed RBE thresholds inherited from prior work; prospective recalibration not performed']}
    (ROOT/'enhanced_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert all(checks.values())
if __name__=='__main__':run()
