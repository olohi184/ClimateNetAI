"""Constrained evidence-oriented controller; no LLM or autonomous network operations."""
from dataclasses import dataclass
import math

@dataclass
class Decision:
    action: str
    reason_codes: list
    evidence: dict
    proposed_next_steps: list
    approval_required: bool
    execution_authorized: bool = False

class AgentController:
    def __init__(self, training, features):
        self.features = list(features)
        self.bounds = {f:(float(training[f].min()), float(training[f].max())) for f in features}

    def inspect_inputs(self, row):
        flags=[]
        for f,(lo,hi) in self.bounds.items():
            try: v=float(row[f])
            except (ValueError, TypeError, KeyError): v=float('nan')
            if not math.isfinite(v): flags.append(f'{f}:missing_or_invalid')
            elif v<lo or v>hi: flags.append(f'{f}:out_of_training_range')
        return flags

    def decide(self, row, rbe, *, telemetry_provenance='historical_replay', observation_age_minutes=None):
        flags=self.inspect_inputs(row)
        reasons=[]
        if telemetry_provenance not in ('historical_replay','verified_device'):
            reasons.append('UNVERIFIED_PROVENANCE')
        if observation_age_minutes is not None and observation_age_minutes>60:
            reasons.append('STALE_CONTEXT')
        if flags: reasons.append('OUT_OF_DOMAIN_INPUT')
        state=rbe['state']
        if state=='ABSTAIN': reasons.append('RBE_ABSTAIN')
        elif state=='CAUTION': reasons.append('RBE_CAUTION')
        elif state=='WARM-UP': reasons.append('INSUFFICIENT_HISTORY')
        elif state!='TRUST': reasons.append('UNKNOWN_RBE_STATE')
        if state=='ABSTAIN' or 'UNVERIFIED_PROVENANCE' in reasons or 'STALE_CONTEXT' in reasons:
            action='ABSTAIN'
        elif reasons:
            action='ESCALATE'
        else:
            action='ACCEPT'
        steps={'ACCEPT':['Log prediction and supporting evidence'],
               'ESCALATE':['Request human review','Collect additional independent telemetry'],
               'ABSTAIN':['Withhold operational recommendation','Request verified fresh telemetry and human review']}[action]
        return Decision(action,reasons or ['ALL_CHECKS_PASSED'],
                        {'rbe_state':state,'prior_n':rbe['n_prior'],
                         'prior_mae_db':None if not math.isfinite(rbe['mae']) else rbe['mae'],
                         'prior_coverage':None if not math.isfinite(rbe['coverage']) else rbe['coverage'],
                         'input_flags':flags,'telemetry_provenance':telemetry_provenance,
                         'observation_age_minutes':observation_age_minutes,
                         'training_feature_bounds':self.bounds},steps,
                        approval_required=action!='ACCEPT')
