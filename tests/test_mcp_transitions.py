import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp_server'))
from server import state_filter, REGIMES, POLICIES

def test_all_transitions():
    test_sequence = [
        ('code_request', 1, 1),
        ('code_request', 1, 1),
        ('error_log', 3, 2),
        ('error_log', 3, 2),
        ('test_output', 4, 3),
        ('architecture_choice', 5, 3),
        ('math_query', 2, 1),
        ('general_chat', 0, 0),
        ('confirmation', 5, 3)
    ]

    print(f"{'Input Observation':<22} | {'Regime Index':<12} | {'Regime Name':<16} | {'Confidence':<10} | {'Policy Action'}")
    print("-" * 80)
    for obs, expected_regime, expected_policy in test_sequence:
        state = state_filter.observe(obs)
        policy = state_filter.prescribe_policy()
        dom = state['regime_index']
        conf = state['confidence']
        act = policy['action_index']
        regime_name = REGIMES[dom].split()[0]
        policy_name = POLICIES[act].split()[0]
        print(f"{obs:<22} | {dom:<12} | {regime_name:<16} | {conf:>6.1f}%    | {act}: {policy_name}")
        assert dom == expected_regime, f"Expected regime {expected_regime}, got {dom}"
        assert act == expected_policy, f"Expected policy {expected_policy}, got {act}"

    print("-" * 80)
    print(">>> ALL 6 COGNITIVE REGIMES & 4 ACTION POLICIES TRANSITIONED PERFECTLY! <<<")

if __name__ == '__main__':
    test_all_transitions()
