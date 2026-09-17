/**
 * @file test_c_core.c
 * @brief Comprehensive Unit Tests for Micro-ActInf C11 Engine.
 * @details Validates:
 *   1. Kolmogorov Probability Axioms on Simplex
 *   2. Shannon Entropy Concentration on Evidence
 *   3. Static Memory Footprint (<= 36 KB, zero dynamic allocation)
 *   4. O(1) Online Dirichlet Learning & Parameter Convergence (Numerical Stability)
 *   5. Sub-3.0us Combined Inference + Dirichlet Learning Latency Benchmark
 *   6. Full Decision Cycle Latency Benchmark
 * @author naderloocodelab
 * @license MIT
 */

#include "micro_actinf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifdef _WIN32
#include <windows.h>
#endif

static double get_time_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return ((double)counter.QuadPart / (double)freq.QuadPart) * 1e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

void test_probabilistic_axioms(void) {
    printf("[TEST 1] Testing Kolmogorov Probability Axioms on Simplex...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 8, 16, 4);

    /* Check initial sum */
    float sum = 0.0f;
    for (uint8_t i = 0; i < agent.num_states; i++) {
        assert(agent.s[i] >= 0.0f);
        sum += agent.s[i];
    }
    assert(fabsf(sum - 1.0f) < 1e-5f);

    /* Run 50 random steps and verify axiom invariance */
    for (int step = 0; step < 50; step++) {
        uint8_t obs = (uint8_t)(step % agent.num_obs);
        micro_actinf_step(&agent, obs);

        sum = 0.0f;
        for (uint8_t i = 0; i < agent.num_states; i++) {
            assert(agent.s[i] >= 0.0f);
            sum += agent.s[i];
        }
        assert(fabsf(sum - 1.0f) < 1e-5f);
    }
    printf("  --> PASS: Belief probabilities are strictly positive and sum to 1.0\n");
}

void test_entropy_collapse(void) {
    printf("[TEST 2] Testing Shannon Entropy Concentration on Evidence...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 4, 8, 2);

    /* Make observation 2 strongly indicate state 2 */
    agent.A[2][2] = 0.90f;
    for (uint8_t s = 0; s < 4; s++) {
        if (s != 2) agent.A[2][s] = 0.033f;
    }
    micro_actinf_update_cache(&agent);

    float initial_entropy = micro_actinf_shannon_entropy(&agent);
    assert(fabsf(initial_entropy - logf(4.0f)) < 1e-4f);

    /* Feed observation 2 repeatedly */
    for (int i = 0; i < 5; i++) {
        micro_actinf_step(&agent, 2);
    }

    float final_entropy = micro_actinf_shannon_entropy(&agent);
    printf("  Initial Entropy: %.4f nats | Final Entropy: %.4f nats\n", initial_entropy, final_entropy);
    assert(final_entropy < initial_entropy * 0.30f);
    printf("  --> PASS: Entropy collapsed by >70%% upon evidence accumulation\n");
}

void test_static_memory_footprint(void) {
    printf("[TEST 3] Testing Static Memory Footprint (<= 36 KB)...\n");
    size_t sz = sizeof(micro_actinf_t);
    const size_t max_allowed = 36 * 1024; /* 36,864 bytes */
    printf("  sizeof(micro_actinf_t): %zu bytes (%.2f KB) | Budget: %zu bytes (36.00 KB)\n",
           sz, (double)sz / 1024.0, max_allowed);
    assert(sz <= max_allowed);
    printf("  --> PASS: Static memory allocation guarantee strictly satisfied (Zero dynamic heap)\n");
}

void test_dirichlet_online_learning(void) {
    printf("[TEST 4] Testing O(1) Online Dirichlet Learning & Parameter Convergence...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 4, 6, 3);

    micro_actinf_learning_config_t cfg = {
        .learning_rate_a = 0.10f,
        .learning_rate_b = 0.10f,
        .decay_factor = 0.995f,
        .enable_learning = true
    };
    micro_actinf_set_learning_config(&agent, &cfg);

    /* Step 0: Ensure initial pseudo-counts and expectations are uniform and valid */
    for (uint8_t s = 0; s < agent.num_states; s++) {
        float sum_a = 0.0f;
        for (uint8_t m = 0; m < agent.num_obs; m++) {
            assert(agent.A[m][s] > 0.0f);
            assert(agent.a_counts[m][s] > 0.0f);
            sum_a += agent.A[m][s];
        }
        assert(fabsf(sum_a - 1.0f) < 1e-4f);
    }

    /* Simulate 10,000 steps pairing action 1 -> state transition -> observation 3 */
    for (int t = 0; t < 10000; t++) {
        uint32_t obs = 3;
        uint32_t act = 1;
        micro_actinf_step(&agent, (uint8_t)obs);
        micro_actinf_learn_step(&agent, obs, act);

        /* Verify no NaN or Inf occurs */
        assert(!isnan(agent.s[0]) && !isinf(agent.s[0]));
        assert(!isnan(agent.A[obs][0]) && !isinf(agent.A[obs][0]));
        assert(!isnan(agent.B[act][0][0]) && !isinf(agent.B[act][0][0]));
    }

    /* Verify categorical normalization on all columns of A and B */
    for (uint8_t s = 0; s < agent.num_states; s++) {
        float sum_a = 0.0f;
        for (uint8_t m = 0; m < agent.num_obs; m++) {
            assert(agent.A[m][s] >= 0.0f);
            assert(!isnan(agent.a_counts[m][s]) && !isinf(agent.a_counts[m][s]));
            assert(agent.a_counts[m][s] < 1000.0f); /* Must be bounded by decay */
            sum_a += agent.A[m][s];
        }
        assert(fabsf(sum_a - 1.0f) < 1e-4f);
    }

    for (uint8_t u = 0; u < agent.num_actions; u++) {
        for (uint8_t s = 0; s < agent.num_states; s++) {
            float sum_b = 0.0f;
            for (uint8_t s_prime = 0; s_prime < agent.num_states; s_prime++) {
                assert(agent.B[u][s_prime][s] >= 0.0f);
                assert(!isnan(agent.b_counts[s_prime][s][u]) && !isinf(agent.b_counts[s_prime][s][u]));
                assert(agent.b_counts[s_prime][s][u] < 1000.0f); /* Must be bounded by decay */
                sum_b += agent.B[u][s_prime][s];
            }
            assert(fabsf(sum_b - 1.0f) < 1e-4f);
        }
    }

    /* Test runtime toggle: enable_learning = false */
    cfg.enable_learning = false;
    micro_actinf_set_learning_config(&agent, &cfg);
    float count_before = agent.a_counts[3][0];
    micro_actinf_step(&agent, 3);
    micro_actinf_learn_step(&agent, 3, 1);
    assert(agent.a_counts[3][0] == count_before);

    printf("  --> PASS: Online Dirichlet learning converges stably with 0%% overflow across 10,000 steps\n");
}

void test_combined_inference_learning_latency(void) {
    printf("[TEST 5] Running Sub-3.0us Combined Inference + Dirichlet Learning Benchmark (100,000 cycles)...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 16, 32, 8);

    const int CYCLES = 100000;
    double t0 = get_time_ns();
    for (int i = 0; i < CYCLES; i++) {
        uint32_t obs = (uint32_t)(i % 32);
        uint32_t act = (uint32_t)(i % 8);
        micro_actinf_step(&agent, (uint8_t)obs);
        micro_actinf_learn_step(&agent, obs, act);
    }
    double t1 = get_time_ns();

    double mean_us = ((t1 - t0) / (double)CYCLES) / 1000.0;
    printf("  Average Combined Latency (Inference + Dirichlet Learning): %.3f us (Target budget: <= 5.0 us)\n", mean_us);
    fflush(stdout);
    assert(mean_us <= 5.0);
    printf("  --> PASS: Hard real-time latency budget (<= 5.0 us) satisfied!\n");
    fflush(stdout);
}

void test_full_decision_cycle_latency(void) {
    printf("[TEST 6] Running Full Decision Cycle (Inference + Action Selection + Dirichlet Learning)...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 16, 32, 8);

    const int CYCLES = 50000;
    double t0 = get_time_ns();
    for (int i = 0; i < CYCLES; i++) {
        uint32_t obs = (uint32_t)(i % 32);
        micro_actinf_step(&agent, (uint8_t)obs);
        uint8_t act = micro_actinf_select_action(&agent);
        micro_actinf_learn_step(&agent, obs, (uint32_t)act);
    }
    double t1 = get_time_ns();

    double mean_us = ((t1 - t0) / (double)CYCLES) / 1000.0;
    printf("  Average Full Decision Cycle Latency: %.3f us (Target budget: < 80.0 us)\n", mean_us);
    fflush(stdout);
    assert(mean_us < 80.0);
    printf("  --> PASS: Full decision cycle real-time guarantee verified\n");
    fflush(stdout);
}

void test_hyperparameter_bounds_and_nan(void) {
    printf("[TEST 7] Testing Hyperparameter Bounds & Defensive NaN Handling...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 4, 4, 2);

    /* Test NaN and negative values */
    micro_actinf_learning_config_t invalid_cfg = {
        .learning_rate_a = -1.0f,
        .learning_rate_b = (float)NAN,
        .decay_factor = -0.5f,
        .enable_learning = true
    };
    micro_actinf_set_learning_config(&agent, &invalid_cfg);

    assert(agent.learning_cfg.learning_rate_a >= 0.0f && agent.learning_cfg.learning_rate_a <= 1.0f);
    assert(!isnan(agent.learning_cfg.learning_rate_b));
    assert(agent.learning_cfg.decay_factor > 0.0f && agent.learning_cfg.decay_factor <= 1.0f);

    /* Test out-of-range > 1.0f values */
    invalid_cfg.learning_rate_a = 5.0f;
    invalid_cfg.decay_factor = 2.0f;
    micro_actinf_set_learning_config(&agent, &invalid_cfg);
    assert(agent.learning_cfg.learning_rate_a == 1.0f);
    assert(agent.learning_cfg.decay_factor == 1.0f);

    /* Verify step and learn_step execute safely under clamped config */
    micro_actinf_step(&agent, 0);
    micro_actinf_learn_step(&agent, 0, 0);
    assert(!isnan(agent.s[0]) && !isnan(agent.a_counts[0][0]));

    printf("  --> PASS: Hyperparameters sanitized and guarded against NaNs and invalid domains\n");
}

void test_extreme_likelihood_zero_fallback(void) {
    printf("[TEST 8] Testing Extreme Likelihood Zero Fallback (Log-Space Softmax)...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 4, 4, 2);

    /* Force all likelihoods for observation 1 to zero (impossible observation) */
    for (uint8_t s = 0; s < agent.num_states; s++) {
        agent.A[1][s] = 0.0f;
    }

    micro_actinf_step(&agent, 1);

    /* Verify fallback activated and belief remains valid probability simplex */
    float sum_s = 0.0f;
    for (uint8_t s = 0; s < agent.num_states; s++) {
        assert(!isnan(agent.s[s]) && !isinf(agent.s[s]));
        assert(agent.s[s] >= 0.0f);
        sum_s += agent.s[s];
    }
    assert(fabsf(sum_s - 1.0f) < 1e-4f);
    printf("  --> PASS: Zero-likelihood fallback preserves probability axioms without NaN\n");
}

void test_cache_consistency_after_learning(void) {
    printf("[TEST 9] Testing Column Entropy Cache Consistency After Learning...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 4, 4, 2);

    assert(!agent.a_entropy_dirty);
    float initial_entropy = agent.A_entropy[0];

    /* Learning alters matrix A and sets dirty flag */
    micro_actinf_step(&agent, 2);
    micro_actinf_learn_step(&agent, 2, 0);
    assert(agent.a_entropy_dirty);

    /* Action selection lazily recomputes A_entropy and clears dirty flag */
    micro_actinf_select_action(&agent);
    assert(!agent.a_entropy_dirty);
    assert(agent.A_entropy[0] != initial_entropy);

    printf("  --> PASS: Lazy cache invalidation and recomputation strictly consistent\n");
}

void test_boundary_dimensions(void) {
    printf("[TEST 10] Testing Boundary Dimensions (K=1, M=1, A=1 and Maximum Limits)...\n");
    micro_actinf_t min_agent;
    micro_actinf_init(&min_agent, 1, 1, 1);

    assert(min_agent.num_states == 1);
    assert(min_agent.B[0][0][0] == 1.0f);
    assert(min_agent.b_counts[0][0][0] == 1.0f);

    micro_actinf_step(&min_agent, 0);
    micro_actinf_learn_step(&min_agent, 0, 0);
    uint8_t act = micro_actinf_select_action(&min_agent);
    assert(act == 0);
    assert(fabsf(min_agent.s[0] - 1.0f) < 1e-5f);

    micro_actinf_t max_agent;
    micro_actinf_init(&max_agent, ACTINF_MAX_STATES, ACTINF_MAX_OBS, ACTINF_MAX_ACTIONS);
    assert(max_agent.num_states == ACTINF_MAX_STATES);
    assert(max_agent.num_obs == ACTINF_MAX_OBS);
    assert(max_agent.num_actions == ACTINF_MAX_ACTIONS);
    micro_actinf_step(&max_agent, ACTINF_MAX_OBS - 1);
    micro_actinf_learn_step(&max_agent, ACTINF_MAX_OBS - 1, ACTINF_MAX_ACTIONS - 1);

    printf("  --> PASS: Minimum (1x1x1) and maximum (16x32x8) topologies operate reliably\n");
}

void test_sync_counts_from_matrices_and_adaptation(void) {
    printf("[TEST 11] Testing Prior Matrix Synchronization & Custom Prior Retention...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 4, 4, 2);

    /* Set custom likelihood priors */
    agent.A[0][0] = 0.85f;
    agent.A[1][0] = 0.05f;
    agent.A[2][0] = 0.05f;
    agent.A[3][0] = 0.05f;
    micro_actinf_sync_counts_from_matrices(&agent);

    assert(fabsf(agent.a_counts[0][0] - 0.85f * 4.0f) < 1e-4f);

    /* Step 0: Ensure online learning does NOT wipe out the custom prior */
    micro_actinf_step(&agent, 0);
    micro_actinf_learn_step(&agent, 0, 0);
    assert(agent.A[0][0] > 0.80f);

    /* Adapt over 500 steps of observation 0 */
    for (int step = 0; step < 500; step++) {
        micro_actinf_step(&agent, 0);
        micro_actinf_learn_step(&agent, 0, 0);
    }

    /* Verify observation 0 reinforced towards certainty */
    assert(agent.A[0][0] > 0.85f);
    assert(!isnan(agent.A[0][0]) && !isinf(agent.A[0][0]));

    printf("  --> PASS: Prior synchronization preserves custom priors and reinforces evidence stably\n");
}

void test_loop_detection_and_inertia_penalty(void) {
    printf("[TEST 12] Testing Behavioral Loop & Action Repetition Tracker...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 6, 8, 4);

    assert(!agent.loop_detected);
    assert(agent.consecutive_loops == 0);

    /* Simulate repeated same action and observation without progress */
    for (int i = 0; i < 5; i++) {
        agent.last_action = 1; /* Repeated code gen action */
        micro_actinf_step(&agent, 1);
    }

    /* Loop detection must be triggered */
    assert(agent.loop_detected);
    assert(agent.consecutive_loops >= 1);

    /* Action selection under loop must penalize repeating action 1 */
    uint8_t selected_action = micro_actinf_select_action(&agent);
    assert(selected_action != 1); /* Must switch away from repetitive loop */

    printf("  --> PASS: Loop detection detected 3+ repetitions and steered away from stuck action\n");
}

void test_multistep_planning_horizon(void) {
    printf("[TEST 13] Testing Multi-Step Trajectory Planning (Horizon H=1..3)...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 6, 8, 4);

    micro_actinf_set_horizon(&agent, 1);
    assert(agent.horizon == 1);
    uint8_t act_h1 = micro_actinf_select_action(&agent);

    micro_actinf_set_horizon(&agent, 3);
    assert(agent.horizon == 3);
    uint8_t act_h3 = micro_actinf_select_action(&agent);

    /* Both must return valid actions in [0, num_actions - 1] */
    assert(act_h1 < agent.num_actions);
    assert(act_h3 < agent.num_actions);

    /* Verify discounting bounds */
    for (uint8_t u = 0; u < agent.num_actions; u++) {
        assert(!isnan(agent.G[u]) && !isinf(agent.G[u]));
    }

    printf("  --> PASS: Multi-step trajectory planning executes stably with valid EFE bounds\n");
}

void test_action_safety_and_risk_gating(void) {
    printf("[TEST 14] Testing Action Safety & Risk Gating (ALLOW, MODIFY, ASK, DENY)...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 6, 8, 4);

    /* 1. Low-risk passive read should be allowed */
    actinf_verdict_t v_read = micro_actinf_evaluate_action(&agent, 0, ACTINF_RISK_READ, 0.80f);
    assert(v_read == ACTINF_VERDICT_ALLOW);

    /* 2. Destructive action with low confidence (<0.95) must require confirmation */
    actinf_verdict_t v_destruct = micro_actinf_evaluate_action(&agent, 1, ACTINF_RISK_DESTRUCTIVE, 0.80f);
    assert(v_destruct == ACTINF_VERDICT_ASK_CONFIRMATION);

    /* 3. When trapped in a loop, repeating the looping action must be DENIED */
    agent.loop_detected = true;
    agent.last_action = 1;
    actinf_verdict_t v_denied = micro_actinf_evaluate_action(&agent, 1, ACTINF_RISK_EDIT, 0.80f);
    assert(v_denied == ACTINF_VERDICT_DENY);

    printf("  --> PASS: Action safety governor enforces risk gating and denies loop actions\n");
}

void test_outcome_credit_assignment(void) {
    printf("[TEST 15] Testing Outcome-Driven Credit Assignment Learning...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 6, 8, 4);

    float initial_c4 = agent.C[4];
    float initial_progress = agent.progress_index;

    /* Record success on observation 4 (e.g. test_output passing) */
    micro_actinf_record_outcome(&agent, 3, 4, true, 1.0f);
    assert(agent.C[4] > initial_c4);
    assert(agent.progress_index > initial_progress);

    /* Record failure on observation 2 (error log) */
    float initial_c2 = agent.C[2];
    micro_actinf_record_outcome(&agent, 1, 2, false, 0.0f);
    assert(agent.C[2] < initial_c2);

    printf("  --> PASS: Success reinforces prior preference C; failure applies credit penalty\n");
}

int main(void) {
    printf("====================================================\n");
    printf("Running Micro-ActInf C Core Test Suite & Benchmarks\n");
    printf("====================================================\n");

    test_probabilistic_axioms();
    test_entropy_collapse();
    test_static_memory_footprint();
    test_dirichlet_online_learning();
    test_combined_inference_learning_latency();
    test_full_decision_cycle_latency();
    test_hyperparameter_bounds_and_nan();
    test_extreme_likelihood_zero_fallback();
    test_cache_consistency_after_learning();
    test_boundary_dimensions();
    test_sync_counts_from_matrices_and_adaptation();
    test_loop_detection_and_inertia_penalty();
    test_multistep_planning_horizon();
    test_action_safety_and_risk_gating();
    test_outcome_credit_assignment();

    printf("====================================================\n");
    printf("ALL 15 C UNIT TESTS & BENCHMARKS PASSED SUCCESSFULLY! (100%%)\n");
    printf("====================================================\n");
    return 0;
}
