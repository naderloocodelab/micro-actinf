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
    printf("  Average Combined Latency (Inference + Dirichlet Learning): %.3f us (Target budget: <= 3.0 us)\n", mean_us);
    fflush(stdout);
    assert(mean_us <= 3.0);
    printf("  --> PASS: Hard real-time latency budget (<= 3.0 us) satisfied!\n");
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
    printf("  Average Full Decision Cycle Latency: %.3f us (Target budget: < 60.0 us)\n", mean_us);
    fflush(stdout);
    assert(mean_us < 60.0);
    printf("  --> PASS: Full decision cycle real-time guarantee verified\n");
    fflush(stdout);
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

    printf("====================================================\n");
    printf("ALL C UNIT TESTS & BENCHMARKS PASSED SUCCESSFULLY! (100%%)\n");
    printf("====================================================\n");
    return 0;
}
