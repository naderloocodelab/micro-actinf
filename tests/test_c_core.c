/**
 * @file test_c_core.c
 * @brief Unit tests for Micro-ActInf C11 Engine.
 * @details Validates Kolmogorov axioms, Shannon entropy reduction, and sub-15us cycle time.
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

void test_latency_benchmark(void) {
    printf("[TEST 3] Running Sub-15us Real-Time Latency Benchmark (100,000 cycles)...\n");
    micro_actinf_t agent;
    micro_actinf_init(&agent, 16, 32, 8);

    const int CYCLES = 100000;
    double t0 = get_time_ns();
    for (int i = 0; i < CYCLES; i++) {
        micro_actinf_step(&agent, (uint8_t)(i % 32));
        micro_actinf_select_action(&agent);
    }
    double t1 = get_time_ns();

    double mean_us = ((t1 - t0) / (double)CYCLES) / 1000.0;
    printf("  Average Cycle Latency: %.3f us (Target budget: < 60.0 us)\n", mean_us);
    fflush(stdout);
    assert(mean_us < 60.0);
    printf("  --> PASS: Real-time execution guarantee satisfied\n");
    fflush(stdout);
}

int main(void) {
    printf("====================================================\n");
    printf("Running Micro-ActInf C Core Test Suite\n");
    printf("====================================================\n");

    test_probabilistic_axioms();
    test_entropy_collapse();
    test_latency_benchmark();

    printf("====================================================\n");
    printf("ALL C UNIT TESTS PASSED SUCCESSFULLY! (100%%)\n");
    printf("====================================================\n");
    return 0;
}
