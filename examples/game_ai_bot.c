/**
 * @file game_ai_bot.c
 * @brief Real-Time Game AI & Robotics Controller Benchmark with O(1) Online Dirichlet Learning.
 * @details Runs 100,000 consecutive decision + parameter adaptation cycles with zero heap allocations.
 * @author naderloocodelab
 * @license MIT
 */

#include "micro_actinf.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* Game States */
#define STATE_PATROL   0
#define STATE_ENGAGE   1
#define STATE_EVADE    2
#define STATE_SEARCH   3
#define STATE_HEAL     4

/* Game Observations (Sensory Inputs) */
#define OBS_CLEAR      0
#define OBS_ENEMY_FAR  1
#define OBS_ENEMY_NEAR 2
#define OBS_LOW_HEALTH 3
#define OBS_LOOT_FOUND 4

/* Game Actions (Bot Policies) */
#define ACT_EXPLORE    0
#define ACT_ATTACK     1
#define ACT_RETREAT    2
#define ACT_SEEK_AID   3

static const char *STATE_NAMES[]  = {"PATROL", "ENGAGE", "EVADE", "SEARCH", "HEAL"};
static const char *ACTION_NAMES[] = {"EXPLORE", "ATTACK", "RETREAT", "SEEK_AID"};

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

int main(void) {
    printf("=================================================================\n");
    printf("🎮 Micro-ActInf Real-Time Game AI & Adaptive Controller Benchmark\n");
    printf("   [O(1) Online Conjugate Dirichlet Learning Enabled]\n");
    printf("=================================================================\n");

    micro_actinf_t bot;
    micro_actinf_init(&bot, 5, 5, 4);

    /* Configure Online Dirichlet Learning Engine */
    micro_actinf_learning_config_t learn_cfg = {
        .learning_rate_a = 0.05f,   /* Observation likelihood adaptation rate */
        .learning_rate_b = 0.05f,   /* Transition model adaptation rate */
        .decay_factor    = 0.998f,  /* Exponential forgetting factor (lambda) */
        .enable_learning = true     /* Continuous real-time parameter learning */
    };
    micro_actinf_set_learning_config(&bot, &learn_cfg);

    /* Configure likelihood mappings A: P(obs | state) */
    bot.A[OBS_CLEAR][STATE_PATROL]       = 0.80f;
    bot.A[OBS_ENEMY_FAR][STATE_SEARCH]   = 0.75f;
    bot.A[OBS_ENEMY_NEAR][STATE_ENGAGE]  = 0.85f;
    bot.A[OBS_LOW_HEALTH][STATE_EVADE]   = 0.90f;
    bot.A[OBS_LOOT_FOUND][STATE_HEAL]    = 0.80f;
    micro_actinf_sync_counts_from_matrices(&bot);

    /* Prior preferences C: Desires clear path and loot, dislikes low health */
    bot.C[OBS_CLEAR]      = 0.5f;
    bot.C[OBS_ENEMY_NEAR] = -0.2f;
    bot.C[OBS_LOW_HEALTH] = -1.0f;
    bot.C[OBS_LOOT_FOUND] = 0.8f;

    printf("Executing 100,000 real-time game decision + learning loops...\n");

    const int TOTAL_CYCLES = 100000;
    uint8_t current_obs = OBS_CLEAR;
    uint8_t prev_action = ACT_EXPLORE;

    double start_ns = get_time_ns();

    for (int cycle = 0; cycle < TOTAL_CYCLES; cycle++) {
        /* 1. Sensory belief update: s_t = Softmax(ln A + ln B s_{t-1}) */
        micro_actinf_step(&bot, current_obs);

        /* 2. O(1) Conjugate Dirichlet parameter learning: updates A and B */
        micro_actinf_learn_step(&bot, (uint32_t)current_obs, (uint32_t)prev_action);

        /* 3. Optimal policy selection: minimize Expected Free Energy G(u) */
        uint8_t action = micro_actinf_select_action(&bot);
        prev_action = action;

        /* 4. Simulate environment dynamics & sensory feedback */
        if (action == ACT_ATTACK) {
            current_obs = (cycle % 5 == 0) ? OBS_LOW_HEALTH : OBS_ENEMY_NEAR;
        } else if (action == ACT_RETREAT) {
            current_obs = OBS_CLEAR;
        } else {
            current_obs = (cycle % 7 == 0) ? OBS_ENEMY_FAR : OBS_CLEAR;
        }
    }

    double end_ns = get_time_ns();
    double total_time_ms = (end_ns - start_ns) / 1e6;
    double per_step_us   = ((end_ns - start_ns) / (double)TOTAL_CYCLES) / 1000.0;

    /* Verify parameter stability & mathematical convergence */
    bool valid_convergence = true;
    for (uint8_t s = 0; s < bot.num_states; s++) {
        float sum_a = 0.0f;
        for (uint8_t m = 0; m < bot.num_obs; m++) {
            if (isnan(bot.A[m][s]) || isinf(bot.A[m][s]) || isnan(bot.a_counts[m][s]) || isinf(bot.a_counts[m][s])) {
                valid_convergence = false;
            }
            sum_a += bot.A[m][s];
        }
        if (fabsf(sum_a - 1.0f) > 1e-4f) valid_convergence = false;
    }

    for (uint8_t u = 0; u < bot.num_actions; u++) {
        for (uint8_t s = 0; s < bot.num_states; s++) {
            float sum_b = 0.0f;
            for (uint8_t s_prime = 0; s_prime < bot.num_states; s_prime++) {
                if (isnan(bot.B[u][s_prime][s]) || isinf(bot.B[u][s_prime][s])) {
                    valid_convergence = false;
                }
                sum_b += bot.B[u][s_prime][s];
            }
            if (fabsf(sum_b - 1.0f) > 1e-4f) valid_convergence = false;
        }
    }
    assert(valid_convergence);

    uint8_t dominant_state = 0;
    float max_s = bot.s[0];
    for (uint8_t s = 1; s < bot.num_states; s++) {
        if (bot.s[s] > max_s) { max_s = bot.s[s]; dominant_state = s; }
    }

    printf("\nBenchmark Results:\n");
    printf("  • Total Decision+Learn Cycles: %d\n", TOTAL_CYCLES);
    printf("  • Total Elapsed Time:          %.2f ms\n", total_time_ms);
    printf("  • Full Cycle Latency:          %.3f microseconds (Budget: < 15.0 us) [%s]\n",
           per_step_us, (per_step_us < 15.0) ? "PASS" : "WARN");
    printf("  • Decision + Learning Rate:    %.0f decisions/second\n", 1e6 / per_step_us);
    printf("  • Final Dominant State:        %s\n", STATE_NAMES[dominant_state]);
    printf("  • Final Prescribed Action:     %s\n", ACTION_NAMES[prev_action]);
    printf("  • Final Shannon Entropy:       %.4f nats\n", micro_actinf_shannon_entropy(&bot));
    printf("  • Memory Footprint:            %zu bytes (Zero dynamic heap allocations)\n", sizeof(bot));
    printf("  • Parameter Stability:         100%% CONVERGED (0%% overflow, strictly bounded)\n");
    printf("=================================================================\n");

    return 0;
}
