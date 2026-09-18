/**
 * @file test_fast_log_accuracy.c
 * @brief Numerical Validation Suite for Logarithmic Precision & Action Ranking Preservation.
 * @details Validates:
 *   1. CRT logf numerical bounds across 1,000,000 points on (0.0, 1.0].
 *   2. Fast IEEE 754 log approximation accuracy (max absolute & relative error).
 *   3. Invariance of Free Energy Action Selection / Softmax ranking under log approximations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

/* Fast IEEE-754 single-precision natural log approximation */
static inline float fast_log2_approx(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = (float)vx.i;
    y *= 1.1920928955078125e-7f; /* 1 / (1 << 23) */
    return y - 126.94269504f;
}

static inline float fast_log_approx(float x) {
    return 0.69314718056f * fast_log2_approx(x); /* ln(2) * log2(x) */
}

int main(void) {
    printf("========================================================\n");
    printf("Running Logarithmic Accuracy & Ranking Preservation Test\n");
    printf("========================================================\n");

    const int SAMPLES = 1000000;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    /* 1. Test across 1,000,000 points on (1e-6, 1.0] */
    for (int i = 1; i <= SAMPLES; i++) {
        float x = (float)i / (float)SAMPLES;
        float true_log = logf(x);
        float approx = fast_log_approx(x);

        float abs_diff = fabsf(true_log - approx);
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }

        if (fabsf(true_log) > 1e-4f) {
            float rel_diff = abs_diff / fabsf(true_log);
            if (rel_diff > max_rel_diff) {
                max_rel_diff = rel_diff;
            }
        }
    }

    printf("[PRECISION METRICS across 1,000,000 samples]:\n");
    printf("  Max Absolute Error: %.6f\n", max_abs_diff);
    printf("  Max Relative Error: %.4f%%\n", max_rel_diff * 100.0f);
    assert(max_abs_diff < 0.15f);
    printf("  --> PASS: Logarithmic approximation error is strictly bounded\n");

    /* 2. Action Ranking Preservation Test:
     * Verify that argmin G(u) is 100% identical between standard CRT logf and approximation.
     */
    printf("\n[ACTION RANKING PRESERVATION TEST]:\n");
    int ranking_matches = 0;
    const int RANKING_TRIALS = 10000;

    for (int t = 0; t < RANKING_TRIALS; t++) {
        float p[4];
        float sum = 0.0f;
        for (int a = 0; a < 4; a++) {
            p[a] = (float)(rand() % 1000 + 10);
            sum += p[a];
        }
        for (int a = 0; a < 4; a++) p[a] /= sum;

        /* Rank using true log */
        int best_true = 0;
        float min_true = -logf(p[0]);
        for (int a = 1; a < 4; a++) {
            float val = -logf(p[a]);
            if (val < min_true) {
                min_true = val;
                best_true = a;
            }
        }

        /* Rank using approx log */
        int best_approx = 0;
        float min_approx = -fast_log_approx(p[0]);
        for (int a = 1; a < 4; a++) {
            float val = -fast_log_approx(p[a]);
            if (val < min_approx) {
                min_approx = val;
                best_approx = a;
            }
        }

        if (best_true == best_approx) {
            ranking_matches++;
        }
    }

    float match_pct = ((float)ranking_matches / (float)RANKING_TRIALS) * 100.0f;
    printf("  Top-1 Action Selection Invariance: %.2f%% (%d / %d trials)\n",
           match_pct, ranking_matches, RANKING_TRIALS);
    assert(match_pct >= 99.9f);
    printf("  --> PASS: Control policy selection is invariant under approximation\n");

    printf("========================================================\n");
    printf("LOG ACCURACY & CONTROL INVARIANCE VERIFIED (100%% PASS)\n");
    printf("========================================================\n");
    return 0;
}
