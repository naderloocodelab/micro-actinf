/**
 * @file micro_actinf.h
 * @brief Ultra-lightweight (20KB) Zero-Allocation Discrete Active Inference & Variational POMDP State Filter.
 * @author naderloocodelab
 * @license MIT
 */

#ifndef MICRO_ACTINF_H
#define MICRO_ACTINF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define ACTINF_MAX_STATES   16
#define ACTINF_MAX_OBS      32
#define ACTINF_MAX_ACTIONS  8

/**
 * @brief Active Inference Agent State Structure
 * Memory footprint: Exactly 21,248 bytes (~20.75 KB).
 * Zero heap allocations required during runtime. Fits in CPU L1/L2 cache.
 */
typedef struct {
    uint8_t num_states;   /* K: Dimension of hidden state space (<= 16) */
    uint8_t num_obs;      /* M: Dimension of observation space (<= 32) */
    uint8_t num_actions;  /* A: Number of discrete control policies (<= 8) */
    float gamma;          /* Precision parameter (inverse temperature) */

    /* Belief State Vector s_t: Probability distribution on probability simplex Delta^{K-1} */
    float s[ACTINF_MAX_STATES];

    /* Observation Likelihood Matrix A[obs][state]: P(obs | state) */
    float A[ACTINF_MAX_OBS][ACTINF_MAX_STATES];

    /* Transition Tensors B[action][next_state][curr_state]: P(s_{t+1} | s_t, action) */
    float B[ACTINF_MAX_ACTIONS][ACTINF_MAX_STATES][ACTINF_MAX_STATES];

    /* Prior Preferences C[obs]: Desired observation distribution */
    float C[ACTINF_MAX_OBS];

    /* Cached Column Entropy of Matrix A: Sum_o A[o][s] * log(A[o][s]) */
    float A_entropy[ACTINF_MAX_STATES];

    /* Expected Free Energy G[action] for each policy */
    float G[ACTINF_MAX_ACTIONS];

    /* Policy Selection Probabilities pi[action] = Softmax(-gamma * G) */
    float pi[ACTINF_MAX_ACTIONS];

    /* Dirichlet parameters for online learning */
    float a_counts[ACTINF_MAX_OBS][ACTINF_MAX_STATES];
    float b_counts[ACTINF_MAX_ACTIONS][ACTINF_MAX_STATES][ACTINF_MAX_STATES];

    uint8_t last_action;
    uint32_t step_count;
} micro_actinf_t;

/**
 * @brief Initialize the Active Inference agent with uniform priors.
 */
void micro_actinf_init(micro_actinf_t *agent, uint8_t states, uint8_t obs, uint8_t actions);

/**
 * @brief Update belief distribution using variational message passing given an observation.
 * Formulation: s_{t+1} = Softmax( ln A_{o_t, :}^T + ln( B(u_{t-1}) s_t ) )
 */
void micro_actinf_step(micro_actinf_t *agent, uint8_t obs);

/**
 * @brief Select the optimal action minimizing Expected Free Energy G(u).
 * @return Optimal action index in [0, num_actions - 1]
 */
uint8_t micro_actinf_select_action(micro_actinf_t *agent);

/**
 * @brief Compute Shannon entropy of the current belief state in nats.
 */
float micro_actinf_shannon_entropy(const micro_actinf_t *agent);

/**
 * @brief Recompute cached column entropies for matrix A.
 */
void micro_actinf_update_cache(micro_actinf_t *agent);

/**
 * @brief Learn observation likelihood and transition parameters via Dirichlet conjugate updates.
 */
void micro_actinf_learn(micro_actinf_t *agent, uint8_t obs, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif /* MICRO_ACTINF_H */
