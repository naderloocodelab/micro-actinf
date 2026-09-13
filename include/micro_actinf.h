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

#define MICRO_ACTINF_K      ACTINF_MAX_STATES
#define MICRO_ACTINF_M      ACTINF_MAX_OBS
#define MICRO_ACTINF_A      ACTINF_MAX_ACTIONS

#define ACTINF_EPSILON           1e-12f
#define ACTINF_DIRICHLET_EPSILON 1e-6f

/**
 * @brief Online Dirichlet Learning Configuration
 */
typedef struct {
    float learning_rate_a;   /* eta_a: observation learning rate (default: 0.05f) */
    float learning_rate_b;   /* eta_b: transition learning rate (default: 0.05f) */
    float decay_factor;      /* lambda: forgetting factor (default: 0.998f) */
    bool enable_learning;    /* runtime toggle for parameter adaptation */
} micro_actinf_learning_config_t;

/**
 * @brief Active Inference Agent State Structure
 * Memory footprint: <= 36 KB (Exactly 20,896 bytes).
 * Zero heap allocations required during runtime. Fits in CPU L1/L2 cache.
 */
typedef struct {
    uint8_t num_states;   /* K: Dimension of hidden state space (<= 16) */
    uint8_t num_obs;      /* M: Dimension of observation space (<= 32) */
    uint8_t num_actions;  /* A: Number of discrete control policies (<= 8) */
    float gamma;          /* Precision parameter (inverse temperature) */

    /* Belief State Vector s_t: Probability distribution on probability simplex Delta^{K-1} */
    float s[ACTINF_MAX_STATES];

    /* Previous Belief Vector s_{t-1}: Tracked for O(1) transition Dirichlet learning */
    float s_prev[ACTINF_MAX_STATES];

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

    /* Dirichlet accumulators for online learning */
    float a_counts[MICRO_ACTINF_M][MICRO_ACTINF_K];
    float b_counts[MICRO_ACTINF_K][MICRO_ACTINF_K][MICRO_ACTINF_A];
    micro_actinf_learning_config_t learning_cfg;

    uint8_t last_action;
    uint32_t step_count;
} micro_actinf_t;

/**
 * @brief Initialize the Active Inference agent with uniform priors.
 */
void micro_actinf_init(micro_actinf_t *agent, uint8_t states, uint8_t obs, uint8_t actions);

/**
 * @brief Configure online Dirichlet learning hyperparameters.
 */
void micro_actinf_set_learning_config(micro_actinf_t *agent, const micro_actinf_learning_config_t *cfg);

/**
 * @brief Update belief distribution using variational message passing given an observation.
 * Formulation: s_{t+1} = Softmax( ln A_{o_t, :}^T + ln( B(u_{t-1}) s_t ) )
 */
void micro_actinf_step(micro_actinf_t *agent, uint8_t obs);

/**
 * @brief Execute O(1) conjugate Dirichlet parameter learning update.
 * Recursive pseudo-count updates for matrices A and B with exponential decay.
 * @param agent Pointer to active inference agent.
 * @param observation Observed observation index o_t in [0, num_obs - 1].
 * @param prev_action Previous control action u_{t-1} in [0, num_actions - 1].
 */
void micro_actinf_learn_step(micro_actinf_t *agent, uint32_t observation, uint32_t prev_action);

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
