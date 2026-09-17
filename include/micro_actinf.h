/**
 * @file micro_actinf.h
 * @brief Ultra-lightweight Zero-Allocation Discrete Active Inference & Variational POMDP Cognitive Governor.
 * @details Core computational engine for autonomous AI agents & embedded real-time systems.
 * Features:
 *   - Bounded multi-step policy trajectory planning (Horizon H <= 4)
 *   - Cost-aware Expected Free Energy (Pragmatic, Epistemic, Token/Latency/Risk Costs)
 *   - Analytical Bayesian Belief Filtering with Adaptive Prior Decay (Inertia-Immune)
 *   - Behavioral Loop & Action Repetition Fingerprint Tracker
 *   - Action Safety & Gating Engine (ALLOW, MODIFY, ASK_CONFIRMATION, DENY)
 *   - O(1) Online Conjugate Dirichlet Learning with Outcome Credit Assignment
 *   - Zero heap allocation (100% static footprint <= 36 KB, fits in CPU L1/L2 cache)
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

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(MICRO_ACTINF_BUILD_DLL)
    #define MICRO_ACTINF_API __declspec(dllexport)
  #elif defined(MICRO_ACTINF_STATIC)
    #define MICRO_ACTINF_API
  #else
    #define MICRO_ACTINF_API
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define MICRO_ACTINF_API __attribute__((visibility("default")))
  #else
    #define MICRO_ACTINF_API
  #endif
#endif

#define ACTINF_MAX_STATES      16
#define ACTINF_MAX_OBS         32
#define ACTINF_MAX_ACTIONS     8
#define ACTINF_MAX_HORIZON     4
#define ACTINF_HISTORY_LEN     8

#define ACTINF_EPSILON           1e-12f
#define ACTINF_DIRICHLET_EPSILON 1e-6f

/**
 * @brief Action Safety Risk Classifications
 */
typedef enum {
    ACTINF_RISK_READ       = 0,  /* Passive inspection, query, search */
    ACTINF_RISK_ANALYZE    = 1,  /* Computation, profiling, planning */
    ACTINF_RISK_TEST       = 2,  /* Non-destructive test execution */
    ACTINF_RISK_EDIT       = 3,  /* Modifying existing file/state */
    ACTINF_RISK_EXECUTE    = 4,  /* Running shell commands, scripts */
    ACTINF_RISK_DESTRUCTIVE= 5   /* Deletion, force overwrite, production deploy */
} actinf_risk_level_t;

/**
 * @brief Action Governance Verdict
 */
typedef enum {
    ACTINF_VERDICT_ALLOW            = 0, /* Execute immediately with high confidence */
    ACTINF_VERDICT_MODIFY           = 1, /* Downgrade action parameters (e.g. read-only) */
    ACTINF_VERDICT_ASK_CONFIRMATION = 2, /* Require user confirmation before proceeding */
    ACTINF_VERDICT_DENY             = 3  /* Block action (severe risk, loop, or out-of-regime) */
} actinf_verdict_t;

/**
 * @brief Online Dirichlet Learning & Adaptation Configuration
 */
typedef struct {
    float learning_rate_a;   /* eta_a: observation learning rate (default: 0.05f) */
    float learning_rate_b;   /* eta_b: transition learning rate (default: 0.05f) */
    float decay_factor;      /* lambda: forgetting factor (default: 0.998f) */
    bool enable_learning;    /* runtime toggle for parameter adaptation */
} micro_actinf_learning_config_t;

/**
 * @brief Cost and Weight Configuration for Expected Free Energy (EFE)
 */
typedef struct {
    float beta_epistemic;    /* Epistemic exploration weight (default: 1.0f) */
    float weight_token_cost; /* Token consumption penalty weight (default: 0.2f) */
    float weight_latency;    /* Execution latency penalty weight (default: 0.1f) */
    float weight_risk;       /* Action risk penalty weight (default: 0.5f) */
    float loop_penalty;      /* Penalty incurred when repetitive loop is detected (default: 2.0f) */
    float gamma_discount;    /* Multi-step trajectory temporal discount (default: 0.90f) */
} micro_actinf_cost_config_t;

/**
 * @brief Single historical step record for loop & inertia fingerprinting
 */
typedef struct {
    uint8_t state_dom;
    uint8_t action;
    uint8_t obs;
    float entropy;
} actinf_step_record_t;

/**
 * @brief Active Inference Agent State Structure
 * Memory footprint: <= 36 KB (deterministic static storage). Zero heap allocation.
 */
typedef struct {
    uint8_t num_states;   /* K: Dimension of hidden state space (<= 16) */
    uint8_t num_obs;      /* M: Dimension of observation space (<= 32) */
    uint8_t num_actions;  /* A: Number of discrete control policies (<= 8) */
    uint8_t horizon;      /* H: Planning horizon (1 to ACTINF_MAX_HORIZON, default: 2) */
    bool a_entropy_dirty; /* Flag indicating A_entropy cache needs recomputation */
    float gamma;          /* Softmax precision parameter (inverse temperature, default: 2.0) */
    float alpha_prior;    /* Adaptive prior decay/mixing factor (prevents lock-in, default: 0.25) */

    /* Belief State Vector s_t: Probability distribution on probability simplex Delta^{K-1} */
    float s[ACTINF_MAX_STATES];

    /* Previous Belief Vector s_{t-1}: Tracked for O(1) transition Dirichlet learning */
    float s_prev[ACTINF_MAX_STATES];

    /* Observation Likelihood Matrix A[obs][state]: P(obs | state) */
    float A[ACTINF_MAX_OBS][ACTINF_MAX_STATES];

    /* Transition Tensors B[action][next_state][curr_state]: P(s_{t+1} | s_t, action) */
    float B[ACTINF_MAX_ACTIONS][ACTINF_MAX_STATES][ACTINF_MAX_STATES];

    /* Prior Preferences C[obs]: Desired observation distribution (Pragmatic Goals) */
    float C[ACTINF_MAX_OBS];

    /* Action Cost Profile: [action][0=token_cost, 1=latency_cost, 2=base_risk] */
    float action_costs[ACTINF_MAX_ACTIONS][3];

    /* Action Risk Levels */
    uint8_t action_risk[ACTINF_MAX_ACTIONS];

    /* Cached Column Entropy of Matrix A: Sum_o A[o][s] * log(A[o][s]) */
    float A_entropy[ACTINF_MAX_STATES];

    /* Expected Free Energy G[action] for each policy */
    float G[ACTINF_MAX_ACTIONS];

    /* Policy Selection Probabilities pi[action] = Softmax(-gamma * G) */
    float pi[ACTINF_MAX_ACTIONS];

    /* Dirichlet accumulators for online learning */
    float a_counts[ACTINF_MAX_OBS][ACTINF_MAX_STATES];
    float b_counts[ACTINF_MAX_STATES][ACTINF_MAX_STATES][ACTINF_MAX_ACTIONS];

    /* Hyperparameter configurations */
    micro_actinf_learning_config_t learning_cfg;
    micro_actinf_cost_config_t cost_cfg;

    /* Behavioral Loop Detection & Fingerprint Ring Buffer */
    actinf_step_record_t history[ACTINF_HISTORY_LEN];
    uint8_t history_idx;
    uint8_t history_count;
    uint8_t consecutive_loops;
    bool loop_detected;

    /* Runtime Progress & Diagnostics */
    float progress_index;       /* Cumulative goal alignment index [0.0, 1.0] */
    float last_entropy;         /* Previous Shannon entropy for velocity tracking */
    float entropy_velocity;     /* Rate of entropy collapse (negative is converging) */

    uint8_t last_action;
    uint32_t step_count;
} micro_actinf_t;

/**
 * @brief Initialize Active Inference agent with uniform/calibrated priors.
 */
MICRO_ACTINF_API void micro_actinf_init(micro_actinf_t *agent, uint8_t states, uint8_t obs, uint8_t actions);

/**
 * @brief Configure online Dirichlet learning hyperparameters.
 */
MICRO_ACTINF_API void micro_actinf_set_learning_config(micro_actinf_t *agent, const micro_actinf_learning_config_t *cfg);

/**
 * @brief Configure cost and EFE evaluation parameters.
 */
MICRO_ACTINF_API void micro_actinf_set_cost_config(micro_actinf_t *agent, const micro_actinf_cost_config_t *cfg);

/**
 * @brief Set planning trajectory horizon H (bounded in [1, ACTINF_MAX_HORIZON]).
 */
MICRO_ACTINF_API void micro_actinf_set_horizon(micro_actinf_t *agent, uint8_t horizon);

/**
 * @brief Update belief distribution using variational message passing given an observation.
 * Formulation: s_{prior} = (1 - alpha)*B(u)*s_{prev} + alpha*(1/K)
 *              s_{t+1}   = (A_{o_t, :} \odot s_{prior}) / sum(...)
 */
MICRO_ACTINF_API void micro_actinf_step(micro_actinf_t *agent, uint8_t obs);

/**
 * @brief Select optimal control action minimizing multi-step Cost-Aware Expected Free Energy.
 * Evaluates candidates, balances pragmatic goal value with epistemic information gain,
 * subtracts token/latency/risk costs, and applies loop avoidance penalties.
 */
MICRO_ACTINF_API uint8_t micro_actinf_select_action(micro_actinf_t *agent);

/**
 * @brief Evaluate an action proposed by an LLM against the Active Inference cognitive regime and risk model.
 * @param agent Pointer to active inference agent.
 * @param proposed_action Action index proposed by agent/LLM.
 * @param risk Explicit risk level of the operation.
 * @param confidence_threshold Minimum confidence required to approve high-risk action.
 * @return Verdict (ALLOW, MODIFY, ASK_CONFIRMATION, or DENY).
 */
MICRO_ACTINF_API actinf_verdict_t micro_actinf_evaluate_action(micro_actinf_t *agent,
                                                               uint8_t proposed_action,
                                                               actinf_risk_level_t risk,
                                                               float confidence_threshold);

/**
 * @brief Record execution outcome and perform credit assignment learning.
 * Updates Dirichlet pseudo-counts, aligns prior preferences C(o), and updates progress index.
 * @param agent Pointer to agent.
 * @param action Executed action.
 * @param outcome_obs Resulting observation.
 * @param success True if action produced expected productive progress; false on failure/error.
 * @param progress_delta Quantitative progress metric delta.
 */
MICRO_ACTINF_API void micro_actinf_record_outcome(micro_actinf_t *agent,
                                                  uint8_t action,
                                                  uint8_t outcome_obs,
                                                  bool success,
                                                  float progress_delta);

/**
 * @brief Execute O(1) conjugate Dirichlet parameter learning update.
 */
MICRO_ACTINF_API void micro_actinf_learn_step(micro_actinf_t *agent, uint32_t observation, uint32_t prev_action);

/**
 * @brief Compute Shannon entropy of current belief state in nats.
 */
MICRO_ACTINF_API float micro_actinf_shannon_entropy(const micro_actinf_t *agent);

/**
 * @brief Get dominant state index and confidence percentage.
 */
MICRO_ACTINF_API uint8_t micro_actinf_get_dominant_state(const micro_actinf_t *agent, float *out_confidence);

/**
 * @brief Synchronize Dirichlet pseudo-counts from likelihood matrix A and transition tensor B.
 */
MICRO_ACTINF_API void micro_actinf_sync_counts_from_matrices(micro_actinf_t *agent);

/**
 * @brief Reset agent belief state to uniform prior while preserving learned matrices.
 */
MICRO_ACTINF_API void micro_actinf_reset_state(micro_actinf_t *agent);

/**
 * @brief Recompute cached column entropies for matrix A.
 */
MICRO_ACTINF_API void micro_actinf_update_cache(micro_actinf_t *agent);

#ifdef __cplusplus
}
#endif

#endif /* MICRO_ACTINF_H */
