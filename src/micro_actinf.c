/**
 * @file micro_actinf.c
 * @brief High-performance C11 Implementation of Micro-ActInf.
 * @author naderloocodelab
 * @license MIT
 */

#include "micro_actinf.h"
#include <math.h>
#include <string.h>

#define ACTINF_EPSILON 1e-12f

static void softmax_inplace(float *vec, uint8_t n) {
    float max_val = vec[0];
    for (uint8_t i = 1; i < n; i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }

    float sum = 0.0f;
    for (uint8_t i = 0; i < n; i++) {
        vec[i] = expf(vec[i] - max_val);
        sum += vec[i];
    }

    if (sum > ACTINF_EPSILON) {
        float inv_sum = 1.0f / sum;
        for (uint8_t i = 0; i < n; i++) {
            vec[i] *= inv_sum;
        }
    } else {
        float inv_n = 1.0f / (float)n;
        for (uint8_t i = 0; i < n; i++) {
            vec[i] = inv_n;
        }
    }
}

void micro_actinf_update_cache(micro_actinf_t *agent) {
    if (!agent) return;
    for (uint8_t s = 0; s < agent->num_states; s++) {
        float h_col = 0.0f;
        for (uint8_t o = 0; o < agent->num_obs; o++) {
            if (agent->A[o][s] > ACTINF_EPSILON) {
                h_col += agent->A[o][s] * logf(agent->A[o][s]);
            }
        }
        agent->A_entropy[s] = h_col;
    }
}

void micro_actinf_set_learning_config(micro_actinf_t *agent, const micro_actinf_learning_config_t *cfg) {
    if (!agent || !cfg) return;
    agent->learning_cfg = *cfg;
}

void micro_actinf_init(micro_actinf_t *agent, uint8_t states, uint8_t obs, uint8_t actions) {
    if (!agent) return;

    agent->num_states = (states > 0 && states <= ACTINF_MAX_STATES) ? states : ACTINF_MAX_STATES;
    agent->num_obs = (obs > 0 && obs <= ACTINF_MAX_OBS) ? obs : ACTINF_MAX_OBS;
    agent->num_actions = (actions > 0 && actions <= ACTINF_MAX_ACTIONS) ? actions : ACTINF_MAX_ACTIONS;
    agent->gamma = 2.0f;
    agent->last_action = 0;
    agent->step_count = 0;

    /* Initialize default online Dirichlet learning configuration */
    agent->learning_cfg.learning_rate_a = 0.05f;
    agent->learning_cfg.learning_rate_b = 0.05f;
    agent->learning_cfg.decay_factor = 0.998f;
    agent->learning_cfg.enable_learning = true;

    /* Initialize uniform prior belief & previous belief */
    float p_s = 1.0f / (float)agent->num_states;
    for (uint8_t s = 0; s < agent->num_states; s++) {
        agent->s[s] = p_s;
        agent->s_prev[s] = p_s;
    }

    /* Initialize observation likelihood matrix A with pseudo-counts */
    float p_o = 1.0f / (float)agent->num_obs;
    for (uint8_t o = 0; o < agent->num_obs; o++) {
        for (uint8_t s = 0; s < agent->num_states; s++) {
            agent->A[o][s] = p_o;
            agent->a_counts[o][s] = 1.0f;
        }
    }

    micro_actinf_update_cache(agent);

    /* Initialize transition matrices B with weak self-persistence & b_counts */
    for (uint8_t u = 0; u < agent->num_actions; u++) {
        for (uint8_t j = 0; j < agent->num_states; j++) {
            for (uint8_t i = 0; i < agent->num_states; i++) {
                if (i == j) {
                    agent->B[u][i][j] = 0.50f;
                } else {
                    agent->B[u][i][j] = 0.50f / (float)(agent->num_states - 1);
                }
                agent->b_counts[i][j][u] = 1.0f;
            }
        }
    }

    /* Initialize preferences C (neutral) */
    for (uint8_t o = 0; o < agent->num_obs; o++) {
        agent->C[o] = 0.0f;
    }

    for (uint8_t u = 0; u < agent->num_actions; u++) {
        agent->G[u] = 0.0f;
        agent->pi[u] = 1.0f / (float)agent->num_actions;
    }
}

void micro_actinf_step(micro_actinf_t *agent, uint8_t obs) {
    if (!agent || obs >= agent->num_obs) return;

    const uint8_t n_states = agent->num_states;
    const uint8_t last_act = agent->last_action;
    float * restrict s = agent->s;
    float * restrict s_prev = agent->s_prev;

    /* Save previous belief vector s_{t-1} for O(1) online learning */
    for (uint8_t i = 0; i < n_states; i++) {
        s_prev[i] = s[i];
    }

    /* 1. Prior state prediction via previous action: s_prior = B(u_{t-1}) * s_{t-1} */
    /* 2. Analytical Variational Bayes belief update:
     *    s_{t+1} = Softmax( ln A_{o_t, :}^T + ln( B(u_{t-1}) s_t ) )
     *            = (A_{o_t, :} \odot s_prior) / \sum (A_{o_t, :} \odot s_prior)
     */
    float sum = 0.0f;
    float unnorm[ACTINF_MAX_STATES];
    const float * restrict a_row = agent->A[obs];

    for (uint8_t i = 0; i < n_states; i++) {
        float s_prior = 0.0f;
        const float * restrict b_row = agent->B[last_act][i];
        for (uint8_t j = 0; j < n_states; j++) {
            s_prior += b_row[j] * s_prev[j];
        }
        float p = a_row[i] * s_prior;
        unnorm[i] = p;
        sum += p;
    }

    if (sum > ACTINF_EPSILON) {
        float inv_sum = 1.0f / sum;
        for (uint8_t i = 0; i < n_states; i++) {
            s[i] = unnorm[i] * inv_sum;
        }
    } else {
        /* Fallback: Log-space Softmax for extreme numerical boundary conditions */
        float log_posterior[ACTINF_MAX_STATES];
        for (uint8_t i = 0; i < n_states; i++) {
            float likelihood = (a_row[i] > ACTINF_EPSILON) ? a_row[i] : ACTINF_EPSILON;
            float prior = (unnorm[i] > ACTINF_EPSILON) ? unnorm[i] : ACTINF_EPSILON;
            log_posterior[i] = logf(likelihood) + logf(prior);
        }
        softmax_inplace(log_posterior, n_states);
        for (uint8_t i = 0; i < n_states; i++) {
            s[i] = log_posterior[i];
        }
    }

    agent->step_count++;
}

uint8_t micro_actinf_select_action(micro_actinf_t *agent) {
    if (!agent) return 0;

    /* Compute Expected Free Energy G(u) for each policy u */
    for (uint8_t u = 0; u < agent->num_actions; u++) {
        /* Predicted state under action u */
        float s_pred[ACTINF_MAX_STATES] = {0.0f};
        for (uint8_t i = 0; i < agent->num_states; i++) {
            for (uint8_t j = 0; j < agent->num_states; j++) {
                s_pred[i] += agent->B[u][i][j] * agent->s[j];
            }
        }

        /* Predicted observation under action u */
        float o_pred[ACTINF_MAX_OBS] = {0.0f};
        for (uint8_t o = 0; o < agent->num_obs; o++) {
            for (uint8_t i = 0; i < agent->num_states; i++) {
                o_pred[o] += agent->A[o][i] * s_pred[i];
            }
        }

        /* Pragmatic Value: sum_o o_pred[o] * C[o] */
        float pragmatic = 0.0f;
        for (uint8_t o = 0; o < agent->num_obs; o++) {
            pragmatic += o_pred[o] * agent->C[o];
        }

        /* Epistemic Value (Information Gain / Mutual Information):
         * Sum_{o, i} s_pred[i] * A[o][i] * (log A[o][i] - log o_pred[o])
         * = Sum_i s_pred[i] * A_entropy[i] - Sum_o o_pred[o] * log o_pred[o]
         */
        float epistemic = 0.0f;
        for (uint8_t o = 0; o < agent->num_obs; o++) {
            if (o_pred[o] > ACTINF_EPSILON) {
                epistemic -= o_pred[o] * logf(o_pred[o]);
            }
        }
        for (uint8_t i = 0; i < agent->num_states; i++) {
            epistemic += s_pred[i] * agent->A_entropy[i];
        }

        /* G(u) = - (Pragmatic + Epistemic) */
        agent->G[u] = -(pragmatic + 0.5f * epistemic);
    }

    /* Calculate policy probabilities: pi(u) = Softmax(-gamma * G) */
    float logits[ACTINF_MAX_ACTIONS];
    for (uint8_t u = 0; u < agent->num_actions; u++) {
        logits[u] = -agent->gamma * agent->G[u];
    }
    softmax_inplace(logits, agent->num_actions);
    for (uint8_t u = 0; u < agent->num_actions; u++) {
        agent->pi[u] = logits[u];
    }

    /* Greedy selection of minimum Free Energy action */
    uint8_t best_u = 0;
    float max_prob = agent->pi[0];
    for (uint8_t u = 1; u < agent->num_actions; u++) {
        if (agent->pi[u] > max_prob) {
            max_prob = agent->pi[u];
            best_u = u;
        }
    }

    agent->last_action = best_u;
    return best_u;
}

float micro_actinf_shannon_entropy(const micro_actinf_t *agent) {
    if (!agent) return 0.0f;
    float h = 0.0f;
    for (uint8_t s = 0; s < agent->num_states; s++) {
        if (agent->s[s] > ACTINF_EPSILON) {
            h -= agent->s[s] * logf(agent->s[s]);
        }
    }
    return h;
}

void micro_actinf_learn_step(micro_actinf_t *agent, uint32_t observation, uint32_t prev_action) {
    if (!agent || observation >= (uint32_t)agent->num_obs || prev_action >= (uint32_t)agent->num_actions) return;
    if (!agent->learning_cfg.enable_learning) return;

    const float lambda_a = agent->learning_cfg.decay_factor;
    const float lambda_b = agent->learning_cfg.decay_factor;
    const float eta_a = agent->learning_cfg.learning_rate_a;
    const float eta_b = agent->learning_cfg.learning_rate_b;
    const uint8_t n_states = agent->num_states;
    const uint8_t n_obs = agent->num_obs;

    const float * restrict s = agent->s;
    const float * restrict s_prev = agent->s_prev;

    /* 1. Observation Pseudo-Count Update:
     *    a_{o_t, s} <- lambda_a * a_{o_t, s} + eta_a * s_t(s)  \forall s in {0, ..., K-1}
     */
    float * restrict a_obs = agent->a_counts[observation];
    for (uint8_t k = 0; k < n_states; k++) {
        a_obs[k] = lambda_a * a_obs[k] + eta_a * s[k];
    }

    /* 2. Transition Pseudo-Count Update:
     *    b_{s', s, u_{t-1}} <- lambda_b * b_{s', s, u_{t-1}} + eta_b * s_t(s') * s_{t-1}(s)  \forall s, s' in {0, ..., K-1}
     */
    for (uint8_t s_prime = 0; s_prime < n_states; s_prime++) {
        const float eta_s_prime = eta_b * s[s_prime];
        for (uint8_t k = 0; k < n_states; k++) {
            agent->b_counts[s_prime][k][prev_action] = 
                lambda_b * agent->b_counts[s_prime][k][prev_action] + eta_s_prime * s_prev[k];
        }
    }

    /* 3. Expectation Mapping (Normalized Categorical):
     *    A_{o, s} = (a_{o, s} + eps) / sum_{m=0}^{M-1} (a_{m, s} + eps)
     *    B_{s', s, u} = (b_{s', s, u} + eps) / sum_{k=0}^{K-1} (b_{k, s, u} + eps)
     *    (Pre-calculate inverse column sums to avoid divisions; enforce eps = 10^-6)
     */
    /* Matrix A expectation mapping: contiguous row traversal */
    float col_sums_a[ACTINF_MAX_STATES] = {0.0f};
    for (uint8_t m = 0; m < n_obs; m++) {
        const float * restrict a_row = agent->a_counts[m];
        for (uint8_t k = 0; k < n_states; k++) {
            col_sums_a[k] += a_row[k];
        }
    }
    const float eps_obs = (float)n_obs * ACTINF_DIRICHLET_EPSILON;
    float inv_col_a[ACTINF_MAX_STATES];
    for (uint8_t k = 0; k < n_states; k++) {
        inv_col_a[k] = 1.0f / (col_sums_a[k] + eps_obs);
    }
    for (uint8_t m = 0; m < n_obs; m++) {
        const float * restrict a_row = agent->a_counts[m];
        float * restrict A_row = agent->A[m];
        for (uint8_t k = 0; k < n_states; k++) {
            A_row[k] = (a_row[k] + ACTINF_DIRICHLET_EPSILON) * inv_col_a[k];
        }
    }

    /* Matrix B expectation mapping for prev_action */
    float col_sums_b[ACTINF_MAX_STATES] = {0.0f};
    for (uint8_t s_prime = 0; s_prime < n_states; s_prime++) {
        for (uint8_t k = 0; k < n_states; k++) {
            col_sums_b[k] += agent->b_counts[s_prime][k][prev_action];
        }
    }
    const float eps_states = (float)n_states * ACTINF_DIRICHLET_EPSILON;
    float inv_col_b[ACTINF_MAX_STATES];
    for (uint8_t k = 0; k < n_states; k++) {
        inv_col_b[k] = 1.0f / (col_sums_b[k] + eps_states);
    }
    for (uint8_t s_prime = 0; s_prime < n_states; s_prime++) {
        float * restrict B_row = agent->B[prev_action][s_prime];
        for (uint8_t k = 0; k < n_states; k++) {
            B_row[k] = (agent->b_counts[s_prime][k][prev_action] + ACTINF_DIRICHLET_EPSILON) * inv_col_b[k];
        }
    }
}

void micro_actinf_learn(micro_actinf_t *agent, uint8_t obs, float learning_rate) {
    if (!agent || obs >= agent->num_obs) return;
    float old_lr_a = agent->learning_cfg.learning_rate_a;
    agent->learning_cfg.learning_rate_a = learning_rate;
    micro_actinf_learn_step(agent, (uint32_t)obs, (uint32_t)agent->last_action);
    agent->learning_cfg.learning_rate_a = old_lr_a;
}
