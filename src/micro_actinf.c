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

void micro_actinf_init(micro_actinf_t *agent, uint8_t states, uint8_t obs, uint8_t actions) {
    if (!agent) return;

    agent->num_states = (states > 0 && states <= ACTINF_MAX_STATES) ? states : ACTINF_MAX_STATES;
    agent->num_obs = (obs > 0 && obs <= ACTINF_MAX_OBS) ? obs : ACTINF_MAX_OBS;
    agent->num_actions = (actions > 0 && actions <= ACTINF_MAX_ACTIONS) ? actions : ACTINF_MAX_ACTIONS;
    agent->gamma = 2.0f;
    agent->last_action = 0;
    agent->step_count = 0;

    /* Initialize uniform prior belief */
    float p_s = 1.0f / (float)agent->num_states;
    for (uint8_t s = 0; s < agent->num_states; s++) {
        agent->s[s] = p_s;
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

    /* Initialize transition matrices B with weak self-persistence */
    for (uint8_t u = 0; u < agent->num_actions; u++) {
        for (uint8_t j = 0; j < agent->num_states; j++) {
            for (uint8_t i = 0; i < agent->num_states; i++) {
                if (i == j) {
                    agent->B[u][i][j] = 0.50f;
                } else {
                    agent->B[u][i][j] = 0.50f / (float)(agent->num_states - 1);
                }
                agent->b_counts[u][i][j] = 1.0f;
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

    /* 1. Prior state prediction via previous action: s_prior = B(u_{t-1}) * s_t */
    float s_prior[ACTINF_MAX_STATES] = {0.0f};
    for (uint8_t i = 0; i < agent->num_states; i++) {
        for (uint8_t j = 0; j < agent->num_states; j++) {
            s_prior[i] += agent->B[agent->last_action][i][j] * agent->s[j];
        }
    }

    /* 2. Variational Bayes belief update: ln(s) = ln(A[o, :]) + ln(s_prior) */
    float log_posterior[ACTINF_MAX_STATES];
    for (uint8_t i = 0; i < agent->num_states; i++) {
        float likelihood = (agent->A[obs][i] > ACTINF_EPSILON) ? agent->A[obs][i] : ACTINF_EPSILON;
        float prior = (s_prior[i] > ACTINF_EPSILON) ? s_prior[i] : ACTINF_EPSILON;
        log_posterior[i] = logf(likelihood) + logf(prior);
    }

    /* 3. Normalize via Softmax */
    softmax_inplace(log_posterior, agent->num_states);
    for (uint8_t i = 0; i < agent->num_states; i++) {
        agent->s[i] = log_posterior[i];
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

void micro_actinf_learn(micro_actinf_t *agent, uint8_t obs, float learning_rate) {
    if (!agent || obs >= agent->num_obs) return;

    /* Dirichlet count updates for A matrix */
    for (uint8_t s = 0; s < agent->num_states; s++) {
        agent->a_counts[obs][s] += learning_rate * agent->s[s];
    }

    /* Normalize columns of A */
    for (uint8_t s = 0; s < agent->num_states; s++) {
        float col_sum = 0.0f;
        for (uint8_t o = 0; o < agent->num_obs; o++) {
            col_sum += agent->a_counts[o][s];
        }
        if (col_sum > ACTINF_EPSILON) {
            float inv_col = 1.0f / col_sum;
            for (uint8_t o = 0; o < agent->num_obs; o++) {
                agent->A[o][s] = agent->a_counts[o][s] * inv_col;
            }
        }
    }

    micro_actinf_update_cache(agent);
}
