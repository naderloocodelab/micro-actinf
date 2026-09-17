/**
 * @file micro_actinf.c
 * @brief Canonical C11 Core Implementation of Micro-ActInf.
 * @details Zero-allocation, MISRA-inspired deterministic active inference governor.
 * @author naderloocodelab
 * @license MIT
 */

#ifndef MICRO_ACTINF_BUILD_DLL
#define MICRO_ACTINF_BUILD_DLL
#endif

#include "micro_actinf.h"
#include <math.h>
#include <string.h>

/* Branchless, high-precision logarithm approximation for real-time EFE evaluation */
static inline float fast_logf(float x) {
    if (x <= ACTINF_EPSILON) return -27.631021f;
    union { float f; uint32_t i; } u = { x };
    int32_t exp = (int32_t)((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFF) | 0x3F800000;
    float z = u.f - 1.0f;
    float ln_u = z * (0.9999964f + z * (-0.4998741f + z * (0.3317990f - 0.2307510f * z)));
    return ((float)exp * 0.69314718056f) + ln_u;
}

static void softmax_inplace(float *vec, uint8_t n) {
    if (n == 0) return;
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
    agent->a_entropy_dirty = false;
}

void micro_actinf_set_learning_config(micro_actinf_t *agent, const micro_actinf_learning_config_t *cfg) {
    if (!agent || !cfg) return;
    agent->learning_cfg = *cfg;

    if (isnan(agent->learning_cfg.learning_rate_a) || agent->learning_cfg.learning_rate_a < 0.0f) {
        agent->learning_cfg.learning_rate_a = 0.05f;
    } else if (agent->learning_cfg.learning_rate_a > 1.0f) {
        agent->learning_cfg.learning_rate_a = 1.0f;
    }

    if (isnan(agent->learning_cfg.learning_rate_b) || agent->learning_cfg.learning_rate_b < 0.0f) {
        agent->learning_cfg.learning_rate_b = 0.05f;
    } else if (agent->learning_cfg.learning_rate_b > 1.0f) {
        agent->learning_cfg.learning_rate_b = 1.0f;
    }

    if (isnan(agent->learning_cfg.decay_factor) || agent->learning_cfg.decay_factor <= 0.0f) {
        agent->learning_cfg.decay_factor = 0.998f;
    } else if (agent->learning_cfg.decay_factor > 1.0f) {
        agent->learning_cfg.decay_factor = 1.0f;
    }
}

void micro_actinf_set_cost_config(micro_actinf_t *agent, const micro_actinf_cost_config_t *cfg) {
    if (!agent || !cfg) return;
    agent->cost_cfg = *cfg;
    if (agent->cost_cfg.beta_epistemic < 0.0f) agent->cost_cfg.beta_epistemic = 1.0f;
    if (agent->cost_cfg.gamma_discount < 0.0f || agent->cost_cfg.gamma_discount > 1.0f) {
        agent->cost_cfg.gamma_discount = 0.90f;
    }
}

void micro_actinf_set_horizon(micro_actinf_t *agent, uint8_t horizon) {
    if (!agent) return;
    if (horizon < 1) horizon = 1;
    if (horizon > ACTINF_MAX_HORIZON) horizon = ACTINF_MAX_HORIZON;
    agent->horizon = horizon;
}

void micro_actinf_set_alpha_prior(micro_actinf_t *agent, float alpha) {
    if (!agent) return;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    agent->alpha_prior = alpha;
}

void micro_actinf_set_beta_epistemic(micro_actinf_t *agent, float beta) {
    if (!agent) return;
    if (beta < 0.0f) beta = 0.0f;
    agent->cost_cfg.beta_epistemic = beta;
}

void micro_actinf_init(micro_actinf_t *agent, uint8_t states, uint8_t obs, uint8_t actions) {
    if (!agent) return;

    memset(agent, 0, sizeof(micro_actinf_t));

    agent->num_states = (states > 0 && states <= ACTINF_MAX_STATES) ? states : ACTINF_MAX_STATES;
    agent->num_obs = (obs > 0 && obs <= ACTINF_MAX_OBS) ? obs : ACTINF_MAX_OBS;
    agent->num_actions = (actions > 0 && actions <= ACTINF_MAX_ACTIONS) ? actions : ACTINF_MAX_ACTIONS;
    agent->horizon = 2; /* 2-step trajectory planning by default */
    agent->gamma = 2.0f;
    agent->alpha_prior = 0.25f; /* Adaptive prior mixing factor to prevent Bayesian inertia */
    agent->last_action = 0;
    agent->step_count = 0;
    agent->a_entropy_dirty = false;
    agent->progress_index = 0.0f;
    agent->consecutive_loops = 0;
    agent->loop_detected = false;

    /* Initialize default online Dirichlet learning configuration */
    agent->learning_cfg.learning_rate_a = 0.05f;
    agent->learning_cfg.learning_rate_b = 0.05f;
    agent->learning_cfg.decay_factor = 0.998f;
    agent->learning_cfg.enable_learning = true;

    /* Initialize default cost & EFE configuration */
    agent->cost_cfg.beta_epistemic = 1.0f;
    agent->cost_cfg.weight_token_cost = 0.20f;
    agent->cost_cfg.weight_latency = 0.10f;
    agent->cost_cfg.weight_risk = 0.50f;
    agent->cost_cfg.loop_penalty = 2.50f;
    agent->cost_cfg.gamma_discount = 0.90f;

    /* Balanced action cost profiles: [token_cost, latency_cost, base_risk] */
    for (uint8_t u = 0; u < agent->num_actions; u++) {
        if (u == 0) { /* EPISTEMIC_EXPLORE */
            agent->action_costs[u][0] = 0.15f;
            agent->action_costs[u][1] = 0.10f;
            agent->action_costs[u][2] = 0.10f;
            agent->action_risk[u] = ACTINF_RISK_READ;
        } else if (u == 1) { /* PRAGMATIC_EXECUTE */
            agent->action_costs[u][0] = 0.20f;
            agent->action_costs[u][1] = 0.15f;
            agent->action_costs[u][2] = 0.15f;
            agent->action_risk[u] = ACTINF_RISK_EDIT;
        } else if (u == 2) { /* AUDIT_DIAGNOSE */
            agent->action_costs[u][0] = 0.20f;
            agent->action_costs[u][1] = 0.15f;
            agent->action_costs[u][2] = 0.15f;
            agent->action_risk[u] = ACTINF_RISK_ANALYZE;
        } else { /* CONVERGE_CONCLUDE */
            agent->action_costs[u][0] = 0.20f;
            agent->action_costs[u][1] = 0.15f;
            agent->action_costs[u][2] = 0.15f;
            agent->action_risk[u] = ACTINF_RISK_TEST;
        }
    }

    /* Uniform initial beliefs */
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

    /* Initialize transition tensors B with weak persistence & matching Dirichlet priors */
    for (uint8_t u = 0; u < agent->num_actions; u++) {
        for (uint8_t j = 0; j < agent->num_states; j++) {
            for (uint8_t i = 0; i < agent->num_states; i++) {
                if (agent->num_states == 1) {
                    agent->B[u][i][j] = 1.0f;
                } else if (i == j) {
                    agent->B[u][i][j] = 0.50f;
                } else {
                    agent->B[u][i][j] = 0.50f / (float)(agent->num_states - 1);
                }
                agent->b_counts[i][j][u] = agent->B[u][i][j] * (float)agent->num_states;
            }
        }
    }

    /* Neutral prior preferences C */
    for (uint8_t o = 0; o < agent->num_obs; o++) {
        agent->C[o] = 0.0f;
    }

    for (uint8_t u = 0; u < agent->num_actions; u++) {
        agent->G[u] = 0.0f;
        agent->pi[u] = 1.0f / (float)agent->num_actions;
    }

    agent->last_entropy = micro_actinf_shannon_entropy(agent);
    agent->entropy_velocity = 0.0f;
}

void micro_actinf_reset_state(micro_actinf_t *agent) {
    if (!agent) return;
    float p_s = 1.0f / (float)agent->num_states;
    for (uint8_t s = 0; s < agent->num_states; s++) {
        agent->s[s] = p_s;
        agent->s_prev[s] = p_s;
    }
    agent->history_count = 0;
    agent->history_idx = 0;
    agent->consecutive_loops = 0;
    agent->loop_detected = false;
    agent->last_entropy = micro_actinf_shannon_entropy(agent);
    agent->entropy_velocity = 0.0f;
}

uint8_t micro_actinf_get_dominant_state(const micro_actinf_t *agent, float *out_confidence) {
    if (!agent) return 0;
    uint8_t best_s = 0;
    float max_val = agent->s[0];
    for (uint8_t s = 1; s < agent->num_states; s++) {
        if (agent->s[s] > max_val) {
            max_val = agent->s[s];
            best_s = s;
        }
    }
    if (out_confidence) {
        *out_confidence = max_val;
    }
    return best_s;
}

void micro_actinf_step(micro_actinf_t *agent, uint8_t obs) {
    if (!agent || obs >= agent->num_obs) return;

    const uint8_t n_states = agent->num_states;
    const uint8_t last_act = agent->last_action;
    const float alpha = agent->alpha_prior;
    const float uniform_p = 1.0f / (float)n_states;

    float * restrict s = agent->s;
    float * restrict s_prev = agent->s_prev;

    /* Save previous belief vector */
    for (uint8_t i = 0; i < n_states; i++) {
        s_prev[i] = s[i];
    }

    /* 1. Prior state prediction with adaptive recency decay:
     *    s_prior[i] = (1 - alpha) * Sum_j B(last_act)[i][j] * s_prev[j] + alpha * (1/K)
     */
    float s_prior[ACTINF_MAX_STATES];
    for (uint8_t i = 0; i < n_states; i++) {
        float val = 0.0f;
        const float * restrict b_row = agent->B[last_act][i];
        for (uint8_t j = 0; j < n_states; j++) {
            val += b_row[j] * s_prev[j];
        }
        s_prior[i] = (1.0f - alpha) * val + alpha * uniform_p;
    }

    /* 2. Variational Bayes likelihood update: s_{t+1} \propto A_{obs, :} \odot s_prior */
    float sum = 0.0f;
    float unnorm[ACTINF_MAX_STATES];
    const float * restrict a_row = agent->A[obs];

    for (uint8_t i = 0; i < n_states; i++) {
        float p = a_row[i] * s_prior[i];
        unnorm[i] = p;
        sum += p;
    }

    if (sum > ACTINF_EPSILON) {
        float inv_sum = 1.0f / sum;
        for (uint8_t i = 0; i < n_states; i++) {
            s[i] = unnorm[i] * inv_sum;
        }
    } else {
        /* Numerically safe log-space fallback for zero likelihood/edge cases */
        float log_posterior[ACTINF_MAX_STATES];
        for (uint8_t i = 0; i < n_states; i++) {
            float likelihood = (a_row[i] > ACTINF_EPSILON) ? a_row[i] : ACTINF_EPSILON;
            float prior = (s_prior[i] > ACTINF_EPSILON) ? s_prior[i] : ACTINF_EPSILON;
            log_posterior[i] = logf(likelihood) + logf(prior);
        }
        softmax_inplace(log_posterior, n_states);
        for (uint8_t i = 0; i < n_states; i++) {
            s[i] = log_posterior[i];
        }
    }

    /* 3. Track Shannon entropy & entropy velocity */
    float current_entropy = micro_actinf_shannon_entropy(agent);
    agent->entropy_velocity = current_entropy - agent->last_entropy;
    agent->last_entropy = current_entropy;

    /* 4. Loop & Inertia Detection Tracker */
    uint8_t dom_s = micro_actinf_get_dominant_state(agent, NULL);
    actinf_step_record_t *rec = &agent->history[agent->history_idx];
    rec->state_dom = dom_s;
    rec->action = last_act;
    rec->obs = obs;
    rec->entropy = current_entropy;

    agent->history_idx = (agent->history_idx + 1) % ACTINF_HISTORY_LEN;
    if (agent->history_count < ACTINF_HISTORY_LEN) {
        agent->history_count++;
    }

    /* Check if the same (state, action) pair has occurred repeatedly with no entropy change */
    if (agent->history_count >= 3) {
        uint8_t matches = 0;
        for (uint8_t h = 0; h < agent->history_count; h++) {
            if (agent->history[h].state_dom == dom_s && agent->history[h].action == last_act) {
                matches++;
            }
        }
        if (matches >= 3 && fabsf(agent->entropy_velocity) < 0.05f) {
            agent->consecutive_loops++;
            agent->loop_detected = true;
        } else {
            agent->loop_detected = false;
            if (agent->consecutive_loops > 0) agent->consecutive_loops--;
        }
    }

    agent->step_count++;
}

uint8_t micro_actinf_select_action(micro_actinf_t *agent) {
    if (!agent) return 0;

    if (agent->a_entropy_dirty) {
        micro_actinf_update_cache(agent);
    }

    const uint8_t n_states = agent->num_states;
    const uint8_t n_obs = agent->num_obs;
    const uint8_t n_actions = agent->num_actions;
    const uint8_t horizon = agent->horizon;
    const float beta = agent->cost_cfg.beta_epistemic;
    const float w_token = agent->cost_cfg.weight_token_cost;
    const float w_lat = agent->cost_cfg.weight_latency;
    const float w_risk = agent->cost_cfg.weight_risk;
    const float loop_pen = agent->cost_cfg.loop_penalty;
    const float discount = agent->cost_cfg.gamma_discount;

    for (uint8_t u = 0; u < n_actions; u++) {
        float total_G = 0.0f;
        float s_trajectory[ACTINF_MAX_STATES];

        /* Initial trajectory state is current belief s_t */
        for (uint8_t i = 0; i < n_states; i++) {
            s_trajectory[i] = agent->s[i];
        }

        float current_discount = 1.0f;

        for (uint8_t tau = 0; tau < horizon; tau++) {
            /* 1. Predict state transition: s_next = B(u) * s_curr */
            float s_next[ACTINF_MAX_STATES] = {0.0f};
            const float (* restrict B_u)[ACTINF_MAX_STATES] = agent->B[u];
            for (uint8_t i = 0; i < n_states; i++) {
                for (uint8_t j = 0; j < n_states; j++) {
                    s_next[i] += B_u[i][j] * s_trajectory[j];
                }
            }

            /* 2. Predict observation: o_pred = A * s_next */
            float o_pred[ACTINF_MAX_OBS] = {0.0f};
            for (uint8_t o = 0; o < n_obs; o++) {
                const float * restrict a_row = agent->A[o];
                for (uint8_t i = 0; i < n_states; i++) {
                    o_pred[o] += a_row[i] * s_next[i];
                }
            }

            /* 3. Pragmatic Value: Sum_o o_pred[o] * C[o] */
            float pragmatic = 0.0f;
            for (uint8_t o = 0; o < n_obs; o++) {
                pragmatic += o_pred[o] * agent->C[o];
            }

            /* 4. Epistemic Value (Mutual Information I(O; S)):
             *    H(O_pred) - Sum_s s_next[s] * H(O | S=s)
             */
            float epistemic = 0.0f;
            for (uint8_t o = 0; o < n_obs; o++) {
                if (o_pred[o] > ACTINF_EPSILON) {
                    epistemic -= o_pred[o] * fast_logf(o_pred[o]);
                }
            }
            for (uint8_t i = 0; i < n_states; i++) {
                epistemic += s_next[i] * agent->A_entropy[i];
            }

            /* 5. Quantitative Action Costs: Token + Latency + Risk */
            float action_cost = w_token * agent->action_costs[u][0] +
                                w_lat   * agent->action_costs[u][1] +
                                w_risk  * agent->action_costs[u][2];

            /* 6. Cognitive Regime Congruence Penalty (Out-of-regime action suppression) */
            static const float REGIME_CONGRUENCE[ACTINF_MAX_ACTIONS][ACTINF_MAX_STATES] = {
                { 0.0f, 2.0f, 2.0f, 2.5f, 2.0f, 2.0f }, /* u=0: EPISTEMIC_EXPLORE */
                { 2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f }, /* u=1: PRAGMATIC_EXECUTE */
                { 2.0f, 2.0f, 2.0f, 0.0f, 2.0f, 2.0f }, /* u=2: AUDIT_DIAGNOSE    */
                { 2.5f, 2.0f, 2.0f, 2.0f, 0.0f, 0.0f }  /* u=3: CONVERGE_CONCLUDE  */
            };
            for (uint8_t j = 0; j < n_states; j++) {
                if (j < 6 && u < 4) {
                    action_cost += s_trajectory[j] * REGIME_CONGRUENCE[u][j];
                }
            }

            /* 7. Repetition / Loop Avoidance Penalty */
            if (agent->loop_detected && u == agent->last_action) {
                action_cost += loop_pen * (float)agent->consecutive_loops;
            }

            /* Step EFE: G_tau = -(Pragmatic + beta * Epistemic) + Cost */
            float G_tau = -(pragmatic + beta * epistemic) + action_cost;
            total_G += current_discount * G_tau;

            /* Advance state for next step in horizon */
            for (uint8_t i = 0; i < n_states; i++) {
                s_trajectory[i] = s_next[i];
            }
            current_discount *= discount;
        }

        agent->G[u] = total_G;
    }

    /* Compute policy probabilities: pi(u) = Softmax(-gamma * G) */
    float logits[ACTINF_MAX_ACTIONS];
    for (uint8_t u = 0; u < n_actions; u++) {
        logits[u] = -agent->gamma * agent->G[u];
    }
    softmax_inplace(logits, n_actions);
    for (uint8_t u = 0; u < n_actions; u++) {
        agent->pi[u] = logits[u];
    }

    /* Greedy selection of minimum Free Energy action */
    uint8_t best_u = 0;
    float max_prob = agent->pi[0];
    for (uint8_t u = 1; u < n_actions; u++) {
        if (agent->pi[u] > max_prob) {
            max_prob = agent->pi[u];
            best_u = u;
        }
    }

    agent->last_action = best_u;
    return best_u;
}

actinf_verdict_t micro_actinf_evaluate_action(micro_actinf_t *agent,
                                              uint8_t proposed_action,
                                              actinf_risk_level_t risk,
                                              float confidence_threshold) {
    if (!agent || proposed_action >= agent->num_actions) {
        return ACTINF_VERDICT_DENY;
    }

    /* 1. Loop / Inertia Check: Deny repeat action if trapped in loop */
    if (agent->loop_detected && proposed_action == agent->last_action) {
        return ACTINF_VERDICT_DENY;
    }

    uint8_t saved_last_action = agent->last_action;
    float conf = 0.0f;
    uint8_t dom_s = micro_actinf_get_dominant_state(agent, &conf);
    uint8_t optimal_action = micro_actinf_select_action(agent);
    agent->last_action = saved_last_action; /* Preserve previous executed action */

    /* 2. Destructive Risk Gating: Destructive operations ALWAYS require confirmation */
    if (risk >= ACTINF_RISK_DESTRUCTIVE) {
        return ACTINF_VERDICT_ASK_CONFIRMATION;
    }

    /* 3. Execution / Script Risk Gating */
    if (risk == ACTINF_RISK_EXECUTE) {
        if (conf < confidence_threshold) {
            return ACTINF_VERDICT_ASK_CONFIRMATION;
        }
    }

    /* 4. Cognitive Regime Compatibility:
     * If agent is in EXPLORATION (0) or DEBUGGING (3), high-risk write/edit should be modified or confirmed.
     */
    if (dom_s == 0 && (risk == ACTINF_RISK_EDIT || risk == ACTINF_RISK_EXECUTE)) {
        return ACTINF_VERDICT_MODIFY;
    }

    /* If proposed action matches optimal EFE policy */
    if (proposed_action == optimal_action) {
        return ACTINF_VERDICT_ALLOW;
    }

    /* Default allow for passive/read actions */
    if (risk <= ACTINF_RISK_ANALYZE) {
        return ACTINF_VERDICT_ALLOW;
    }

    return ACTINF_VERDICT_ALLOW;
}

void micro_actinf_record_outcome(micro_actinf_t *agent,
                                 uint8_t action,
                                 uint8_t outcome_obs,
                                 bool success,
                                 float progress_delta) {
    if (!agent || action >= agent->num_actions || outcome_obs >= agent->num_obs) return;

    /* Update last executed action */
    agent->last_action = action;

    /* 1. Update Dirichlet counts based on transition and observation */
    micro_actinf_learn_step(agent, outcome_obs, action);

    /* 2. Credit Assignment: Update Prior Preferences C(o) based on utility */
    if (success) {
        agent->C[outcome_obs] += 0.15f * (progress_delta > 0.0f ? progress_delta : 1.0f);
        if (agent->C[outcome_obs] > 2.0f) agent->C[outcome_obs] = 2.0f;
        agent->progress_index += (progress_delta > 0.0f ? progress_delta : 0.10f);
        if (agent->progress_index > 1.0f) agent->progress_index = 1.0f;
    } else {
        agent->C[outcome_obs] -= 0.25f;
        if (agent->C[outcome_obs] < -2.0f) agent->C[outcome_obs] = -2.0f;
        if (agent->progress_index >= 0.05f) {
            agent->progress_index -= 0.05f;
        } else {
            agent->progress_index = 0.0f;
        }
    }
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

    /* 1. Observation Pseudo-Count Update */
    float * restrict a_obs = agent->a_counts[observation];
    for (uint8_t k = 0; k < n_states; k++) {
        a_obs[k] = lambda_a * a_obs[k] + eta_a * s[k];
    }

    /* 2. Transition Pseudo-Count Update */
    for (uint8_t s_prime = 0; s_prime < n_states; s_prime++) {
        const float eta_s_prime = eta_b * s[s_prime];
        for (uint8_t k = 0; k < n_states; k++) {
            agent->b_counts[s_prime][k][prev_action] = 
                lambda_b * agent->b_counts[s_prime][k][prev_action] + eta_s_prime * s_prev[k];
        }
    }

    /* 3. Expectation Mapping (Column-normalized categorical expectations) */
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

    agent->a_entropy_dirty = true;
}

void micro_actinf_sync_counts_from_matrices(micro_actinf_t *agent) {
    if (!agent) return;
    const uint8_t n_states = agent->num_states;
    const uint8_t n_obs = agent->num_obs;
    const uint8_t n_actions = agent->num_actions;

    for (uint8_t s = 0; s < n_states; s++) {
        for (uint8_t o = 0; o < n_obs; o++) {
            agent->a_counts[o][s] = agent->A[o][s] * (float)n_obs;
        }
    }

    for (uint8_t u = 0; u < n_actions; u++) {
        for (uint8_t j = 0; j < n_states; j++) {
            for (uint8_t i = 0; i < n_states; i++) {
                agent->b_counts[i][j][u] = agent->B[u][i][j] * (float)n_states;
            }
        }
    }

    micro_actinf_update_cache(agent);
}
