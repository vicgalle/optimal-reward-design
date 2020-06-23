import jax
import jax.numpy as np
from jax import grad, jit

import rlax


def run_simple_RL():

    def simple_reward(action):
        if action == 0:
            return np.array([1.])
        else:
            return np.array([0.])


    rng = jax.random.PRNGKey(0)

    grad_PG_loss = jit(grad(rlax.policy_gradient_loss))
    w_t = np.array([1.])

    d = 4
    rng, iter_rng = jax.random.split(rng)
    logits = jax.random.normal(iter_rng, shape=(1, d))

    N = 100
    for _ in range(N):
        # sample action given policy
        rng, iter_rng = jax.random.split(rng)
        a = jax.random.categorical(iter_rng, logits)
        
        # observe reward
        r = simple_reward(a)

        # update policy
        logits -= 0.1 * grad_PG_loss(logits, a, r, w_t)
        print(rlax.policy_gradient_loss(logits, a, r, w_t))

    print(logits)

def run_data_coop_game(seed, swf):

    def data_coop_reward(a_C, a_DDO):
        if a_C == 0 and a_DDO == 0:  # both defect
            return np.array([1.]), np.array([1.])
        elif a_C == 0 and a_DDO == 1:
            return np.array([5.]), np.array([0.])
        elif a_C == 1 and a_DDO == 0:
            return np.array([0.]), np.array([5.])
        else:
            return np.array([6.]), np.array([6.])

    rng = jax.random.PRNGKey(seed)

    grad_PG_loss = jit(grad(rlax.policy_gradient_loss))
    w_t = np.array([1.])

    log = False
    d = 2
    rng, iter_rng = jax.random.split(rng)
    logits_C = jax.random.normal(iter_rng, shape=(1, d))
    rng, iter_rng = jax.random.split(rng)
    logits_DDO = jax.random.normal(iter_rng, shape=(1, d))

    N = 500

    r_Cs = []
    r_DDOs = []

    for _ in range(N):
        # sample actions given policies
        rng, iter_rng = jax.random.split(rng)
        a_C = jax.random.categorical(iter_rng, logits_C)
        rng, iter_rng = jax.random.split(rng)
        a_DDO = jax.random.categorical(iter_rng, logits_DDO)
        
        # observe rewards
        r_C, r_DDO = data_coop_reward(a_C, a_DDO)
        r_Cs.append(r_C)
        r_DDOs.append(r_DDO)

        # update policies
        logits_C -= 0.05 * grad_PG_loss(logits_C, a_C, r_C, w_t)
        logits_DDO -= 0.05 * grad_PG_loss(logits_DDO, a_DDO, r_DDO, w_t)
        
        if log:
            print('C', rlax.policy_gradient_loss(logits_C, a_C, r_C, w_t))
            print('DDO', rlax.policy_gradient_loss(logits_DDO, a_DDO, r_DDO, w_t))
            print('SU', 0.5*(r_C + r_DDO))

    print(logits_C, logits_DDO)
    #print(.5 * (np.mean(np.array(r_Cs)) + np.mean(np.array(r_DDOs))))
    return swf(np.array(r_Cs), np.array(r_DDOs))


def run_data_coop_game_with_regulator(seed, swf):

    def data_coop_reward(a_C, a_DDO):
        if a_C == 0 and a_DDO == 0:  # both defect
            return np.array([1.]), np.array([1.])
        elif a_C == 0 and a_DDO == 1:
            return np.array([5.]), np.array([0.])
        elif a_C == 1 and a_DDO == 0:
            return np.array([0.]), np.array([5.])
        else:
            return np.array([6.]), np.array([6.])

    def redistribute(r_C, r_DDO, a_R):
        tax = 0.
        if a_R == 0:
            tax = 0.
        elif a_R == 1:
            tax = 0.1
        elif a_R == 2:
            tax = 0.2
        else:
            tax = 0.3

        wealth = tax * (r_C + r_DDO)
        r_C = r_C - tax * r_C + wealth/2.
        r_DDO = r_DDO - tax * r_DDO + wealth/2.

        return r_C, r_DDO, tax

    rng = jax.random.PRNGKey(seed)

    grad_PG_loss = jit(grad(rlax.policy_gradient_loss))
    w_t = np.array([1.])

    log = False
    d = 2
    rng, iter_rng = jax.random.split(rng)
    logits_C = jax.random.normal(iter_rng, shape=(1, d))
    rng, iter_rng = jax.random.split(rng)
    logits_DDO = jax.random.normal(iter_rng, shape=(1, d))
    rng, iter_rng = jax.random.split(rng)
    logits_R = np.array([[1, 1, 1, 1.]])

    N = 500

    r_Cs = []
    r_DDOs = []
    taxes = []

    for i in range(N):
        # sample actions given policies
        rng, iter_rng = jax.random.split(rng)
        a_C = jax.random.categorical(iter_rng, logits_C)
        rng, iter_rng = jax.random.split(rng)
        a_DDO = jax.random.categorical(iter_rng, logits_DDO)
        rng, iter_rng = jax.random.split(rng)
        a_R = jax.random.categorical(iter_rng, logits_R)
        
        # observe rewards
        r_C, r_DDO = data_coop_reward(a_C, a_DDO)
        r_Cs.append(r_C)
        r_DDOs.append(r_DDO)

        r_C, r_DDO, tax = redistribute(r_C, r_DDO, a_R)
        taxes.append(tax)

        # update policies
        logits_C -= 0.05 * grad_PG_loss(logits_C, a_C, r_C, w_t)
        logits_DDO -= 0.05 * grad_PG_loss(logits_DDO, a_DDO, r_DDO, w_t)
        if i % 10 == 1:
            R = np.array(r_Cs[:10]).mean() + np.array(r_DDOs[:10]).mean()
            logits_R -= 0.05 * grad_PG_loss(logits_R, a_R, .5*np.array([R]), w_t)
        
        if log:
            print('C', rlax.policy_gradient_loss(logits_C, a_C, r_C, w_t))
            print('DDO', rlax.policy_gradient_loss(logits_DDO, a_DDO, r_DDO, w_t))
            print('SU', 0.5*(r_C + r_DDO))

    print('logits:', logits_C, logits_DDO, logits_R)
    print('mean SU:', .5 * (np.mean(np.array(r_Cs)) + np.mean(np.array(r_DDOs))))
    print('mean tax', np.array(taxes).mean())

    return swf(np.array(r_Cs), np.array(r_DDOs))

def run_data_coop_game_with_gaussian_regulator(seed, swf):

    def data_coop_reward(a_C, a_DDO):
        if a_C == 0 and a_DDO == 0:  # both defect
            return np.array([1.]), np.array([1.])
        elif a_C == 0 and a_DDO == 1:
            return np.array([5.]), np.array([0.])
        elif a_C == 1 and a_DDO == 0:
            return np.array([0.]), np.array([5.])
        else:
            return np.array([6.]), np.array([6.])

    def gaussian_logprob(logits, a):
        return np.mean(-((a - logits)/0.05)**2)

    def redistribute(r_C, r_DDO, a_R):
        tax = 0.5*jax.nn.sigmoid(a_R)

        wealth = tax * (r_C + r_DDO)
        r_C = r_C - tax * r_C + wealth/2.
        r_DDO = r_DDO - tax * r_DDO + wealth/2.

        return r_C, r_DDO, tax

    rng = jax.random.PRNGKey(seed)

    grad_PG_loss = jit(grad(rlax.policy_gradient_loss))
    w_t = np.array([1.])

    log = False
    d = 2
    rng, iter_rng = jax.random.split(rng)
    logits_C = jax.random.normal(iter_rng, shape=(1, d))
    rng, iter_rng = jax.random.split(rng)
    logits_DDO = jax.random.normal(iter_rng, shape=(1, d))
    rng, iter_rng = jax.random.split(rng)
    logits_R = np.array([0.])  # the mean of the Gaussian

    N = 500

    r_Cs = []
    r_DDOs = []
    taxes = []

    for i in range(N):
        # sample actions given policies
        rng, iter_rng = jax.random.split(rng)
        a_C = jax.random.categorical(iter_rng, logits_C)
        rng, iter_rng = jax.random.split(rng)
        a_DDO = jax.random.categorical(iter_rng, logits_DDO)
        rng, iter_rng = jax.random.split(rng)
        a_R = 0.05 * jax.random.normal(iter_rng) + logits_R
        
        # observe rewards
        r_C, r_DDO = data_coop_reward(a_C, a_DDO)
        r_Cs.append(r_C)
        r_DDOs.append(r_DDO)

        r_C, r_DDO, tax = redistribute(r_C, r_DDO, a_R)
        taxes.append(tax)

        # update policies
        logits_C -= 0.05 * grad_PG_loss(logits_C, a_C, r_C, w_t)
        logits_DDO -= 0.05 * grad_PG_loss(logits_DDO, a_DDO, r_DDO, w_t)
        if i % 10 == 1:
            R = np.array(r_Cs[:10]).mean() + np.array(r_DDOs[:10]).mean()
            logits_R -= 0.005 * R * grad(gaussian_logprob)(logits_R, a_R)
        
        if log:
            print('C', rlax.policy_gradient_loss(logits_C, a_C, r_C, w_t))
            print('DDO', rlax.policy_gradient_loss(logits_DDO, a_DDO, r_DDO, w_t))
            print('SU', 0.5*(r_C + r_DDO))

    print('logits:', logits_C, logits_DDO, logits_R)
    print('mean SU:', .5 * (np.mean(np.array(r_Cs)) + np.mean(np.array(r_DDOs))))
    print('mean tax', np.array(taxes).mean())

    return 0.5 * (np.array(r_Cs) + np.array(r_DDOs))
    return swf(np.array(r_Cs), np.array(r_DDOs))

def utilitarian_swf(r_Cs, r_DDOs):
    return 0.5 * (r_Cs + r_DDOs)

def rawlsian_swf(r_Cs, r_DDOs):
    rs = np.hstack([r_Cs, r_DDOs])
    return rs.min(axis=1)

