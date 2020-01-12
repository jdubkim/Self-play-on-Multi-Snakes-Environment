import os
import time
import sys
sys.path.append('../')
import datetime
import joblib
import numpy as np
import os.path
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
import multiprocessing

import random
import gym
from config import Config
import utils

class MultiModel(object):
    def __init__(self, main_model, opponent_models: list):
        self.opponent_models = opponent_models

        def multi_step(obs, opponents_obs, states, dones):
            actions, values, ret_sts, neglogpacs = main_model.step(obs, states, dones)

            opponent_actions = [[] for _ in range(len(self.opponent_models))]

            for i, opponent in enumerate(self.opponent_models):
                if opponent is None:
                    opponent_actions[i] = [1 for i in range(len(actions))]
                else:
                    opponent_actions[i], _, _, _ = self.opponent_models[i].step(opponents_obs[i], states, dones)

            actions_list = opponent_actions.copy()
            actions_list.insert(0, actions)
            self.full_actions = list(zip(*actions_list)) #TODO: Check if this code works | Done: Find a way to zip n lists
            print("---- TEST ----")
            print("action is : ", actions)
            print("opponent actions is : ", opponent_actions)
            print("full actions is : ", self.full_actions)
            return actions, values, ret_sts, neglogpacs

        self.multi_step = multi_step
        self.value = main_model.value

class Model(object):
    def __init__(self, *, policy, ob_shape, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, scope_name):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_shape, ac_space, nbatch_act, 1, scope_name, reuse=False)
        train_model = policy(sess, ob_shape, ac_space, nbatch_train, nsteps, scope_name, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        with tf.variable_scope(scope_name):
            params = tf.trainable_variables(scope_name)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_file):
            ps = sess.run(params)
            joblib.dump(ps, Config.MODEL_DIR + Config.EXPR_DIR + save_file)

        update_placeholders = []
        update_ops = []

        for p in params:
            update_placeholder = tf.placeholder(p.dtype, shape=p.get_shape())
            update_placeholders.append(update_placeholder)
            update_op = p.assign(update_placeholder)
            update_ops.append(update_op)

        def load(load_file):
            loaded_params = joblib.load(Config.MODEL_DIR + Config.EXPR_DIR + load_file)

            feed_dict = {}

            for update_placeholder, loaded_p in zip(update_placeholders, loaded_params):
                feed_dict[update_placeholder] = loaded_p

            sess.run(update_ops, feed_dict=feed_dict)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101


class Runner(object):
    def __init__(self, *, env, model, opponent_models, nsteps, gamma, lam):
        self.env = env
        self.model = MultiModel(model, opponent_models)
        nenv = env.num_envs
        input_shape = utils.get_shape(env.observation_space)
        self.main_obs = np.zeros((nenv,) + input_shape, dtype=model.train_model.X.dtype.name)
        self.opponent_obs = [] # list
        self.n_opponents = len(opponent_models)
        for i in range(self.n_opponents):
            if len(self.opponent_obs) >= self.n_opponents:
                break
            oppo_obs = np.zeros((nenv,) + input_shape, dtype=model.train_model.X.dtype.name)
            self.opponent_obs.append(oppo_obs)

        multi_agent_obs = env.reset()
        self.use_multi_agent_obs(multi_agent_obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def use_multi_agent_obs(self, multi_agent_obs):
        self.main_obs[:] = multi_agent_obs[:, :, :, 0:3]

        for i in range(self.n_opponents):
            self.opponent_obs[i][:] = multi_agent_obs[:, :, :, 3 * (i + 1): 3 * (i + 2)]

    def run(self):
        mb_obs = []
        mb_rewards = []
        mb_actions = []
        mb_values = []
        mb_dones = []
        mb_neglogpacs = []

        mb_states = self.states
        epinfos = []

        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = \
                self.model.multi_step(self.main_obs, self.opponent_obs, self.states, self.dones)
            mb_obs.append(self.main_obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            multi_agent_obs, rewards, self.dones, infos = self.env.step(self.model.full_actions)
            self.use_multi_agent_obs(multi_agent_obs)

            for info in infos:
                temp_info = info.get('episode')
                if temp_info:
                    epinfos.append(temp_info)

            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.main_obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.main_obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0):

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)

    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)

    total_timesteps = int(total_timesteps)

    # Log
    csv_writer = logger.CSVOutputFormat('{0}.csv'.format(Config.EXPR_NAME))
    tensorboard_writer = logger.TensorBoardOutputFormat('./tensorboard/ppo/')

    nenvs = env.num_envs
    ob_shape = utils.get_shape(env.observation_space)
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda scope_name: Model(policy=policy, ob_shape=ob_shape, ac_space=ac_space, nbatch_act=nenvs,
                                          nbatch_train=nbatch_train,
                                          nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                          max_grad_norm=max_grad_norm, scope_name=scope_name)

    if save_interval and logger.get_dir():
        import cloudpickle
        with open(os.path.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))

    model = make_model(Config.PRIMARY_MODEL_SCOPE)
    opponent_models = []

    baseline_file = None

    for i in range(Config.NUM_SNAKES - 1):
        opponent_model = make_model(Config.OPPONENT_MODEL_SCOPE[i])
        opponent_models.append(opponent_model)

    if baseline_file is not None:
        model.load(baseline_file)
        print("opponent model ")
        for opponent in opponent_models:
            if opponent is not None:
                opponent.load(baseline_file)

    runner = Runner(env=env, model=model, opponent_models=opponent_models,
                    nsteps=nsteps, gamma=gamma, lam=lam)

    maxlen = 100
    epinfobuf = deque(maxlen=maxlen)
    first_start_time = time.time()

    next_highscore = 5
    highscore_interval = 1

    opponent_save_interval = Config.OPPONENT_SAVE_INTERVAL
    max_saved_opponents = Config.MAX_SAVED_OPPONENTS

    model_idx = 0

    opponents_idx = [0 for _ in range(Config.NUM_SNAKES - 1)]
    num_opponents = [0 for _ in range(Config.NUM_SNAKES - 1)]

    for i in range(Config.NUM_SNAKES - 1):
        model.save(utils.get_opponent_file(i, opponents_idx[i]))
        opponents_idx[i] += 1
        num_opponents[i] += 1

    nupdates = total_timesteps // nbatch

    selected_opponents_idx = [0 for _ in range(len(opponent_models))]

    for update in range(1, nupdates + 1):
        for i, opponent_model in enumerate(opponent_models):
            if opponent_model is not None:
                selected_opponents_idx[i] = random.randint(0, max(num_opponents[i] - 1, 0))
                print('Loading checkpoint ' + str(selected_opponents_idx[i]) + '...')
                opponent_model.load(utils.get_opponent_file(i, selected_opponents_idx[i]))

        assert nbatch % nminibatches == 0
        print("here")
        nbatch_train = nbatch // nminibatches
        start_time = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        lossvals = np.mean(mblossvals, axis=0)
        current_time = time.time()
        fps = int(nbatch / (current_time - start_time))

        ep_rew_mean = safemean([epinfo['r'] for epinfo in epinfobuf])

        print("opponent models is ", opponent_models)
        for i, opponent_model in enumerate(opponent_models) :
            if update % opponent_save_interval == 0 and opponent_model is not None:
                print('Saving opponent model{0} {1} ...'.format(i, opponents_idx[i]))

                model.save(utils.get_opponent_file(i, opponents_idx[i]))

                opponents_idx[i] += 1
                num_opponents[i] = max(opponents_idx[i], num_opponents[i])
                opponents_idx[i] = opponents_idx[i] % max_saved_opponents

        if update % log_interval == 0 or update == 1:

            logger.logkv('num_opponents', num_opponents[0])
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            # logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean ' + str(maxlen), ep_rew_mean)
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', current_time - first_start_time)
            # logger.logkv('nenvs nsteps nmb nopte', [nenvs, nsteps, nminibatches, noptepochs])
            logger.logkv('ep_rew_mean', ep_rew_mean)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)

            kvs = logger.getkvs()
            csv_writer.writekvs(kvs)
            tensorboard_writer.writekvs(kvs)
            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            model.save('snake_model_num{0}_{1}.pkl'.format(Config.NUM_SNAKES, model_idx))
            model_idx += 1

        if (ep_rew_mean > next_highscore) and Config.NUM_SNAKES == 1:
            print('saving agent with new highscore ', next_highscore, '...')
            next_highscore += highscore_interval
            model.save('highscore_model.pkl')

    model.save('snake_model_num{0}_{1}_final.pkl'.format(Config.NUM_SNAKES, model_idx))

    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

