# Author: Aqeel Anwar(ICSRL)
# Created: 4/14/2020, 7:15 AM
# Email: aqeel.anwar@gatech.edu

import tensorflow as tf
import numpy as np
from network.loss_functions import huber_loss, mse_loss
from network.network import *
from numpy import linalg as LA

class initialize_network_DeepQLearning():
    def __init__(self, cfg, name, vehicle_name):
        self.g = tf.Graph()
        self.vehicle_name = vehicle_name

        self.first_frame = True
        self.last_frame = []
        with self.g.as_default():
            stat_writer_path = cfg.network_path + self.vehicle_name + '/return_plot/'
            loss_writer_path = cfg.network_path + self.vehicle_name + '/loss' + name + '/'
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)
            # name_array = 'D:/train/loss'+'/'+name
            self.loss_writer = tf.summary.FileWriter(loss_writer_path)
            self.env_type = cfg.env_type
            self.input_size = cfg.input_size
            self.num_actions = cfg.num_actions

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, cfg.input_size, cfg.input_size, 3], name='States')

            # self.X = tf.image.resize_images(self.X1, (227, 227))

            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            self.target = tf.placeholder(tf.float32, shape=[None], name='Qvals')
            self.actions = tf.placeholder(tf.int32, shape=[None], name='Actions')

            # self.model = AlexNetDuel(self.X, cfg.num_actions, cfg.train_fc)
            self.model = C3F2(self.X, cfg.num_actions, cfg.train_fc)

            self.predict = self.model.output
            ind = tf.one_hot(self.actions, cfg.num_actions)
            pred_Q = tf.reduce_sum(tf.multiply(self.model.output, ind), axis=1)
            self.loss = huber_loss(pred_Q, self.target)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.loss, name="train")

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()
            self.all_vars = tf.trainable_variables()

            self.sess.graph.finalize()

        # Load custom weights from custom_load_path if required
        if cfg.custom_load:
            print('Loading weights from: ', cfg.custom_load_path)
            self.load_network(cfg.custom_load_path)


    def get_vars(self):
        return self.sess.run(self.all_vars)


    def initialize_graphs_with_average(self, agent, agent_on_same_network):
        values = {}
        var = {}
        all_assign = {}
        for name_agent in agent_on_same_network:
            values[name_agent] = agent[name_agent].network_model.get_vars()
            var[name_agent] = agent[name_agent].network_model.all_vars
            all_assign[name_agent] = []

        for i in range(len(values[name_agent])):
            val = []
            for name_agent in agent_on_same_network:
                val.append(values[name_agent][i])
            # Take mean here
            mean_val = np.average(val, axis=0)
            for name_agent in agent_on_same_network:
                # all_assign[name_agent].append(tf.assign(var[name_agent][i], mean_val))
                var[name_agent][i].load(mean_val, agent[name_agent].network_model.sess)


    def Q_val(self, xs):
        target = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        return self.sess.run(self.predict,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                        self.target: target, self.actions: actions})


    def train_n(self, xs, ys, actions, batch_size, dropout_rate, lr, epsilon, iter):
        _, loss, Q = self.sess.run([self.train, self.loss, self.predict],
                                   feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.X1: xs,
                                              self.target: ys, self.actions: actions})
        meanQ = np.mean(Q)
        maxQ = np.max(Q)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss', group=self.vehicle_name, value=LA.norm(loss) / batch_size, index=iter)
        self.log_to_tensorboard(tag='Epsilon', group=self.vehicle_name, value=epsilon, index=iter)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=iter)
        self.log_to_tensorboard(tag='MeanQ', group=self.vehicle_name, value=meanQ, index=iter)
        self.log_to_tensorboard(tag='MaxQ', group=self.vehicle_name, value=maxQ, index=iter)


    def action_selection(self, state):
        target = np.zeros(shape=[state.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[state.shape[0]])
        qvals = self.sess.run(self.predict,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.target: target, self.actions: actions})

        if qvals.shape[0] > 1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
        else:
            # Evaluating one sample
            action = np.zeros(1)
            action[0] = np.argmax(qvals)

        return action.astype(int)


    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)


    def save_network(self, save_path, episode=''):
        save_path = save_path + self.vehicle_name + '/' + self.vehicle_name + '_' + str(episode)
        self.saver.save(self.sess, save_path)
        print('Model Saved: ', save_path)


    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)


    def get_weights(self):
        xs = np.zeros(shape=(32, 227, 227, 3))
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        ys = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        return self.sess.run(self.weights,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0,
                                        self.X1: xs,
                                        self.target: ys, self.actions: actions})



###########################################################################
# DeepREINFORCE: Class
###########################################################################


class initialize_network_DeepREINFORCE():
    def __init__(self, cfg, name, vehicle_name):
        self.g = tf.Graph()
        self.vehicle_name = vehicle_name
        self.iter_baseline = 0
        self.iter_policy = 0
        self.first_frame = True
        self.last_frame = []
        self.iter_combined = 0
        with self.g.as_default():
            stat_writer_path = cfg.network_path + self.vehicle_name + '/return_plot/'
            loss_writer_path = cfg.network_path + self.vehicle_name + '/loss' + name + '/'
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)
            # name_array = 'D:/train/loss'+'/'+name
            self.loss_writer = tf.summary.FileWriter(loss_writer_path)
            self.env_type = cfg.env_type
            self.input_size = cfg.input_size
            self.num_actions = cfg.num_actions

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, cfg.input_size, cfg.input_size, 3], name='States')

            # self.X = tf.image.resize_images(self.X1, (227, 227))

            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            # self.target = tf.placeholder(tf.float32, shape=[None], name='action_probs')
            # self.target_baseline = tf.placeholder(tf.float32, shape=[None], name='baseline')
            self.actions = tf.placeholder(tf.int32, shape=[None, 1], name='Actions')
            self.G = tf.placeholder(tf.float32, shape=[None, 1], name='G')
            self.B = tf.placeholder(tf.float32, shape=[None, 1], name='B')

            # Select the deep network
            self.model = C3F2_REINFORCE_with_baseline(self.X, cfg.num_actions, cfg.train_fc)

            self.predict = self.model.output
            self.baseline = self.model.baseline
            self.ind = tf.one_hot(tf.squeeze(self.actions), cfg.num_actions)
            self.prob_action = tf.reduce_sum(tf.multiply(self.predict, self.ind), axis=1)

            loss_policy = tf.reduce_mean(tf.log(tf.transpose([self.prob_action])) * (self.G - self.B))
            loss_entropy = -tf.reduce_mean(tf.multiply((tf.log(self.predict) + 1e-8), self.predict))

            self.loss_main = -loss_policy - .2 * loss_entropy
            self.loss_branch = mse_loss(self.baseline, self.G)

            self.train_main = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.loss_main, name="train_main")

            self.train_branch = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                                                       beta2=0.99).minimize(
                self.loss_branch, name="train_branch")

            # self.train_combined = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
            #                                            beta2=0.99).minimize(
            #     self.loss_combined, name="train_combined")

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()
            self.all_vars = tf.trainable_variables()

            self.sess.graph.finalize()

        # Load custom weights from custom_load_path if required
        if cfg.custom_load:
            print('Loading weights from: ', cfg.custom_load_path)
            self.load_network(cfg.custom_load_path)

    def get_vars(self):
        return self.sess.run(self.all_vars)

    def initialize_graphs_with_average(self, agent, agent_on_same_network):
        values = {}
        var = {}
        all_assign = {}
        for name_agent in agent_on_same_network:
            values[name_agent] = agent[name_agent].network_model.get_vars()
            var[name_agent] = agent[name_agent].network_model.all_vars
            all_assign[name_agent] = []

        for i in range(len(values[name_agent])):
            val = []
            for name_agent in agent_on_same_network:
                val.append(values[name_agent][i])
            # Take mean here
            mean_val = np.average(val, axis=0)
            for name_agent in agent_on_same_network:
                # all_assign[name_agent].append(tf.assign(var[name_agent][i], mean_val))
                var[name_agent][i].load(mean_val, agent[name_agent].network_model.sess)

    def prob_actions(self, xs):
        G = np.zeros(shape=[1], dtype=np.float32)
        B = np.zeros(shape=[1], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        return self.sess.run(self.predict,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                        self.actions: actions,
                                        self.B: B,
                                        self.G: G})

    def train_baseline(self, xs, G, actions, lr, iter):
        self.iter_baseline += 1
        batch_size = xs.shape[0]
        B = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)

        _, loss, baseline_val = self.sess.run([self.train_branch, self.loss_branch, self.baseline],
                                              feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                                         self.X1: xs,
                                                         self.actions: actions,
                                                         self.B: B,
                                                         self.G: G})

        max_baseline = np.max(baseline_val)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss_Baseline', group=self.vehicle_name, value=loss / batch_size,
                                index=self.iter_baseline)
        # self.log_to_tensorboard(tag='Epsilon', group=self.vehicle_name, value=epsilon, index=iter)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=self.iter_baseline)
        # self.log_to_tensorboard(tag='MeanQ', group=self.vehicle_name, value=meanQ, index=iter)
        self.log_to_tensorboard(tag='Max_baseline', group=self.vehicle_name, value=max_baseline,
                                index=self.iter_baseline)

        return baseline_val

    def get_baseline(self, xs):
        lr = 0
        actions = np.zeros(dtype=int, shape=[xs.shape[0], 1])
        B = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        G = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)

        baseline = self.sess.run(self.baseline,
                                 feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                            self.X1: xs,
                                            self.actions: actions,
                                            self.B: B,
                                            self.G: G})
        return baseline

    def train_policy(self, xs, actions, B, G, lr, iter):
        self.iter_policy += 1
        batch_size = xs.shape[0]
        train_eval = self.train_main
        loss_eval = self.loss_main
        predict_eval = self.predict

        _, loss, ProbActions = self.sess.run([train_eval, loss_eval, predict_eval],
                                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                                        self.X1: xs,
                                                        self.actions: actions,
                                                        self.B: B,
                                                        self.G: G})

        MaxProbActions = np.max(ProbActions)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss_Policy', group=self.vehicle_name, value=LA.norm(loss) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=self.iter_policy)
        self.log_to_tensorboard(tag='MaxProb', group=self.vehicle_name, value=MaxProbActions, index=self.iter_policy)

    def action_selection(self, state):
        action = np.zeros(dtype=int, shape=[state.shape[0], 1])
        probs = self.sess.run(self.predict,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.actions: action})

        for j in range(probs.shape[0]):
            action[j] = np.random.choice(self.num_actions, 1, p=probs[j])[0]

        return action.astype(int)

    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)

    def save_network(self, save_path, episode=''):
        save_path = save_path + self.vehicle_name + '/' + self.vehicle_name + '_' + str(episode)
        self.saver.save(self.sess, save_path)
        print('Model Saved: ', save_path)

    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)



# ADDITION PPO!!!!!!!!!!!!!!!!!!

###########################################################################
# DeepPPO: Class
###########################################################################


class initialize_network_DeepPPO():
    def __init__(self, cfg, name, vehicle_name):
        self.g = tf.Graph()
        self.vehicle_name = vehicle_name
        self.iter_baseline = 0
        self.iter_policy = 0
        self.first_frame = True
        self.last_frame = []
        self.iter_combined = 0
        with self.g.as_default():
            stat_writer_path = cfg.network_path + self.vehicle_name + '/return_plot/'
            loss_writer_path = cfg.network_path + self.vehicle_name + '/loss' + name + '/'
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)
            # name_array = 'D:/train/loss'+'/'+name
            self.loss_writer = tf.summary.FileWriter(loss_writer_path)
            self.env_type = cfg.env_type
            self.input_size = cfg.input_size
            self.num_actions = cfg.num_actions
            self.eps_clip = cfg.eps_clip

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, cfg.input_size, cfg.input_size, 3], name='States')

            # self.X = tf.image.resize_images(self.X1, (227, 227))

            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            # self.target = tf.placeholder(tf.float32, shape=[None], name='action_probs')
            # self.target_baseline = tf.placeholder(tf.float32, shape=[None], name='baseline')
            self.actions = tf.placeholder(tf.int32, shape=[None, 1], name='Actions')
            self.TD_target = tf.placeholder(tf.float32, shape=[None, 1], name='TD_target')
            self.prob_old = tf.placeholder(tf.float32, shape=[None, 1], name='prob_old')
            self.GAE = tf.placeholder(tf.float32, shape=[None, 1], name='GAE')

            # Select the deep network
            self.model = C3F2_ActorCriticShared(self.X, cfg.num_actions, cfg.train_fc)
            self.pi = self.model.action_probs
            self.state_value = self.model.state_value

            self.ind = tf.one_hot(tf.squeeze(self.actions), cfg.num_actions)
            self.pi_a = tf.expand_dims(tf.reduce_sum(tf.multiply(self.pi, self.ind), axis=1), axis=1)

            self.ratio = tf.exp(tf.log(self.pi_a+1e-10) - tf.log(self.prob_old+1e-10))
            p1 = self.ratio * self.GAE
            p2 = tf.clip_by_value(self.ratio, 1-self.eps_clip, 1+self.eps_clip)*self.GAE

            # Define losses
            self.loss_actor_op = -tf.reduce_mean(tf.minimum(p1, p2))
            self.loss_entropy = 0.5*tf.reduce_mean(tf.multiply((tf.log(self.pi) + 1e-8), self.pi))
            self.loss_critic_op = 0.5*mse_loss(self.state_value, self.TD_target)

            self.loss_op = self.loss_critic_op + self.loss_actor_op + self.loss_entropy

            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.loss_op, name="train_main")

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()
            self.all_vars = tf.trainable_variables()

            self.sess.graph.finalize()

        # Load custom weights from custom_load_path if required
        if cfg.custom_load:
            print('Loading weights from: ', cfg.custom_load_path)
            self.load_network(cfg.custom_load_path)

    def get_vars(self):
        return self.sess.run(self.all_vars)

    def initialize_graphs_with_average(self, agent, agent_on_same_network):
        values = {}
        var = {}
        all_assign = {}
        for name_agent in agent_on_same_network:
            values[name_agent] = agent[name_agent].network_model.get_vars()
            var[name_agent] = agent[name_agent].network_model.all_vars
            all_assign[name_agent] = []

        for i in range(len(values[name_agent])):
            val = []
            for name_agent in agent_on_same_network:
                val.append(values[name_agent][i])
            # Take mean here
            mean_val = np.average(val, axis=0)
            for name_agent in agent_on_same_network:
                # all_assign[name_agent].append(tf.assign(var[name_agent][i], mean_val))
                var[name_agent][i].load(mean_val, agent[name_agent].network_model.sess)

    def prob_actions(self, xs):
        TD_target = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        prob_old = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        GAE = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        return self.sess.run(self.pi,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                        self.actions: actions,
                                        self.TD_target: TD_target,
                                        self.prob_old: prob_old,
                                        self.GAE: GAE})


    def get_state_value(self, xs):
        lr = 0
        actions = np.zeros(dtype=int, shape=[xs.shape[0], 1])
        TD_target = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        prob_old = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)
        GAE = np.zeros(shape=[xs.shape[0], 1], dtype=np.float32)

        baseline = self.sess.run(self.state_value,
                                 feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                            self.X1: xs,
                                            self.actions: actions,
                                            self.TD_target: TD_target,
                                            self.prob_old: prob_old,
                                            self.GAE: GAE})
        return baseline

    def train_policy(self, xs, actions, TD_target, prob_old, GAE, lr, iter):
        self.iter_policy += 1
        batch_size = xs.shape[0]
        train_eval = self.train_op
        loss_eval = self.loss_op
        predict_eval = self.pi

        _, loss, loss_critic, loss_actor,loss_entropy, ProbActions = self.sess.run([train_eval, loss_eval, self.loss_critic_op, self.loss_actor_op, self.loss_entropy, predict_eval],
                                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: lr,
                                                        self.X1: xs,
                                                        self.actions: actions,
                                                        self.TD_target: TD_target,
                                                        self.prob_old: prob_old,
                                                        self.GAE: GAE})

        MaxProbActions = np.max(ProbActions)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss_Total', group=self.vehicle_name, value=LA.norm(loss) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Actor', group=self.vehicle_name, value=LA.norm(loss_actor) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Critic', group=self.vehicle_name, value=LA.norm(loss_critic) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Loss_Entropy', group=self.vehicle_name, value=LA.norm(loss_entropy) / batch_size,
                                index=self.iter_policy)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=self.iter_policy)
        self.log_to_tensorboard(tag='MaxProb', group=self.vehicle_name, value=MaxProbActions, index=self.iter_policy)

    def action_selection_with_prob(self, state):
        action = np.zeros(dtype=int, shape=[state.shape[0], 1])
        prob_action = np.zeros(dtype=float, shape=[state.shape[0], 1])

        probs = self.sess.run(self.pi,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.actions: action})

        for j in range(probs.shape[0]):
            action[j] = np.random.choice(self.num_actions, 1, p=probs[j])[0]
            prob_action[j] = probs[j][action[j][0]]

        return action.astype(int), prob_action

    def action_selection(self, state):
        action = np.zeros(dtype=int, shape=[state.shape[0], 1])
        prob_action = np.zeros(dtype=float, shape=[state.shape[0], 1])

        probs = self.sess.run(self.pi,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.actions: action})

        for j in range(probs.shape[0]):
            action[j] = np.random.choice(self.num_actions, 1, p=probs[j])[0]
            prob_action[j] = probs[j][action[j][0]]

        return action.astype(int)

    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)

    def save_network(self, save_path, episode=''):
        save_path = save_path + self.vehicle_name + '/' + self.vehicle_name + '_' + str(episode)
        self.saver.save(self.sess, save_path)
        print('Model Saved: ', save_path)

    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)

# MORE ADDITION FROM PEDRA MULTI

class C3F2_ActorCriticShared(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        # # weights_path = 'models/imagenet.npy'
        # weights_path = 'models/prune_weights.npy'
        # weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        # print('Loading pruned weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, k=7, out=96, s=4, p="VALID", trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, k=5, out=64, s=1, p="VALID", trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, k=3, out=64, s=1, p="SAME", trainable=train_conv)

        self.flat = tf.contrib.layers.flatten(self.conv3)

        self.fc1_values = self.FullyConnected(self.flat, units_in=1024, units_out=1024, act='relu', trainable=train_fc6)
        self.fc2_values = self.FullyConnected(self.fc1_values, units_in=1024, units_out=1, act='linear',
                                       trainable=train_fc7)

        self.fc1_actions = self.FullyConnected(self.flat, units_in=1024, units_out=1024, act='relu', trainable=train_fc6)
        self.fc2_actions = self.FullyConnected(self.fc1_actions, units_in=1024, units_out=num_actions, act='softmax',
                                       trainable=train_fc7)

        self.action_probs = self.fc2_actions
        self.state_value = self.fc2_values

    def conv(self, input, k, out, s, p, trainable=True):

        W = tf.Variable(tf.truncated_normal(shape=(k, k, int(input.shape[3]), out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[out], stddev=0.05), trainable=trainable)

        conv_kernel_1 = tf.nn.conv2d(input, W, [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05 / 10, seed=1), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05 / 10, seed=1), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W, b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        elif act == 'softmax':
            return tf.nn.softmax(tf.nn.xw_plus_b(input, W, b))
        else:
            assert (1 == 0)