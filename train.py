import numpy as np
import csv

import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical

from encoder import Encoder, StateSpace
from manager import NetworkManager
from model import model_fn
import random
import time
from utils import Logger
import sys

# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

B = 5  # number of blocks in each cell
K_ = 25  # number of children networks to train

MAX_EPOCHS = 1  # maximum number of epochs to train
BATCHSIZE = 128  # batchsize
REGULARIZATION = 0  # regularization strength
CONTROLLER_CELLS = 100  # number of cells in RNN controller
RNN_TRAINING_EPOCHS = 10
RESTORE_CONTROLLER = True  # restore controller to continue training
NORMAL_CELL_NUMBER= 3
FIRST_LAYER_FILTERs= 48
LOG_FILE='log.txt'
sys.stdout=Logger(LOG_FILE)


operators = ['3x3 dconv', '5x5 dconv', '7x7 dconv',
             '1x7-7x1 conv', '3x3 maxpool', '3x3 avgpool','identity','3x3 conv']  # use the default set of operators, minus identity and conv 3x3


# construct a state space
state_space = StateSpace(B, input_lookback_depth=-1, input_lookforward_depth=4,
                         operators=operators)




# print the state space being searched
state_space.print_state_space()
NUM_TRAILS = state_space.print_total_models(K_)

# prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager


#start recording time
start_time=time.time()


with policy_sess.as_default():
    # create the Encoder and build the internal policy network
    controller = Encoder(policy_sess, state_space, B=B, K=K_,
                         train_iterations=RNN_TRAINING_EPOCHS,
                         reg_param=REGULARIZATION,
                         controller_cells=CONTROLLER_CELLS,
                         restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(dataset, epochs=MAX_EPOCHS, batchsize=BATCHSIZE,cell_number=NORMAL_CELL_NUMBER,filters=FIRST_LAYER_FILTERs)
print()

# train for number of trails
for trial in range(B):
    with policy_sess.as_default():
        K.set_session(policy_sess)

        if trial == 0:
            k = None
        else:
            k = K_

        actions ,input_ids= controller.get_actions(top_k=k)  # get all actions for the previous state

    rewards = []
    model_ids=[]
    for t, action in enumerate(actions):
        # print the action probabilities
        state_space.print_actions(action)
        print("Model #%d / #%d" % (t + 1, len(actions)))
        print("Predicted actions : ", state_space.parse_state_space_list(action))

        # build a model, train and get reward and accuracy from the network manager
        if input_ids==None:
            input_id=None
        else :
            input_id=input_ids[t]
        reward,model_id= manager.get_rewards(model_fn, state_space.parse_state_space_list(action),input_id)
        print("Final Accuracy : ", reward)

        rewards.append(reward)
        model_ids.append(model_id)
        print("\nFinished %d out of %d models ! \n" % (t + 1, len(actions)))

        # write the results of this trial into a file
        with open('test.csv', mode='a+', newline='') as f:
            data = [reward,model_id]
            data.extend(state_space.parse_state_space_list(action))
            writer = csv.writer(f)
            writer.writerow(data)

    with policy_sess.as_default():
        K.set_session(policy_sess)
        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step(rewards,model_ids)
        print("Trial %d: Encoder loss : %0.6f" % (trial + 1, loss))

        controller.update_step()
        print()

#record endding time 
end_time=time.time()
print('Total Time : ',end_time-start_time ,' sec')
print("Finished !")
