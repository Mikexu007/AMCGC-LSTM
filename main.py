# 1,√graph 2,pattern 3.√batch size 4.attention(loss 2FC) 5. √*****LSTM preprocess 6. √****查一下gcn的层数和channels数
# 7. √strategy = 'distance' 8. √2层（32 64 ），√epoch 检测点 5,10,15,20 25    √9. kernel_size, stride_size
import  tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout,Input,Softmax,Reshape,Dot,Multiply,Layer

# from new_lstm_gcn import LSTM

# from new_lstm_multi_gcn import LSTM as  LSTM_multi_gcn # 先gcn后lstm矩阵相乘
from tensorflow.keras.layers import LSTM as LSTM_STAND

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.layers import  Dense, Activation
from graph import Graph
import numpy as np
import pandas as pd
import os


skeleton_dir = '/data5/xushihao/data/nturgbd_skeletons_s01_to_s032_extract_skeleton'

subject_train_set = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
                     45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
                     83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
subject_test_set = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32,
                    33, 36, 37, 39, 40, 41, 42, 43, 44, 48, 51, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                    71, 72, 73,75, 76, 77, 79, 87, 88, 90, 96, 99, 101, 102, 104, 105, 106]

setup_train_set = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
setup_test_set = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

# 固定的参数
max_body = 4
joints = 25
dim = 3
data_dim = max_body * joints * dim
MAX_FRAME = 300
#################
max_frames = 100
#################
num_classes = 120
no_subject_train_set = 63026
no_subject_test_set = 50922
no_setup_train_set = 54471
no_setup_test_set = 59477

# 需调参数
loss='categorical_crossentropy'
strategy = 'spatial'
max_hop = 1 # 关节点最大相接距离
dilation = 1


epochs = 100
batch_size = 32
test_batch_size = batch_size
# optimizer = Adam()
optimizer = Adam()

train_or_test = "train"
GCN_out_channels_0 = 64 # 第一层
mid_channels = GCN_out_channels_0
units = 128
units_lstm = 200 # spatial 、temporal、main network中 普通lstm的units

b1 = ""
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth=True
session = tensorflow.Session(config=config)
import tensorflow.keras.backend as ktf
ktf.set_session(session)
# sum_or_mean = "sum"
sum_or_mean = "mean"

spt = "_spt"
# spt = ""
mc = "_mc"
# mc = "_mc_64-128-128"
# temp = "_temp"
temp = ""

gcn_relu_flag = False
concat_part = 'all'
output_dim_list = [3, 64, 64]#第一个3 为 三维坐标
if mc == "_mc":
    from  mcgcLstm_resn_customLayer import LSTM as LSTM_single_gcn # 先gcn后lstm矩阵相乘
# if mc == "_mc_64-128-128":
#     from new_lstm_single_cf_concat_gcn_res3_ntu_customLayer import LSTM as LSTM_single_gcn

name = "_alpha_beta_{}{}{}_sig_{}_{}_pv{}_{}".format(spt, mc, temp, sum_or_mean, b1, max_frames,str(gcn_relu_flag))
round = 1
flag = "_subject"
if flag == "_setup":
    subject_or_setup = 1
if flag == "_subject":
    subject_or_setup = 0

loss_flag = 'SALossTrue'
# loss_flag = 'SALossTrue'
# loss_flag = 'TALossTrue'
# loss_flag = ''

way_for_beta ='Mean'
# way_for_beta ='Sum'

beta_flag = "_right_beta"

name = "{}_{}_{}_beta{}_pv{}_gcnRelu{}".format(spt, mc, temp, sum_or_mean,  max_frames, str(gcn_relu_flag))
round = "sameAsSt-gcn"
out_dir_name = '{}_{}_mcLstm{}_otherLstm{}_concatPart{}_{}_{}_{}_hop{}_round{}'.format(name,
                                                                        flag,
                                                                        units,
                                                                        units_lstm,
                                                                        concat_part,
                                                                        strategy,
                                                                        str(output_dim_list),
                                                                        loss_flag,
                                                                        max_hop,
                                                                        round)

weight_dir = "/data5/xushihao/weights_with_edgeImportance/"
# units = max_body *joints * mid_channels
digraph_or_undigraph = 'digraph'
match_A_kernel_size = max_hop*2 + 1


#得到选取的样本名
def get_samples(subject_or_setup = 0, train_or_test_index=None):
    skeleton_list = os.listdir(skeleton_dir)
    # print(len(skeleton_list))
    selected_data = []
    if subject_or_setup == 0:
        for name in skeleton_list:
            if int(name[9:12]) in train_or_test_index:
                selected_data.append(name)

    else:
        for name in skeleton_list:
            if int(name[1:4]) in train_or_test_index:
                selected_data.append(name)

    return selected_data


#得到相对位置数据
def skeleton_gen(subject_or_setup = 0,
                 train_or_test_index=None,
                 num_classes = None,
                 batch_size = None,
                 max_frames = None,
                 feat_dim = None):
    selected_data_names = get_samples(subject_or_setup, train_or_test_index)
    len_selected_data = len(selected_data_names)
    x = np.zeros((batch_size, max_frames, feat_dim))
    y = np.zeros((batch_size, num_classes))

    while True:
        indices = list(range(0,len_selected_data))
        np.random.shuffle(indices)
        batch_count = 0
        # rest_samples = len_selected_data
        for index in indices:
            skeleton_raw = np.load(os.path.join(skeleton_dir,selected_data_names[index]))
            skeleton_raw = skeleton_raw.reshape(MAX_FRAME, max_body, joints, dim) # T,M,V,C
            skeleton = skeleton_raw[:max_frames,:,:,:]
            skeleton = skeleton.transpose((3, 0, 2, 1))  # C,T,V,M
            C, T, V, M = skeleton.shape
            x_new = np.zeros((C, T, V, M))
            # x_new[:C, :, :, :] = x
            for i in range(V):
                x_new[:C, :, i, :] = skeleton[:, :, i, :] - skeleton[:, :, 1, :]

            x_new = x_new.transpose((1,3,2,0)) #T,M,V,C
            x_new = x_new.reshape((T,-1))

            label = int(selected_data_names[index][17:20])
            one_hot_label = tensorflow.keras.utils.to_categorical(label - 1, num_classes=num_classes)

            x[batch_count] = x_new
            y[batch_count] = one_hot_label

            del skeleton, x_new

            batch_count += 1

            if batch_count == batch_size:
                ret_x = x
                ret_y = y
                x = np.zeros((batch_size, max_frames, feat_dim))
                y = np.zeros((batch_size, num_classes))
                # rest_samples -= batch_size
                batch_count = 0
                yield (ret_x, ret_y)

            # if rest_samples < batch_size:
            #     if batch_count == len_selected_data % batch_size:
            #         yield (x, y)


class Delay_layer(Layer):
    def __init__(self, **kwargs):
        super(Delay_layer, self).__init__(**kwargs)
        # self.new_input_shape = []

    def build(self, input_shape):
        super(Delay_layer,self).build(input_shape)
        # super(Reshap_input_layer, self).build(input_shape)
        # self.kernel = self.add_variable("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                        self.num_outputs])
    def call(self, inputs):
        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        print("11111111111111111111111")
        print(inputs.get_shape().as_list())
        n, t, tot_dim = inputs.get_shape().as_list()
        add_zeros = tf.zeros_like(inputs[:,0,:])
        add_zeros = tf.expand_dims(add_zeros,axis=1)
        inputs = tf.concat([add_zeros,inputs],axis=1)
        return inputs[:,0:t,:]
    def compute_output_shape(self, input_shape):
        print("000000000000000000000000000")
        print(input_shape)
        return (input_shape[0],input_shape[1],input_shape[2])

class Reduce_sum(Layer):
    def __init__(self, **kwargs):
        super(Reduce_sum, self).__init__(**kwargs)
        # self.new_input_shape = []

    def build(self, input_shape):
        super(Reduce_sum,self).build(input_shape)
        # super(Reshap_input_layer, self).build(input_shape)
        # self.kernel = self.add_variable("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                        self.num_outputs])
    def call(self, inputs):
        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        inputs = tf.reduce_sum(inputs,1)
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

class Reduce_mean(Layer):
    def __init__(self, **kwargs):
        super(Reduce_mean, self).__init__(**kwargs)
        # self.new_input_shape = []

    def build(self, input_shape):
        super(Reduce_mean,self).build(input_shape)
        # super(Reshap_input_layer, self).build(input_shape)
        # self.kernel = self.add_variable("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                        self.num_outputs])
    def call(self, inputs):
        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        inputs = tf.reduce_mean(inputs,1)
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

def main():
    graph = Graph(strategy = strategy,
                  max_hop = max_hop,
                  dilation = dilation,
                  digraph_or_undigraph = digraph_or_undigraph)

    A = tensorflow.convert_to_tensor(graph.A, dtype=tensorflow.float32)

    main_input = Input(batch_shape=(batch_size,max_frames, data_dim), dtype=tf.float32, name='main_input')
    reshape_input = Reshape([ max_frames, max_body, joints, dim])(main_input)
    ## spatial attention
    if spt == "_spt":
        spatial_lstm_data = LSTM_STAND(units_lstm,return_sequences=True, name='spatial_lstm')(main_input)
        data_delay = Delay_layer(name='spatial_delay')(spatial_lstm_data) # n,t,units_lstm
        # data_reshape = Reshap_layer(name='reshape_layer')(data_delay)
        delay_data_reshape = Reshape([max_frames,max_body,joints,units_lstm//joints//max_body])(data_delay)

        data_concat = tf.keras.layers.concatenate([reshape_input, delay_data_reshape],axis=-1)
        data_fc = Dense(units=256,activation="tanh",name="spatial_dense1")(data_concat) # n,t,4,25,256
        data_fcc = Dense(units=1,name="spatial_dense2")(data_fc)# n,t,4,25,1
        alpha = Softmax(axis=3,name="spatial_softmax")(data_fcc)
        alpha = tf.keras.layers.concatenate([alpha, alpha, alpha], axis=-1)

    ## temporal attention
    # temporal_input = Input(shape=(max_frames, data_dim), dtype=tf.float32, name='temporal_input')
    if temp == "_temp":
        data_lstm1 = LSTM_STAND(units_lstm,return_sequences=True, name='temporal_lstm')(main_input)
        data_delay1 = Delay_layer(name='temporal_delay')(data_lstm1) # n,t,units_lstm
        # data_reshape1 = Reshape([batch_size, max_frames, max_body, -1], name='reshape_layer1')(data_delay1)
        data_concat1 = tf.keras.layers.concatenate([main_input, data_delay1], axis=-1)
        # print("corrct beta")
        if b1 == "b1":
            beta = Dense(units=units_lstm, activation="sigmoid", name="temporal_dense")(data_concat1)  # n,t,256
        else:
            if beta_flag == "_right_beta":
                print("right_beta")
                beta = Dense(units=1, activation="sigmoid", name="temporal_dense")(data_concat1)
            else:
                beta = Dense(units=num_classes, activation="sigmoid", name="temporal_dense")(data_concat1)  # n,t,256
    '''beta的维度要和main lstm 最后对应'''

    ##maint lstm
    if spt == "_spt":
        main_input_ = Multiply()([reshape_input, alpha])
    else:
        main_input_ = reshape_input
    main_input_ = Reshape([max_frames,-1])(main_input_)
    data_lstm_gcn = LSTM_single_gcn( units = units,
                                    A = A,
                                    return_sequences=True,
                                    stateful=False,
                                    match_A_kernel_size = match_A_kernel_size,##在Graph 里面是对max_hop 进行了+1 操作，这里也+1
                                    batch_size=batch_size,
                                    joints = joints,
                                    max_body = max_body,
                                    name = 'lstm_gcn',
                                    gcn_relu_flag = gcn_relu_flag,
                                    dim = dim,
                                    concat_part = concat_part,
                                     output_dim_list=output_dim_list)(main_input_)
    data_lstm = LSTM_STAND(units=units_lstm,return_sequences=True,name="lstm_after_gcn_lstm")(data_lstm_gcn)

    if b1 == 'b1':
        if temp == "_temp":
            data_z = Multiply()([beta, data_lstm])
        else:
            data_z = data_lstm
        if sum_or_mean == "sum":
            data_z_ = Reduce_sum(name="main_reduce_sum")(data_z)  ## n,t,120
        if sum_or_mean == "mean":
            data_z_ = Reduce_mean(name="main_reduce_mean")(data_z)  ## n,t,120
        data_fc_lstm = Dense(units=120, name="main_dense")(data_z_)
        outputs = Softmax(name='main_softmax')(data_fc_lstm)
    else:
        data_fc_lstm = Dense(units=120, name="main_dense")(data_lstm)
        if temp == "_temp":
            data_z = Multiply()([beta, data_fc_lstm])
        else:
            data_z = data_fc_lstm
        if sum_or_mean == "sum":
            data_z_ = Reduce_sum(name="main_reduce_sum")(data_z)  ## n,t,120
        if sum_or_mean == "mean":
            data_z_ = Reduce_mean(name="main_reduce_mean")(data_z)  ## n,t,120
        outputs = Softmax(name='main_softmax')(data_z_)

    model = Model(inputs=main_input, outputs=outputs)
    model.summary()

    if "Loss" in loss_flag:

        # attention loss
        lambda1 = 0.01
        lambda2 = 0.001

        alpha_loss = alpha[:, :, :, :, 0]  # n,t,4,25,1
        alpha_loss = tf.reduce_mean(alpha_loss, 1)  # n,4,25,1
        alpha_loss = tf.square(1 - alpha_loss)
        alpha_loss = lambda1 * tf.reduce_sum(tf.reduce_sum(alpha_loss, 1), 1)

        if temp == "_temp":
            if way_for_beta == 'Mean':
                beta = tf.reduce_mean(beta, axis=-1)
            if way_for_beta == "Sum":
                beta = tf.reduce_sum(beta,axis=-1)
            beta_loss = tf.square(beta)  # !!!!!
            beta_loss = lambda2 * tf.reduce_mean(beta_loss, 1)


        def my_loss(y_true, y_pred):
            if loss_flag == "STALossTrue":
                print("\n")
                print("fuck================")
                print("\n")
                result = K.categorical_crossentropy(y_true, y_pred, from_logits=False) + alpha_loss + beta_loss
            if loss_flag == 'SALossTrue':
                print('SALossTrue')
                result = K.categorical_crossentropy(y_true, y_pred, from_logits=False) + alpha_loss
            if loss_flag == 'TALossTrue':
                result = K.categorical_crossentropy(y_true, y_pred, from_logits=False)  + beta_loss
            return result

        model.compile(loss=my_loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        print("\n")
        print("shit================")
        print("\n")

    else:
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

    if not os.path.exists( weight_dir + out_dir_name + '_val_acc'):
        os.makedirs( weight_dir + out_dir_name + '_val_acc')
    weight_path1 = weight_dir + out_dir_name + '_val_acc' + '/{epoch:03d}_{val_acc:0.3f}.h5'

    checkpoint_fit_g = ModelCheckpoint(weight_path1,
                                       monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=3,
                                  verbose=1,
                                  mode='auto',
                                  cooldown=0,
                                  min_lr=0.00001)

    callbacks_fit_g = [checkpoint_fit_g, reduce_lr]

    if subject_or_setup == 0:
        train_gen = skeleton_gen(subject_or_setup = subject_or_setup,
                                         train_or_test_index=subject_train_set,
                                         num_classes = num_classes,
                                         batch_size = batch_size,
                                         max_frames = max_frames,
                                         feat_dim = data_dim)

        valid_gen = skeleton_gen(subject_or_setup = subject_or_setup,
                                 train_or_test_index=subject_test_set,
                                 num_classes = num_classes,
                             batch_size = test_batch_size,
                             max_frames = max_frames,
                             feat_dim = data_dim)
    else:
        train_gen = skeleton_gen(subject_or_setup = subject_or_setup,
                                         train_or_test_index=setup_train_set,
                                         num_classes = num_classes,
                                         batch_size = batch_size,
                                         max_frames = max_frames,
                                         feat_dim = data_dim)

        valid_gen = skeleton_gen(subject_or_setup = subject_or_setup,
                                 train_or_test_index=setup_test_set,
                                 num_classes = num_classes,
                             batch_size = test_batch_size,
                             max_frames = max_frames,
                             feat_dim = data_dim)

    if train_or_test == "train":
        if subject_or_setup == 0:
            model.fit_generator(train_gen,
                                steps_per_epoch= no_subject_train_set / batch_size + 1,
                                validation_data = valid_gen,
                                validation_steps = no_subject_test_set / test_batch_size + 1,
                                callbacks=callbacks_fit_g,
                                epochs=epochs,
                                verbose=1,
                                workers=1,
                                initial_epoch=0
                                )
        else:
            model.fit_generator(train_gen,
                                steps_per_epoch=no_setup_train_set / batch_size + 1,
                                validation_data=valid_gen,
                                validation_steps=no_setup_test_set / test_batch_size + 1,
                                callbacks=callbacks_fit_g,
                                epochs=epochs,
                                verbose=1,
                                workers=1,
                                initial_epoch=0
                                )



    if train_or_test == "re_train":
        # from per_class_acc import load_model_and_predict

        target_dir = weight_dir + out_dir_name + '_val_acc' + "/014_0.679.h5"
        model.load_weights(target_dir)
        # load_model_and_predict(model=model,
        #                        save_figure_name=out_dir_name,
        #                        batch_size=batch_size,
        #                        subject_or_setup=subject_or_setup,
        #                        subject_test_set=subject_test_set,
        #                        test_samples=no_subject_test_set
        #                        )
        model.fit_generator(train_gen,
                            steps_per_epoch=no_subject_train_set / batch_size + 1,
                            validation_data=valid_gen,
                            validation_steps=no_subject_test_set / test_batch_size + 1,
                            callbacks=callbacks_fit_g,
                            epochs=epochs,
                            verbose=1,
                            workers=1,
                            initial_epoch=14
                            )


    if train_or_test == "test":
        from per_class_acc import load_model_and_predict
        target_dir = weight_dir + out_dir_name + '_val_acc' + "/035_0.545.hdf5"
        model = tf.keras.models.load_model(target_dir)
        load_model_and_predict(model = model,
                               save_figure_name = out_dir_name,
                               batch_size = batch_size,
                               subject_or_setup = subject_or_setup,
                               subject_test_set = subject_test_set,
                               test_samples = no_subject_test_set
                               )
    # prediction=model.predict_generator(valid_gen,verbose=1)
    # predict_label = np.argmax(prediction, axis=1)
    # true_label = valid_gen.classes
    # pd.crosstab(true_label, predict_label, rownames=['label'], colnames=['predict'])


if __name__ == "__main__":
    main()