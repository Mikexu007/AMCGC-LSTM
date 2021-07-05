

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Layer, RNN, InputSpec
import tensorflow.keras.activations as activations
import tensorflow.keras.initializers as  initializers
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.constraints as constraints
import tensorflow.keras.backend as K
import os
import numpy as np
import pysnooper




class LSTMCell(Layer):
    """Cell class for the LSTM layer.
    """

    # # @pysnooper.snoop()
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 # kernel_initializer='glorot_uniform',
                 kernel_initializer='he_normal',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 A=None,
                 match_A_kernel_size=None,
                 t_kernel_size=1,
                 t_stride=1,
                 t_dilation=1,
                 bias=True,
                 joints = None,
                 dim = None,
                 max_body = None,
                 batch_size=None,
                 gcn_relu_flag = False,
                 output_dim_list = None,
                 concat_part = None,
                 **kwargs):
        super(LSTMCell, self).__init__(**kwargs)
        self.output_dim_list = output_dim_list
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.activation_relu = activations.get("relu")
        self.gcn_relu_flag = gcn_relu_flag

        self.A = A
        # self.mask = np.ones_like(self.A)
        # self.mask[self.A == 0] = 0
        # self.m_dim_0, self.m_dim_1, self.m_dim_2 = self.mask.shape
        # self.mask = tf.convert_to_tensor(self.mask, dtype=tf.float32)
        # self.adj = tf.convert_to_tensor(self.A, dtype=tf.float32)
        self.concat_part = concat_part
        self.match_A_kernel_size = match_A_kernel_size

        self.max_body = max_body
        self.joints = joints
        self.dim = dim
        self.dim1 = joints * self.dim
        self.dim2 = max_body * joints * self.dim
        self.batch_size = batch_size
        self.one = 1

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_dim = self.max_body * self.joints * self.units + self.max_body*self.units + self.units
        self.state_size = (self.state_dim, self.state_dim)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

        ####建立4个卷积核用来做4个门的gcn卷积

    # # @pysnooper.snoop()
    def build(self, input_shape):
        print("buid.....")
        stride = 1
        print(self.output_dim_list)
        names = self.__dict__
        if self.concat_part != 'att':
            size_A = K.int_shape(self.A)
            names['edge_importance'] = self.add_weight(shape= (len(self.output_dim_list) - 1,size_A[0],size_A[1],size_A[2]),
                                                       initializer = tf.keras.initializers.Ones(),
                                                       name = 'edge_importance')
            for index, output_dim in enumerate(self.output_dim_list) :
                if index == len(self.output_dim_list) - 1:
                    break
                names['kernel_shape_res' + str(index)] = (stride, stride) + (output_dim, self.output_dim_list[index+1]) #
                names['kernel_shape_gcn' + str(index)] = (stride, stride) + (output_dim, self.output_dim_list[index+1] * self.match_A_kernel_size) #

                names['conv_filters_res' + str(index)] = self.output_dim_list[index+1]
                names['conv_filters_gcn' + str(index)] = self.output_dim_list[index+1]* self.match_A_kernel_size

                if index > 0:
                    if output_dim != self.output_dim_list[index+1]:
                        names['new_conv_kernel_res' + str(index)] = self.add_weight(shape=names['kernel_shape_res' + str(index)],
                                                             initializer=self.kernel_initializer,
                                                             name='new_conv_kernel_res'+str(index),
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint)


                names['new_conv_kernel_gcn'+ str(index)] = self.add_weight(shape=names['kernel_shape_gcn' + str(index)],
                                                         initializer=self.kernel_initializer,
                                                         name='new_conv_kernel_gcn'+ str(index),
                                                         regularizer=self.kernel_regularizer,
                                                         constraint=self.kernel_constraint)

                if self.use_bias:
                    if index > 0:
                        if output_dim != self.output_dim_list[index + 1]:
                            names['new_bias_res' + str(index)] = self.add_weight(shape=(names['conv_filters_res' + str(index)],),
                                                                  initializer=self.bias_initializer,
                                                                  name='new_bias_res' + str(index),
                                                                  regularizer=self.bias_regularizer,
                                                                  constraint=self.bias_constraint)

                    names['new_bias_gcn' + str(index)] = self.add_weight(shape=(names['conv_filters_gcn' + str(index)],),
                                                          initializer=self.bias_initializer,
                                                          name='new_bias_gcn' + str(index),
                                                          regularizer=self.bias_regularizer,
                                                          constraint=self.bias_constraint)


        # print(names['new_conv_kernel_res0'])
        # ==============================================
        # input_dim = input_shape[-1]
        # input_dim = self.max_body * self.joints * self.GCN_out_channels
        # input_dim = self.GCN_out_channels
        if self.concat_part == 'gcn':
            input_dim =   self.output_dim_list[-1]

        if self.concat_part == 'att':
            input_dim =  self.output_dim_list[0]

        if self.concat_part == 'all':
            input_dim = self.output_dim_list[0] + self.output_dim_list[-1]

        input_dim1 = self.joints * input_dim
        input_dim2 = self.max_body * self.joints * input_dim

        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_body = self.add_weight(shape=(input_dim1, self.units * 4),
                                      name='kernel_body',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_whole = self.add_weight(shape=(input_dim2, self.units * 4),
                                      name='kernel_whole',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # self.recurrent_kernel = self.add_weight(
        #     shape=(self.units, self.units * 4),
        #     name='recurrent_kernel',
        #     initializer=self.recurrent_initializer,
        #     regularizer=self.recurrent_regularizer,
        #     constraint=self.recurrent_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.recurrent_kernel_body = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel_body',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.recurrent_kernel_whole = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel_whole',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)



        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.max_body, self.joints, self.units,), *args, **kwargs),
                        initializers.Ones()((self.max_body, self.joints, self.units,), *args, **kwargs),
                        self.bias_initializer((self.max_body, self.joints, self.units * 2,), *args, **kwargs),
                    ])
                def bias_initializer1(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.max_body, self.units,), *args, **kwargs),
                        initializers.Ones()((self.max_body, self.units,), *args, **kwargs),
                        self.bias_initializer((self.max_body, self.units * 2,), *args, **kwargs),
                    ])
                def bias_initializer2(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.one, self.units,), *args, **kwargs),
                        initializers.Ones()((self.one, self.units,), *args, **kwargs),
                        self.bias_initializer((self.one, self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
                bias_initializer1 = self.bias_initializer
                bias_initializer2 = self.bias_initializer

            self.bias = self.add_weight(shape=(self.max_body, self.joints, self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias_body = self.add_weight(shape=(self.max_body, self.units * 4,),
                                        name='bias_body',
                                        initializer=bias_initializer1,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias_whole = self.add_weight(shape=(self.one, self.units * 4,),
                                        name='bias_whole',
                                        initializer=bias_initializer2,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)


        else:
            self.bias = None
            self.bias_body = None
            self.bias_whole = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.kernel_i_body = self.kernel_body[:, :self.units]
        self.kernel_f_body = self.kernel_body[:, self.units: self.units * 2]
        self.kernel_c_body = self.kernel_body[:, self.units * 2: self.units * 3]
        self.kernel_o_body = self.kernel_body[:, self.units * 3:]

        self.kernel_i_whole = self.kernel_whole[:, :self.units]
        self.kernel_f_whole = self.kernel_whole[:, self.units: self.units * 2]
        self.kernel_c_whole = self.kernel_whole[:, self.units * 2: self.units * 3]
        self.kernel_o_whole = self.kernel_whole[:, self.units * 3:]

        ########

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        self.recurrent_kernel_i_body = self.recurrent_kernel_body[:, :self.units]
        self.recurrent_kernel_f_body = self.recurrent_kernel_body[:, self.units: self.units * 2]
        self.recurrent_kernel_c_body = self.recurrent_kernel_body[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o_body = self.recurrent_kernel_body[:, self.units * 3:]

        self.recurrent_kernel_i_whole = self.recurrent_kernel_whole[:, :self.units]
        self.recurrent_kernel_f_whole = self.recurrent_kernel_whole[:, self.units: self.units * 2]
        self.recurrent_kernel_c_whole = self.recurrent_kernel_whole[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o_whole = self.recurrent_kernel_whole[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:,:,:self.units]
            self.bias_f = self.bias[:,:,self.units: self.units * 2]
            self.bias_c = self.bias[:,:,self.units * 2: self.units * 3]
            self.bias_o = self.bias[:,:,self.units * 3:]

            self.bias_i_body = self.bias_body[:,:self.units]
            self.bias_f_body = self.bias_body[:,self.units: self.units * 2]
            self.bias_c_body = self.bias_body[:,self.units * 2: self.units * 3]
            self.bias_o_body = self.bias_body[:,self.units * 3:]

            self.bias_i_whole = self.bias_whole[:,:self.units]
            self.bias_f_whole = self.bias_whole[:,self.units: self.units * 2]
            self.bias_c_whole = self.bias_whole[:,self.units * 2: self.units * 3]
            self.bias_o_whole = self.bias_whole[:,self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None

            self.bias_i_body = None
            self.bias_f_body = None
            self.bias_c_body = None
            self.bias_o_body = None

            self.bias_i_whole = None
            self.bias_f_whole = None
            self.bias_c_whole = None
            self.bias_o_whole = None


        self.built = True

    # @pysnooper.snoop()
    def call(self, inputs, states, training=None):
        names_call = self.__dict__
        print("here",inputs.get_shape().as_list())
        inputs = K.reshape(inputs, [self.batch_size, self.max_body, self.joints, self.dim])
        # new_input = K.reshape(inputs, [self.batch_size, self.max_body, self.joints, self.dim])
        new_input = inputs

        if self.concat_part != 'att':
            # for i in range(len(self.output_dim_list) - 1):
            for i, output_dim in enumerate(self.output_dim_list):
                if i == len(self.output_dim_list) - 1:
                    break
                if i > 0:
                    if output_dim == self.output_dim_list[i + 1]:
                        inputs_res = new_input
                    else:
                        inputs_res = K.conv2d(new_input,
                                              names_call['new_conv_kernel_res'+ str(i)],
                                              strides=(1, 1),
                                              dilation_rate=(1, 1),
                                              padding="same")
                        if self.use_bias:
                            inputs_res = K.bias_add(
                                inputs_res,
                                names_call['new_bias_res' + str(i)],
                            )
                        inputs_res = tf.layers.batch_normalization(inputs_res)

                inputs_gcn = K.conv2d(new_input,
                               names_call['new_conv_kernel_gcn'+ str(i)],
                               strides=(1, 1),
                               dilation_rate=(1, 1),
                               padding="same")
                if self.use_bias:
                    inputs_gcn = K.bias_add(
                        inputs_gcn,
                        names_call['new_bias_gcn'+ str(i)]
                    )
                n, m, v, kc = self.batch_size, self.max_body, self.joints, self.output_dim_list[i+1] * self.match_A_kernel_size
                inputs_gcn = K.reshape(inputs_gcn, [n, m, v, self.match_A_kernel_size, self.output_dim_list[i+1]])  # m,v,k,c
                inputs_gcn = tensorflow.transpose(inputs_gcn, perm=[0, 3, 4, 1, 2])  # n,k,c,m,v
                print("edge_imporatnce")
                temp_A = tf.multiply(self.A, names_call['edge_importance'][i])
                inputs_gcn = tensorflow.einsum('nkcmv,kvw->ncmw', inputs_gcn, temp_A)
                inputs_gcn = tensorflow.transpose(inputs_gcn, perm=[0, 2, 3, 1])  ##n,m,w,c, 这个c是out_dim1

                if i == 0:
                    inputs_res_gcn = inputs_gcn
                else:
                    inputs_res_gcn = inputs_res + inputs_gcn

                if self.gcn_relu_flag == True:
                    inputs_res_gcn = self.activation_relu(inputs_res_gcn)

                new_input = inputs_res_gcn


        inputs_res_gcn_final = new_input

        if self.concat_part == 'gcn':
            inputs = inputs_res_gcn_final

        if self.concat_part == 'att':
            inputs = inputs
        if self.concat_part == 'all':
            inputs = K.concatenate([inputs,inputs_res_gcn_final], axis=-1)# n,4,25,gcn+3

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state


        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            ''''''
            if self.concat_part == "gcn":
                new_dim =  self.output_dim_list[-1]
            if self.concat_part == "att":
                new_dim = self.dim
            if self.concat_part == "all":
                new_dim = self.dim + self.output_dim_list[-1]
            x_i = K.reshape(inputs_i, [self.batch_size, self.max_body, self.joints, new_dim])
            x_f = K.reshape(inputs_f, [self.batch_size, self.max_body, self.joints, new_dim])
            x_c = K.reshape(inputs_c, [self.batch_size, self.max_body, self.joints, new_dim])
            x_o = K.reshape(inputs_o, [self.batch_size, self.max_body, self.joints, new_dim])

            x_i_body = K.reshape(inputs_i, [self.batch_size, self.max_body, self.joints * new_dim])
            x_f_body = K.reshape(inputs_f, [self.batch_size, self.max_body, self.joints * new_dim])
            x_c_body = K.reshape(inputs_c, [self.batch_size, self.max_body, self.joints * new_dim])
            x_o_body = K.reshape(inputs_o, [self.batch_size, self.max_body, self.joints * new_dim])

            x_i_whole = K.reshape(inputs_i, [self.batch_size, self.one, self.max_body * self.joints * new_dim])
            x_f_whole = K.reshape(inputs_f, [self.batch_size, self.one, self.max_body * self.joints * new_dim])
            x_c_whole = K.reshape(inputs_c, [self.batch_size, self.one, self.max_body * self.joints * new_dim])
            x_o_whole = K.reshape(inputs_o, [self.batch_size, self.one, self.max_body * self.joints * new_dim])

            ''''''

            x_i = K.dot(x_i, self.kernel_i)# (4,25,units)
            x_f = K.dot(x_f, self.kernel_f)
            x_c = K.dot(x_c, self.kernel_c)
            x_o = K.dot(x_o, self.kernel_o)

            x_i_body = K.dot(x_i_body, self.kernel_i_body)# (4,25,units)
            x_f_body = K.dot(x_f_body, self.kernel_f_body)
            x_c_body = K.dot(x_c_body, self.kernel_c_body)
            x_o_body = K.dot(x_o_body, self.kernel_o_body)

            x_i_whole = K.dot(x_i_whole, self.kernel_i_whole)# (4,25,units)
            x_f_whole = K.dot(x_f_whole, self.kernel_f_whole)
            x_c_whole = K.dot(x_c_whole, self.kernel_c_whole)
            x_o_whole = K.dot(x_o_whole, self.kernel_o_whole)

            if self.use_bias:
                ###todo
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

                x_i_body = K.bias_add(x_i_body, self.bias_i_body)
                x_f_body = K.bias_add(x_f_body, self.bias_f_body)
                x_c_body = K.bias_add(x_c_body, self.bias_c_body)
                x_o_body = K.bias_add(x_o_body, self.bias_o_body)

                x_i_whole = K.bias_add(x_i_whole, self.bias_i_whole)
                x_f_whole = K.bias_add(x_f_whole, self.bias_f_whole)
                x_c_whole = K.bias_add(x_c_whole, self.bias_c_whole)
                x_o_whole = K.bias_add(x_o_whole, self.bias_o_whole)



            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                data_dim = self.max_body * self.joints * self.units

                h_tm1_i = h_tm1[:, 0:data_dim]
                h_tm1_f = h_tm1[:, 0:data_dim]
                h_tm1_c = h_tm1[:, 0:data_dim]
                h_tm1_o = h_tm1[:, 0:data_dim]

                h_tm1_i_body = h_tm1[:, data_dim : data_dim + self.max_body * self.units]
                h_tm1_f_body = h_tm1[:, data_dim : data_dim + self.max_body * self.units]
                h_tm1_c_body = h_tm1[:, data_dim : data_dim + self.max_body * self.units]
                h_tm1_o_body = h_tm1[:, data_dim : data_dim + self.max_body * self.units]

                h_tm1_i_whole = h_tm1[:, data_dim + self.max_body * self.units: ]
                h_tm1_f_whole = h_tm1[:, data_dim + self.max_body * self.units: ]
                h_tm1_c_whole = h_tm1[:, data_dim + self.max_body * self.units: ]
                h_tm1_o_whole = h_tm1[:, data_dim + self.max_body * self.units: ]



            ''''''
            temp_i = K.reshape(h_tm1_i, [self.batch_size, self.max_body, self.joints, self.units])
            temp_f = K.reshape(h_tm1_f, [self.batch_size, self.max_body, self.joints, self.units])
            temp_c = K.reshape(h_tm1_c, [self.batch_size, self.max_body, self.joints, self.units])
            temp_o = K.reshape(h_tm1_o, [self.batch_size, self.max_body, self.joints, self.units])

            temp_i_body = K.reshape(h_tm1_i_body, [self.batch_size, self.max_body, self.units])
            temp_f_body = K.reshape(h_tm1_f_body, [self.batch_size, self.max_body, self.units])
            temp_c_body = K.reshape(h_tm1_c_body, [self.batch_size, self.max_body, self.units])
            temp_o_body = K.reshape(h_tm1_o_body, [self.batch_size, self.max_body, self.units])

            temp_i_whole = K.reshape(h_tm1_i_whole, [self.batch_size, self.one, self.units])
            temp_f_whole = K.reshape(h_tm1_f_whole, [self.batch_size, self.one, self.units])
            temp_c_whole = K.reshape(h_tm1_c_whole, [self.batch_size, self.one, self.units])
            temp_o_whole = K.reshape(h_tm1_o_whole, [self.batch_size, self.one, self.units])

            ''''''

            temp_i = K.dot(temp_i, self.recurrent_kernel_i)
            temp_f = K.dot(temp_f, self.recurrent_kernel_f)
            temp_c = K.dot(temp_c, self.recurrent_kernel_c)
            temp_o = K.dot(temp_o, self.recurrent_kernel_o)

            temp_i_body = K.dot(temp_i_body, self.recurrent_kernel_i_body)
            temp_f_body = K.dot(temp_f_body, self.recurrent_kernel_f_body)
            temp_c_body = K.dot(temp_c_body, self.recurrent_kernel_c_body)
            temp_o_body = K.dot(temp_o_body, self.recurrent_kernel_o_body)

            temp_i_whole = K.dot(temp_i_whole, self.recurrent_kernel_i_whole)
            temp_f_whole = K.dot(temp_f_whole, self.recurrent_kernel_f_whole)
            temp_c_whole = K.dot(temp_c_whole, self.recurrent_kernel_c_whole)
            temp_o_whole = K.dot(temp_o_whole, self.recurrent_kernel_o_whole)

            data_dim = self.max_body * self.joints * self.units
            c_tm1_ = c_tm1[:, 0: data_dim]
            c_tm1_body = c_tm1[:, data_dim : data_dim + self.max_body * self.units]
            c_tm1_whole = c_tm1[:, data_dim + self.max_body * self.units:]


            i = self.recurrent_activation(x_i + temp_i)
            f = self.recurrent_activation(x_f + temp_f)
            c_tm1_ = K.reshape(c_tm1_,[self.batch_size, self.max_body, self.joints, self.units])
            c = f * c_tm1_ + i * self.activation(x_c + temp_c)
            o = self.recurrent_activation(x_o + temp_o)

            i_body = self.recurrent_activation(x_i_body + temp_i_body)
            f_body = self.recurrent_activation(x_f_body + temp_f_body)
            c_tm1_body = K.reshape(c_tm1_body,[self.batch_size, self.max_body,self.units])
            c_body = f_body * c_tm1_body + i_body * self.activation(x_c_body + temp_c_body)
            o_body = self.recurrent_activation(x_o_body + temp_o_body)

            i_whole = self.recurrent_activation(x_i_whole + temp_i_whole)
            f_whole = self.recurrent_activation(x_f_whole + temp_f_whole)
            c_tm1_whole = K.reshape(c_tm1_whole,[self.batch_size, self.one, self.units])
            c_whole = f_whole * c_tm1_whole + i_whole * self.activation(x_c_whole + temp_c_whole)
            o_whole = self.recurrent_activation(x_o_whole + temp_o_whole)

            ''''''

        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)#(4,25,units)
        h_body = o_body * self.activation(c_body)#(4,25,units)
        h_whole = o_whole * self.activation(c_whole)#(4,25,units)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        h = K.reshape(h, [self.batch_size, -1])
        h_body = K.reshape(h_body, [self.batch_size, -1])
        h_whole = K.reshape(h_whole, [self.batch_size, -1])

        c = K.reshape(c, [self.batch_size, -1])
        c_body = K.reshape(c_body, [self.batch_size, -1])
        c_whole = K.reshape(c_whole, [self.batch_size, -1])

        print("correct c")
        h = tf.concat([h, h_body, h_whole], axis= -1)
        c = tf.concat([c, c_body, c_whole], axis= -1)
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(LSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def gcn_conv(self, x_i, conv_kernel):
    #     max_body = 4
    #     joints = 25
    #     dim_gcn = self.GCN_out_channels
    #     batch_size = 32
    #
    #     x_i = K.reshape(x_i, [batch_size, max_body, joints, dim_gcn])
    #     x_i = conv_kernel(x_i)
    #     n, m, v, kc = batch_size, 4, 25, self.GCN_out_channels * self.match_A_kernel_size
    #     x_i = K.reshape(x_i, [n, m, v, self.match_A_kernel_size, self.GCN_out_channels])  # m,v,k,c
    #     x_i = tensorflow.transpose(x_i, perm=[0, 3, 4, 1, 2])  # n,k,c,m,v
    #     x_i = tensorflow.einsum('nkcmv,kvw->ncmw', x_i, self.A)
    #     x_i = tensorflow.transpose(x_i, perm=[0, 2, 3, 1])  ##n,m,w,c
    #     x_i = K.reshape(x_i, [batch_size, -1])  ##
    #     return x_i


class LSTM(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.
    """

    # @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 # kernel_initializer='glorot_uniform',
                 kernel_initializer='he_normal',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 A=None,
                 dim = None,
                 match_A_kernel_size=None,
                 batch_size=None,
                 joints = None,
                 max_body = None,
                 gcn_relu_flag = False,
                 output_dim_list=None,
                 concat_part = None,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = LSTMCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation,
                        A=A,
                        match_A_kernel_size=match_A_kernel_size,
                        batch_size=batch_size,
                        joints=joints,
                        dim = dim,
                        max_body=max_body,
                        gcn_relu_flag = gcn_relu_flag,
                        concat_part = concat_part,
                        output_dim_list = output_dim_list

                        )
        super(LSTM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(LSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(LSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)



