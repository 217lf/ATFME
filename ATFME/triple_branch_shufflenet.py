from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.optimizers import *
from triple_branch_process import *
from callbacks import *
import tensorflow as tf
import pandas as pd
import time
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras.backend.tensorflow_backend as K
from keras.layers import * 
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

train_num_list = []
full_f1_list = []
casme2_f1_list = []
samm_f1_list = []
smic_f1_list = []

full_uar_list = []
casme2_uar_list = []
samm_uar_list = []
smic_uar_list = []

full_matrix_list = []
casme2_matrix_list = []
samm_matrix_list = []
smic_matrix_list = []


def evaluate_metrics(predict_label, true_label, num):
    # predict_label = np.array([0, 1, 2])
    # true_label = np.array([0, 2, 1])
    # macro_f1
    full_f1 = f1_score(true_label, predict_label, average='macro')
    casme2_f1 = f1_score(true_label[0:145], predict_label[0:145], average='macro')
    samm_f1 = f1_score(true_label[145:278], predict_label[145:278], average='macro')
    smic_f1 = f1_score(true_label[278:], predict_label[278:], average='macro')

    full_f1_list.append(full_f1)
    casme2_f1_list.append(casme2_f1)
    samm_f1_list.append(samm_f1)
    smic_f1_list.append(smic_f1)

    print('f1 macro score:', full_f1)
    print('CASME2 f1 macro score:', casme2_f1)
    print('SAMM f1 macro score:', samm_f1)
    print('SMIC-hs f1 macro score:', smic_f1)

    # matrix
    confusion_matrixs = confusion_matrix(true_label, predict_label)
    casme2_confusion_matrix = confusion_matrix(true_label[0:145], predict_label[0:145])
    samm_confusion_matrix = confusion_matrix(true_label[145:278], predict_label[145:278])
    smic_hs_confusion_matrix = confusion_matrix(true_label[278:], predict_label[278:])

    full_matrix_list.append(confusion_matrixs)
    casme2_matrix_list.append(casme2_confusion_matrix)
    samm_matrix_list.append(samm_confusion_matrix)
    smic_matrix_list.append(smic_hs_confusion_matrix)
    print('full confusion_matrix:', confusion_matrixs)
    print('casme2 confusion_matrix:', casme2_confusion_matrix)
    print('samm confusion_matrix:', samm_confusion_matrix)
    print('smic_hs confusion_matrix:', smic_hs_confusion_matrix)

    # uar
    # total uar
    d = np.diag(confusion_matrixs)
    m = np.sum(confusion_matrixs, axis=1)
    uar = np.sum(d / m / num_class)
    print('uar:', uar)
    # casme2 uar
    casme2_d = np.diag(casme2_confusion_matrix)
    casme2_m = np.sum(casme2_confusion_matrix, axis=1)
    casme2_uar = np.sum(casme2_d / casme2_m / num_class)
    print('casme2_uar:', casme2_uar)
    # samm uar
    samm_d = np.diag(samm_confusion_matrix)
    samm_m = np.sum(samm_confusion_matrix, axis=1)
    samm_uar = np.sum(samm_d / samm_m / num_class)
    print('samm_uar:', samm_uar)
    # hs uar
    smic_hs_d = np.diag(smic_hs_confusion_matrix)
    smic_hs_m = np.sum(smic_hs_confusion_matrix, axis=1)
    smic_hs_uar = np.sum(smic_hs_d / smic_hs_m / num_class)
    print('smic_hs_uar:', smic_hs_uar)

    full_uar_list.append(uar)
    casme2_uar_list.append(casme2_uar)
    samm_uar_list.append(samm_uar)
    smic_uar_list.append(smic_hs_uar)

    train_num_list.append(num)
    df = pd.DataFrame(data={'train_num': train_num_list,
                            'full_f1_list': full_f1_list, 'casme2_f1_list': casme2_f1_list,
                            'samm_f1_list': samm_f1_list, 'smic_f1_list': smic_f1_list,
                            'full_uar_list': full_uar_list, 'casme2_uar_list': casme2_uar_list,
                            'samm_uar_list': samm_uar_list, 'smic_uar_list': smic_uar_list,
                            'full_matrix_list': full_matrix_list, 'casme2_matrix_list': casme2_matrix_list,
                            'samm_matrix_list': samm_matrix_list, 'smic_matrix_list': smic_matrix_list
                            })
    df.to_csv('save_result/metrics.csv')

def _group_conv(x, filters, kernel, stride, groups):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]

    # number of input channels per group
    nb_ig = in_channels // groups
    # number of output channels per group
    nb_og = filters // groups

    gc_list = []
    # Determine whether the number of filters is divisible by the number of groups
    assert filters % groups == 0

    for i in range(groups):
        if channel_axis == -1:
            x_group = Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        gc_list.append(Conv2D(filters=nb_og, kernel_size=kernel, strides=stride,
                              padding='same', use_bias=False)(x_group))

    return Concatenate(axis=channel_axis)(gc_list)


def _channel_shuffle(x, groups):

    if K.image_data_format() == 'channels_last':
        height, width, in_channels = K.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, height, width, groups, channels_per_group]
        dim = (0, 1, 2, 4, 3)
        later_shape = [-1, height, width, in_channels]
    else:
        in_channels, height, width = K.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, groups, channels_per_group, height, width]
        dim = (0, 2, 1, 3, 4)
        later_shape = [-1, in_channels, height, width]

    x = Lambda(lambda z: K.reshape(z, pre_shape))(x)
    x = Lambda(lambda z: K.permute_dimensions(z, dim))(x)
    x = Lambda(lambda z: K.reshape(z, later_shape))(x)

    return x


def _shufflenet_unit(inputs, filters, kernel, stride, groups, stage, bottleneck_ratio=1):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(inputs)[channel_axis]
    bottleneck_channels = int(filters * bottleneck_ratio)

    if stage == 2:
        x = Conv2D(filters=bottleneck_channels, kernel_size=kernel, strides=1,
                   padding='same', use_bias=False)(inputs)
    else:
        x = _group_conv(inputs, bottleneck_channels, (1, 1), 1, groups)
    x = BatchNormalization(axis=channel_axis)(x)
    x = ReLU()(x)

    x = _channel_shuffle(x, groups)

    x = DepthwiseConv2D(kernel_size=kernel, strides=stride, depth_multiplier=1,
                        padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if stride == 2:
        x = _group_conv(x, filters - in_channels, (1, 1), 1, groups)
        x = BatchNormalization(axis=channel_axis)(x)
        avg = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(inputs)
        x = Concatenate(axis=channel_axis)([x, avg])

        
        
    else:
        x = _group_conv(x, filters, (1, 1), 1, groups)
        x = BatchNormalization(axis=channel_axis)(x)
        x = add([x, inputs])

    return x


def _stage(x, filters, kernel, groups, repeat, stage):

    x = _shufflenet_unit(x, filters, kernel, 2, groups, stage)

    for i in range(1, repeat):
        x = _shufflenet_unit(x, filters, kernel, 1, groups, stage)

    return x



def ShuffleNet():
    input_1 = Input(shape=[28, 28, 3], name='inputs_x')
    input_2 = Input(shape=[28, 28, 3], name='inputs_y')
    input_3 = Input(shape=[28, 28, 3], name='inputs_z')

    ince_conv1_1 = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=True, activation='relu')(input_1)
    ince_conv1_2 = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=True, activation='relu')(input_2)
    ince_conv1_3 = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=True, activation='relu')(input_3)
    
    
    
    ince_1_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(ince_conv1_1)
    ince_1_2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(ince_conv1_2)
    ince_1_3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(ince_conv1_3)

    ince_2_1 = _stage(ince_1_1, filters=384, kernel=(3, 3), groups=4, repeat=4, stage=2)
    ince_2_2 = _stage(ince_1_2, filters=384, kernel=(3, 3), groups=4, repeat=4, stage=2)
    ince_2_3 = _stage(ince_1_3, filters=384, kernel=(3, 3), groups=4, repeat=4, stage=2)


    ince_conv2_1 = Conv2D(512, kernel_size=2, padding='same', strides=1, name='1x1conv5_out', activation='relu')( ince_2_1)
    ince_conv2_2 = Conv2D(512, kernel_size=2, padding='same', strides=1, name='1x2conv5_out', activation='relu')( ince_2_2)
    ince_conv2_3 = Conv2D(512, kernel_size=2, padding='same', strides=1, name='1x3conv5_out', activation='relu')(ince_2_3)

    
    # Channel fusion attention module(CFAM)
    concat = Concatenate(axis=-1)([ince_conv2_1, ince_conv2_2, ince_conv2_3])
    squeeze = GlobalAveragePooling2D()(concat)
    excitation = Dense(384, activation='relu')(squeeze)
    excitation = Dense(1536, activation='sigmoid')(excitation)
    excitation = Reshape((1,1, 1536))(excitation)
    scale = multiply([concat, excitation])
        
    x = GlobalAveragePooling2D()(scale)
    x = Dropout(0.2)(x)


    x = Dense(num_class, activation='softmax')(x)

    model = Model(inputs=[input_1, input_2, input_3], outputs=x)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # compile model
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = ShuffleNet()

print(model.summary())


if __name__ == '__main__':

    # model = create_inception_model()
    # # model.summary()
    # plot_model(model, model.name+'.png', show_shapes=True, show_layer_names=True)
    pre_epoch = 0
    total_epochs = 60
    epochs = 60
    [filename, train_sub,test_sub] = create_file_list()
    for round_time in range(9, 10):
        start = time.perf_counter()
        numth_test_sample_list = []
        predict_label_list = []
        true_label_list = []
        all_acc = []  # 每一轮验证准确率
        print(len(test_sub))
        for sub in range(68):
            print('********  ', sub, '   folder start:', test_sub[sub])

            [train_x_array, train_y_array,train_z_array, train_label,
             test_x_array, test_y_array,test_z_array,test_label, x_list] = read_file_directly(filename, sub, train_sub,  test_sub)

            '''
            归一化的目标是将每一维特征压缩到一定范围之内，以免不同特征因取值范围不同而影响其权重。
            非常大或非常小的值搭配上不恰当的学习率，往往使得收敛过慢，或者因每次调整的波动太大最终
            无法收敛。归一化去除了这些不稳定因素。
            '''
            train_x_array = train_x_array.astype('float32')
            train_set_mean = train_x_array - np.mean(train_x_array)
            std = np.std(train_x_array)
            train_x_array = train_set_mean / std

            train_y_array = train_y_array.astype('float32')
            train_set_mean = train_y_array - np.mean(train_y_array)
            std = np.std(train_y_array)
            train_y_array = train_set_mean / std
            
            train_z_array = train_z_array.astype('float32')
            train_set_mean = train_z_array - np.mean(train_z_array)
            std = np.std(train_z_array)
            train_z_array = train_set_mean / std

            test_x_array = test_x_array.astype('float32')
            test_set_mean = test_x_array - np.mean(test_x_array)
            std = np.std(test_x_array)
            test_x_array = test_set_mean / std

            test_y_array = test_y_array.astype('float32')
            test_set_mean = test_y_array - np.mean(test_y_array)
            std = np.std(test_y_array)
            test_y_array = test_set_mean / std
            
            test_z_array = test_z_array.astype('float32')
            test_set_mean = test_z_array - np.mean(test_z_array)
            std = np.std(test_z_array)
            test_z_array = test_set_mean / std


            for len in range(np.shape(x_list)[0]):
                numth_test_sample_list.append(x_list[len])
            train_label = to_categorical(train_label, num_class)
            test_label = to_categorical(test_label, num_class)

            checkpoint_path = './checkpoint/model_roundtime_'+str(round_time)+'_sub_'+str(sub)+'_epoch_'+str(total_epochs)+'.h5'
            checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)
            callback_list = [checkpoint, tensorborad]
            if not os.path.exists('./checkpoint/model_roundtime_'+str(round_time)+'_sub_'+str(sub)+'_epoch_'+str(total_epochs)+'.h5'):
                model =  ShuffleNet()
                hist = model.fit({'inputs_x': train_x_array, 'inputs_y': train_y_array,'inputs_z': train_z_array},
                                 train_label,
                                 batch_size=32,
                                 class_weight='auto',
                                 shuffle=True,
                                 epochs=epochs, verbose=2, callbacks=callback_list)  # starts training
                model.save_weights('./checkpoint/model_roundtime_'+str(round_time)+'_sub_'+str(sub)+'_epoch_'+str(total_epochs)+'.h5')
                # save loss
                his_log = './his_log/shufflenet_roundtime_'+str(round_time)+'_sub_' + str(sub) + '_epoch_'+str(total_epochs)+'.txt'
                with open(his_log, 'w') as f:
                    f.write(str(hist.history))
                    f.close()
            model2 = ShuffleNet()
            model2.load_weights(checkpoint_path)
            emotion_classes = model2.predict(
                {'inputs_x': test_x_array, 'inputs_y': test_y_array,'inputs_z': test_z_array }, batch_size=1)
            # revert clssed to 0,1,2...,and save to predict file
            # revert clssed to 0,1,2...,and save to predict file
            predict_label = np.argmax(emotion_classes, axis=1)
            predict_label_list += list(predict_label)
            print('本轮预测标签：', predict_label)
            true_label = np.argmax(test_label, axis=1)
            true_label_list += list(true_label)
            print('本轮真实标签：', true_label)
            right = 0
            for i in range(len(predict_label)):
                if predict_label[i] == true_label[i]:
                    right += 1
            print('正确数：', right)
            print('本轮验证准确率：', (right / true_label.shape[0]))
            all_acc.append((right / true_label.shape[0]))
            print('各轮验证准确率：', all_acc)
            print('目前留一验证准确率：', sum(np.array(all_acc)) / len(all_acc))
            K.clear_session()
            tf.reset_default_graph()

            end = time.perf_counter()
            print('Running time: %s Seconds' % (end - start))

        print('各轮验证准确率：', all_acc)
        all_acc = np.array(all_acc)
        result_f = open('./save_result/acc_roundtime_' + str(round_time) + '_epoch_' + str(total_epochs) + '.txt', 'w')
        result_f.write(str(all_acc))
        result_f.close()
        # save_predict_label
        predict_label_list = np.array(predict_label_list)
        result_f = open('./save_result/result__roundtime_' + str(round_time) + '_epoch_' + str(total_epochs) + '.txt',
                        'w')
        result_f.write(str(predict_label_list))
        result_f.close()
        # save_true label
        true_f = open('./save_result/true_roundtime_' + str(round_time) + '_epoch_' + str(total_epochs) + '.txt', 'w')
        true_label_list = np.array(true_label_list)
        true_f.write(str(true_label_list))
        true_f.close()
        # save file names
        file_f = open('./save_result/file_names_roundtime_' + str(round_time) + '_epoch_' + str(total_epochs) + '.txt',
                      'w')
        # numth_test_sample_list = np.ndarray(numth_test_sample_list)
        numth_test_sample_list = np.array(numth_test_sample_list)

        # print(numth_test_sample_list)
        file_f.write(str(numth_test_sample_list))
        file_f.close()
        evaluate_metrics(predict_label_list, true_label_list, round_time)

        numth_test_sample_list = numth_test_sample_list.tolist()
        true_label_list = true_label_list.tolist()
        predict_label_list = predict_label_list.tolist()

        da = pd.DataFrame(data={'file_list': numth_test_sample_list, 'true_label': true_label_list,
                                'predict_label': predict_label_list
                                })
        da.to_csv('./save_result/results_roundtime_' + str(round_time) + '_epoch_' + str(total_epochs) + '.csv')
        #print the evaluate results
        #plot_Matrix(confusion_matrixs, classes=3, title=None, cmap=plt.confusion_matrixs.Blues)
