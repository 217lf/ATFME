import glob
import cv2
import numpy as np


def create_file_list():
    filename = glob.glob('./flow_data/*/*')
    print(len(filename))

    sublist = []
    for i in range(len(filename)):
        if filename[i].split('/')[-1].split('_')[2] == 'smic':
            sublist.append(filename[i].split('/')[-1].split('_')[2] + '_'
                           + filename[i].split('/')[-1].split('_')[4])
        else:
            sublist.append(filename[i].split('/')[-1].split('_')[2] + '_' + filename[i].split('/')[-1].split('_')[3])

    sublist_quchong = list(set(sublist))  # 去掉重复的文件名
    sublist_quchong.sort()  # 排序
    train_sub = []  # 训练受试者列表
    test_sub = []  # 测试受试者列表
    for i in range(len(sublist_quchong)):
        tmp_list = []
        tmp_list.append(sublist_quchong[i])
        test_sub.append(tmp_list)
        train_val = list(set(sublist_quchong) - set(tmp_list))
        tmp = []
        tmp_train = list(set(train_val) - set(tmp))
        train_sub.append(tmp_train)
    return filename, train_sub, test_sub

def read_file_directly(filename, test_subject, train_sub, test_sub):
    fold = test_subject
    train_sub_list = []
    test_sub_list = []
    test_sample_list = []
    for i in range(len(filename)):
        # temp for smic_hs
        # temp1 for casme2 and samm
        tmp = filename[i].split('/')[-1].split('_')[2] + '_' + filename[i].split('/')[-1].split('_')[4]
        tmp1 = filename[i].split('/')[-1].split('_')[2] + '_' + filename[i].split('/')[-1].split('_')[3]
        for j in range(len(train_sub[fold])):
            if tmp == train_sub[fold][j]:
                train_sub_list.append(filename[i])
            if tmp1 == train_sub[fold][j]:
                train_sub_list.append(filename[i])
        for m in range(len(test_sub[fold])):
            if tmp == test_sub[fold][m]:
                test_sub_list.append(filename[i])
            if tmp1 == test_sub[fold][m]:
                test_sub_list.append(filename[i])

    x_train_image_list = []
    y_train_image_list = []
    z_train_image_list = []
    train_label_list = []
    for i in range(len(train_sub_list)):
        # print('train_img:',train_sub_list[i])
        img = cv2.imread(train_sub_list[i])
        img = cv2.resize(img, (28, 28))
        img = np.array(img)
        # print(train_sub_list[i].split('/'))
        tmp = int(train_sub_list[i].split('/')[-2])
        if train_sub_list[i].split('/')[-1].split('_')[1] == 'x':
            x_train_image_list.append(img)
            train_label_list.append(tmp)

        elif train_sub_list[i].split('/')[-1].split('_')[1] == 'y':
            y_train_image_list.append(img)

        else:
            z_train_image_list.append(img)

    x_test_image_list = []
    y_test_image_list = []
    z_test_image_list = []
    test_label_list = []
    for i in range(len(test_sub_list)):
        # print('test_img:', test_sub_list[i])
        img = cv2.imread(test_sub_list[i])
        img = cv2.resize(img, (28, 28))
        img = np.array(img)
        tmp = int(test_sub_list[i].split('/')[-2])

        if test_sub_list[i].split('/')[-1].split('_')[1] == 'x':
            x_test_image_list.append(img)
            test_label_list.append(tmp)

        elif test_sub_list[i].split('/')[-1].split('_')[1] == 'y':
            y_test_image_list.append(img)

        else:
            z_test_image_list.append(img)

    train_x_array = np.array(x_train_image_list)
    train_y_array = np.array(y_train_image_list)
    train_z_array = np.array(z_train_image_list)
    train_label = np.array(train_label_list)
    test_x_array = np.array(x_test_image_list)
    test_y_array = np.array(y_test_image_list)
    test_z_array = np.array(z_test_image_list)
    test_label = np.array(test_label_list)
    return train_x_array, train_y_array, train_z_array, train_label,test_x_array, test_y_array, test_z_array, test_label, test_sample_list
