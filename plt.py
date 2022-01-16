import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np


seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import models
from models import segmentation
from utils.metrics.metrics import accuracy
from utils.generals import make_batch
import matplotlib.pyplot as plt
import itertools


def grid_gray_image(imgs, each_row: int):
    '''
    imgs shape: batch * size (e.g., 64x32x32, 64 is the number of the gray images, and (32, 32) is the size of each gray image)
    '''
    row_num = imgs.shape[0]//each_row
    for i in range(row_num):
        img = imgs[i*each_row]
        img = (img - img.min()) / (img.max() - img.min())
        for j in range(1, each_row):
            tmp_img = imgs[i*each_row+j]
            tmp_img = (tmp_img - tmp_img.min()) / (tmp_img.max() - tmp_img.min())
            img = np.hstack((img, tmp_img))
        if i == 0:
            ans = img
        else:
            ans = np.vstack((ans, img))
    return ans


def plt_show(x, fig_name="feature_map", each_row=8):
    img = x.cpu().detach().squeeze(0).numpy()
    img = grid_gray_image(img, each_row=each_row)
    plt.figure(figsize=(15, 15))
    plt.imshow(img, cmap='gray')
    plt.savefig(fig_name)
    plt.show()


def forward(model, x):
    print(x.shape)
    x = model.conv1(x)  # 112
    base_path = './images/'
    plt_show(x, fig_name=base_path+'feature_map[{}].png'.format(0))

    x = model.bn1(x)
    x = model.relu(x)
    if model.low_head_type:
        x = model.low_head_module(x)
    else:
        x = model.maxpool(x)  # 56

    if model.at_layer[0]:
        x = model.at0(x)
    
    plt_show(x, fig_name=base_path+'feature_map[{}].png'.format(1))

    x = model.layer1(x)  # 56
    if model.at_layer[1]:
        x = model.at1(x)
    
    plt_show(x, fig_name=base_path+'feature_map[{}].png'.format(2))

    x = model.layer2(x)  # 28
    if model.at_layer[2]:
        x = model.at2(x)
    
    plt_show(x, fig_name=base_path+'feature_map[{}].png'.format(3))

    x = model.layer3(x)  # 14
    if model.at_layer[3]:
        x = model.at3(x)
    
    plt_show(x, fig_name=base_path+'feature_map[{}].png'.format(4))

    x = model.layer4(x)  # 7
    if model.at_layer[4]:
        x = model.at4(x)
    
    plt_show(x, fig_name=base_path+'feature_map[{}].png'.format(5))

    if model._bc_kw:
        x = model.fc(x)
    else:
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(model.drop(x))
    return x


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(cm, classes, path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    total_num = 0
    right_num = 0
    for i in range(7):
        print("\n")
        for j in range(7):
            print(cm[i][j],"  ")
            if i ==j:
                total_num += cm[i][j]
                right_num += cm[i][j]
            else:
                total_num += cm[i][j]
    print("test_acc: ",right_num/total_num)


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path+'cfm.jpg')
    plt.show()


def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    # load model and data_loader
    model = get_model(configs)
    train_set, val_set, test_set = get_dataset(configs)

    from trainers.tta_trainer import FER2013Trainer

    # from trainers.centerloss_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    model = trainer._model
    checkpoint = torch.load(configs['model_path'])
    model.load_state_dict(checkpoint['net'])
    
    EMO_DICT = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral",
    }
    conf_matrix = torch.zeros(len(EMO_DICT), len(EMO_DICT))
    model.eval()
    with torch.no_grad():
        for idx in range(len(test_set)):
            images, targets = test_set[idx]
            targets = torch.LongTensor([targets])

            images = make_batch(images)
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # 绘制可视化图像
            forward(model, images)
            
            exit(0)

            outputs = model(images)
            outputs = torch.nn.functional.softmax(outputs, 1)

            # outputs.shape [tta_size, 7]
            outputs = torch.sum(outputs, 0)

            outputs = torch.unsqueeze(outputs, 0)
            # print(outputs.shape)
            # TODO: try with softmax first and see the change
            # acc = accuracy(outputs, targets)[0]
            conf_matrix = confusion_matrix(outputs, targets, conf_matrix)
            conf_matrix = conf_matrix.cpu()
    
    conf_matrix = np.array(conf_matrix.cpu()) # 将混淆矩阵从gpu转到cpu再转到np
    corrects = conf_matrix.diagonal(offset=0)# 抽取对角线的每种分类的识别正确个数
    per_kinds = conf_matrix.sum(axis=1) # 抽取每个分类数据总的测试条数

    print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), len(test_set)))
    print(conf_matrix)

    attack_types = [EMO_DICT[i] for i in range(len(EMO_DICT))]
    plot_confusion_matrix(conf_matrix, classes=attack_types, path='./images/', normalize=True, title='Normalized confusion matrix')


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    try:
        return models.__dict__[configs["arch"]]
    except KeyError:
        return segmentation.__dict__[configs["arch"]]


def get_dataset(configs):
    """
    This function get raw dataset
    """
    from utils.datasets.fer2013dataset import fer2013

    # todo: add transform
    train_set = fer2013("train", configs)
    val_set = fer2013("val", configs)
    tta = configs['tta'] if 'tta' in configs else False
    tta_size = configs['tta_size'] if 'tta_size' in configs else 10
    test_set = fer2013("test", configs, tta=tta, tta_size=tta_size)
    return train_set, val_set, test_set


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/plt_configs.json", help='model config path')
    opt = parser.parse_args()
    config_path = opt.config_path
    config_path = config_path.strip('\'').strip('\"')
    main(config_path)
