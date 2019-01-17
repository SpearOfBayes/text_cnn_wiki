import numpy as np
import re
import jieba
import jieba.analyse
from encode_table import labels
from tqdm import *

label_dic = {}

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub(r"[\u4E00-\u9FA5]+", " ", string)
    return string.strip().lower()

def segmentation(word):
	# allowPos() : https://blog.csdn.net/hhtnan/article/details/77650128
	l =  jieba.analyse.extract_tags(word, topK=20, withWeight=False, allowPOS=('nz', 'n', 'vn', 'v', 'a'))
	tmp = ' ' #指定连接字符
	return tmp.join(l)

def onehot_encode(y):
    # for i in range(len(label_set)):
    #     # label_dic是一个全局变量
    #     label_dic[label_set[i]] = i
    # codes = map(lambda x: label_dic[x], y)
    codes = map(lambda x: labels[x], y)
    y_encoded = []
    length = len(labels) #获取标签集的长度
    # 为每一个数编码
    print('正在对标签进行one hot编码。。。。')
    for code in tqdm(codes):
        array = [0 for _ in range(length)]
        array[code] = 1
        y_encoded.append(array)
    return y_encoded

def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # 从文件夹中加载数据
    # 首先讲文本文件里面的内容按行读取出来
    # 然后做成一个list形式
    # 接着trip去掉每一行首尾
    # positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # x_text将两种分类的数据集合并
    # clean_str用于清除掉一些无用的字符，然后一律转化成小写
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # 创建标签
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    # y的结构是这样的：
    # [
    #   [0 1]
    #   [0 1]
    #   [0 1]
    #   ...
    #   [1 0]
    #   [1 0]
    #   [1 0]
    # ]

    x_text = []
    y = []
    # 读取data_file中的所有数据记录
    f = open(data_file, "r", encoding='utf-8')
    # 掉第一行
    f.readline()
    while True:
        line = f.readline() #读取每行数据
        if not line:
            break
        item, label = line.split(',')
        #去掉末尾的'\n'
        label = label.strip('\n')
        # clean数据
        # item, label = clean_str(item), clean_str(label)
        # 加入数据集

        x_text.append(item)
        y.append(label)

    f.close()

    print('正在对商品名进行分词。。。。')
    for i in tqdm(range(len(x_text))):
        x_text[i] = segmentation(x_text[i])
    y = onehot_encode(y)
    y = np.array(y)

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    创建一个数据集的批量迭代器
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # 每一轮都打乱数据
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data_to_be_predicted(data_file):
    f = open(data_file, 'r')
    return f.read().split('\n')


def get_label_name(index):
    for key in labels:
        if labels[key] == index:
            return key

    return 'not found'

def get_label_name_list(label_nums):
    label_names = []
    for i in range(len(label_nums)):
        label_names.append(get_label_name(int(label_nums[i])))
    return label_names
    
if __name__ == '__main__':
    [x_text, y] = load_data_and_labels('./data/mini_train.csv')
    print(x_text[:10])
    print(y[:10])