"""
@version: 0.1
@author: Blade He
@license: Morningstar 
@contact: blade.he@morningstar.com
@site: 
@software: PyCharm
@file: install_need_package.py
@time: 2019/03/12
"""
import os
import contextlib
import platform

system = platform.system()
print('current system is {0}'.format(system))
iswindows = False
pip = 'pip'
python = 'python'
if system.lower() == 'windows':
    iswindows = True
else:
    pip = 'pip3'
    python = 'python3'


def Schedule(blocknum, blocksize, totalsize):
    per = 100.0 * blocknum * blocksize / totalsize
    if per > 100:
        per = 100
    print("  " + "%.2f%% 已经下载的大小:%ld 文件大小:%ld" % (per, blocknum * blocksize, totalsize) + '\r')


headers = {'Connection': "keep-alive",
           'Upgrade-Insecure-Requests': '1',
           'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
           'Accept-Encoding': 'gzip, deflate, br',
           'Accept-Language': 'zh-CN,zh;q=0.9'}


def urlretrieve(url, filename=None, reporthook=None, params=None):
    import requests
    print('download file: {0}'.format(url))
    with contextlib.closing(requests.get(url,
                                         stream=True,
                                         headers=headers,
                                         params=params)) as fp:  # 打开网页
        header = fp.headers
        with open(filename, 'wb+') as tfp:  # w是覆盖原文件，a是追加写入 #打开文件
            blocksize = 10240
            totalsize = -1
            blocknum = 0
            if "content-length" in header:
                totalsize = int(header["Content-Length"])  # 文件的总大小理论值
                print('totalsize: {0}'.format(totalsize))
            else:
                print('can not find content-length')
            if reporthook and totalsize > 0:
                reporthook(blocknum, blocksize, totalsize)  # 写入前运行一次回调函数

            for chunk in fp.iter_content(chunk_size=blocksize):
                if chunk:
                    tfp.write(chunk)  # 写入
                    tfp.flush()
                    blocknum += 1
                    if reporthook and totalsize > 0:
                        reporthook(blocknum, blocksize, totalsize)  # 每写入一次就运行一次回调函数


def download_numpy():
    os.system('{0} install requests'.format(pip))
    downloadfolder = './download'
    if not os.path.exists(downloadfolder):
        os.makedirs(downloadfolder)

    url = r'https://download.lfd.uci.edu/pythonlibs/u2hcgva4/numpy-1.16.2+mkl-cp36-cp36m-win_amd64.whl'
    print('download file from: {0} begin'.format(url))
    filename = url.split('/')[len(url.split('/')) - 1]
    print(filename)
    filepath = os.path.join(downloadfolder,filename)
    urlretrieve(url=url, filename=filepath, reporthook=Schedule)
    print('download file from: {0} end'.format(url))

    if os.path.exists(filepath):
        os.system('{0} install {1}'.format(pip, filepath))


def startjob():
    print("""
#############################################################################################
Please confirm whether have installed C++ compile environment.
For Windows user, you should install Visual Studio 2015 or Visual Studio 2017 with VC++ components,
Or it's better to download .whl install package from https://www.lfd.uci.edu/~gohlke/pythonlibs/.

If opertation is Windows, the command of pip should be pip, python should be python
Otherwise, the command of pip should be pip3, python should be python3
Please check them at first!
#############################################################################################
""")
    try:
        import time
        tzname = time.tzname
        if tzname is not None and \
            len(tzname) > 0 and \
            'china' in str(tzname[0]).lower():
            print('Set python install package mirror in China begin')
            userfolder = os.path.expanduser('~')
            pip_config_folder = os.path.join(userfolder, 'pip')
            if not os.path.exists(pip_config_folder):
                os.makedirs(pip_config_folder)

            pip_config_file = os.path.join(pip_config_folder, 'pip.ini')
            print('pip config file is: {0}'.format(pip_config_file))
            if not os.path.exists(pip_config_file):
                with open(pip_config_file, mode='w', encoding='utf-8') as file:
                    file.writelines(['[global]\n', 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple'])
            else:
                print('You have set python install package mirror in China.')
            print('Set python install package mirror in China end')
            print('The python install package mirror in China is:')
            with open(pip_config_file, mode='r', encoding='utf-8') as file:
                print(file.read())
    except Exception as e:
        print(e)

    print('Upgrade pip begin')
    os.system('{0} -m {1} install --upgrade pip'.format(python, pip))
    print('Upgrade pip end')

    os.system('{0} install Cython'.format(pip))

    install_numpy_sklearn()
    install_tensor_flow_torch()

    all_requirement_file = 'all_requirements.txt'
    if os.path.exists(all_requirement_file):
        print('Install all of packages in requirements')
        with open(all_requirement_file, mode='r', encoding='utf-8') as file:
            for package in file.readlines():
                if package and len(package) > 0:
                    try:
                        os.system('{0} install {1}'.format(pip, package))
                    except Exception as e:
                        print(e)
        print('All of packages in requirements have been installed, please notice warning or error during installation.')

    install_nltk()
    install_spacy()
    print('Install process is done!')


def install_tensor_flow_torch():
    has_gpu = False
    try:
        print('Install nvidia-ml-py3')
        os.system('{0} install nvidia-ml-py3'.format(pip))
        import pynvml
        pynvml.nvmlInit()
        # handle = len(pynvml.nvmlDeviceGetHandleByIndex(0))  # 这里的0是GPU id
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('GPU memory size: {0}'.format(meminfo.total))  # 第二块显卡总的显存大小
        # print('GPU memory used: {0}'.format(meminfo.used))  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
        # print('GPU memory free: {0}'.format(meminfo.free))  # 第二块显卡剩余显存大小
        print('GPU Amount: {0}'.format(pynvml.nvmlDeviceGetCount()))  # 显示有几块GPU
        has_gpu = True
        print('Install Tensorflow-GPU')
        os.system('{0} install tensorflow-gpu==1.13.1'.format(pip))
        try:
            print('Try to run Tensorflow')
            test_tensor_flow()
            print('Succeed to run Tensorflow, congratulations!')
        except Exception as e:
            print(e)
            print("""
            Please confirm below requirements on GPU machine:
            Python 3.6.x
            Tensorflow 1.13
            CUDA 10.0: https://developer.nvidia.com/cuda-10.0-download-archive
            cuDNN v7.50 (Feb 21, 2019) for CUDA 10: https://developer.nvidia.com/cudnn
            If you have installed them, please restart your computer to retry!
            """)
        print('Install Pytorch-GPU')
        try:
            import torch
        except:
            if iswindows:
                os.system('{0} install https://download.pytorch.org/whl/cu100/torch-1.0.1-cp36-cp36m-win_amd64.whl'
                          .format(pip))
            else:
                os.system('{0} install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl'
                          .format(pip))
            os.system('{0} install torchvision'.format(pip))
    except Exception as e:
        print(e)
        if not has_gpu:
            print('Install Tensorflow-CPU')
            try:
                import tensorflow
            except:
                os.system('{0} install tensorflow==1.12.0'.format(pip))
            print('Install Pytorch-CPU')
            try:
                import torch
            except:
                if iswindows:
                    os.system('{0} install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-win_amd64.whl'
                              .format(pip))
                else:
                    os.system('{0} install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl'
                              .format(pip))
                os.system('{0} install torchvision'.format(pip))


def test_tensor_flow():
    import tensorflow as tf
    const = tf.constant(2.0, name='const')

    b = tf.Variable(2.0, name='b')
    c = tf.Variable(1.0, dtype=tf.float32, name='c')

    d = tf.add(b, c, name='d')
    e = tf.add(c, const, name='e')
    a = tf.multiply(d, e, name='a')

    init_op = tf.global_variables_initializer()
    # session
    with tf.Session() as sess:
        sess.run(init_op)
        # 计算
        a_out = sess.run(a)
        print("Variable a is {}".format(a_out))



def install_numpy_sklearn():
    print('Check whether have installed Numpy and Scikit-Learn')
    try:
        import numpy
        print('Numpy has been installed.')
    except Exception as e:
        print('need install numpy')
        if iswindows:
            download_numpy()
        else:
            os.system('{0} install numpy'.format(pip))

    try:
        import sklearn
        print('Scikit-Learn has been installed.')
    except Exception as e:
        print('need install scikit-learn')
        os.system('{0} install scikit-learn'.format(pip))


def install_nltk():
    try:
        import nltk
        install_nltk_data()
    except:
        print('Install nltk')
        os.system('{0} install nltk'.format(pip))
        install_nltk_data()


def install_nltk_data():
    try:
        import nltk
        print('Install necessary nltk data')
        data_list = ['stopwords',
                     'wordnet',
                     'averaged_perceptron_tagger',
                     'punkt']
        for data in data_list:
            try:
                print('install nltk: {0}'.format(data))
                nltk.download(data)
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)


def install_spacy():
    try:
        import spacy
        install_spacy_model()
    except:
        print('Install Spacy')
        os.system('{0} install spacy'.format(pip))
        install_spacy_model()


def install_spacy_model():
    print('Install necessary Spacy model')
    try:
        import en_core_web_sm
        print('Spacy model: en_core_web_sm has been installed.')
    except:
        print('Install Spacy English Small Model')
        os.system('{0} -m spacy download en_core_web_sm'.format(python))
        try:
            os.system('{0} -m spacy download en'.format(python))
        except Exception as e:
            print(e)

    try:
        import en_core_web_lg
        print('Spacy model: en_core_web_lg has been installed.')
    except:
        print(
            'Install Spacy English Large Model, this package is big, if you need abort this process, please enter CTRL+C to break.')
        os.system('{0} -m spacy download en_core_web_lg'.format(python))

if __name__ == '__main__':
    startjob()