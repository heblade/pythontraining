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


def Schedule(a, b, c):
    per = 100.0 * a * b / c
    if per > 100:
        per = 100
    print("  " + "%.2f%% 已经下载的大小:%ld 文件大小:%ld" % (per, a * b, c) + '\r')


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
Please confirm whether have installed C++ compile environment, 
In Windows, you should install Visual Studio 2015 or Visual Studio 2017 with VC++ components.""")
    print('If opertation is Windows, the command of pip should be pip, python should be python')
    print('Otherwise, the command of pip should be pip3, python should be python3')
    print('Please check them at first!')

    try:
        import numpy
    except Exception as e:
        print('need install numpy')
        if iswindows:
            download_numpy()
        else:
            os.system('{0} install numpy'.format(pip))

    try:
        import sklearn
    except Exception as e:
        print('need install scikit-learn')
        os.system('{0} install sklearn'.format(pip))

    if os.path.exists('all_requirements.txt'):
        print('Install all of packages in requirements')
        os.system('{0} install -r all_requirements.txt'.format(pip))
        print('All of packages in requirements have been installed, please notice warning or error during installation.')
    try:
        import spacy
        print('Install necessary Spacy model')
        try:
            import en_core_web_sm
        except:
            print('Install Spacy English Small Model')
            os.system('{0} -m spacy download en_core_web_sm'.format(python))

        try:
            import en_core_web_lg
        except:
            print('Install Spacy English Large Model')
            os.system('{0} -m spacy download en_core_web_lg'.format(python))
    except:
        print('Please install Spacy!')
    print('Install process is done!')


if __name__ == '__main__':
    startjob()