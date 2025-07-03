# python 配置

## python安装和换源

```bash
python -m pip install --upgrade pip   // 更新pip
pip3 config set global.index-url <https://pypi.mirrors.ustc.edu.cn/simple/>
pip3 config set global.index-url <https://pypi.tuna.tsinghua.edu.cn/simple>
```

## 通过requirements.txt文件进行批量安装

```bash
pip3 install -r requirements.txt

requirements.txt

numpy==1.22.2
matplotlib==3.5.1
pandas==1.4.0
```

## 查看已安装的库

```bash
pip3 list
```

## 离线安装

一、使用.whl文件
下载".whl"文件，网址：PyPI · The Python Package Index
 进入下载路径，在cmd中输入"pip install xxx.whl"
这种方法好像也要联网，有时会报SSL error。。
二、使用.tar.gz文件
下载".tar.gz"文件，网址：PyPI · The Python Package Index
进入下载路径，解压
进入文件目录，在cmd中输入"python setup.py install"
这种方法就不会报SSL的错误了，但有的包只有.whl文件，没有.tar.gz文件。。
三、使用源代码
从GitHub上下载源代码（.zip格式）：GitHub: Where the world builds software · GitHub
进入下载路径，解压
进入文件目录，在cmd中输入"python setup.py install"
这种方法通常不会报错，但安装的一般是最新版本。。

## VScode Python调试

目的：需要调试程序的时候提前注入环境变量

解决方案：在vscode的launch.json文件中进行修改，增加env关键字，和填写你想注入的环境变量

```bash
{
    // 使用 IntelliSense 了解相关属性。
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: <https://go.microsoft.com/fwlink/?linkid=830387>
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python 调试程序: 当前文件",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {"HF_ENDPOINT":"http://127.0.0.1:8080"}
        }
    ]
}
```

## 快速创建一个简易 HTTP 服务器（http.server）

使用场景: 简单测试两个主机是否是连通的（在有python环境的基础上）

安装
Python3 内置标准模块，无需安装。（在之前的 Python2 版本名称是 SimpleHTTPServer）
教程
用命令行创建
http.server 支持以 Python 解释器的 -m 参数直接调用。
通过执行如下命令创建一个最简单的 HTTP 服务器：

```bash
python -m http.server
```

服务器默认监听端口是 8000，支持自定义端口号：

```bash
python -m http.server 9000
```

服务器默认绑定到所有接口，可以通过 -b/--bind 指定地址，如本地主机：

```bash
python -m http.server --bind 127.0.0.1
```

服务器默认工作目录为当前目录，可通过 -d/--directory 参数指定工作目录：

```bash
python -m http.server --directory /tmp/
```

此外，可以通过传递参数 --cgi 启用 CGI 请求处理程序：

```bash
python -m http.server --cgi
```

## pip install llm-recipes 和pip install -e . 直接的区别

pip install xxx
这个命令用于安装一个已发布的 Python 包。当你在 PyPI（Python Package Index）上找到一个名为 xxx的包，并且它已经被发布并且可以被下载和安装时，你可以使用这个命令来安装它。安装后，包将被放置在你的 Python 环境的 xxx目录下，你可以通过导入 xxx 包来使用其中的模块和功能。
pip install -e .
这个命令用于安装一个从本地目录直接安装 Python 包。这里的 -e 参数代表 "editable"，意味着安装的包将被安装在本地目录下，并且在你修改源代码后，无需重新安装，包的版本会自动更新。这个命令通常用于开发阶段，当你正在开发一个包并且需要频繁地测试和修改代码时。
当你使用 pip install -e . 命令时，包将被安装在当前目录的 src 子目录下（如果存在的话），或者直接在当前目录下。这样，当你在本地目录下修改代码并重新运行 pip install -e . 时，修改的代码会立即生效，无需重新安装整个包。
