# Git 常用命令速查手册

## 1. 基础配置命令

```bash
# 配置用户信息
git config --global user.name "your_username"    # 设置全局用户名
git config --global user.email "your_email"      # 设置全局邮箱
git config --list                                # 查看所有配置信息
git config --global --edit                       # 直接编辑配置文件
git config --global core.autocrlf true          # 配置Git自动处理行结束符(Windows建议设置true)
git config core.editor vim                      # 设置默认编辑器为vim
```

## 2. 仓库初始化和远程关联

```bash
git init                                        # 将当前目录初始化为Git仓库
git remote add origin <远程仓库URL>             # 添加远程仓库地址
git remote -v                                   # 查看已配置的远程仓库
```

## 3. 基本操作命令

```bash
# 文件暂存和提交
git add <文件名>                                # 将指定文件添加到暂存区
git add .                                      # 将所有修改的文件添加到暂存区
git commit -m "提交说明"                        # 提交暂存区的文件到仓库
git commit -s -m "提交说明"                     # 在提交信息中添加签名
git commit -v --amend                          # 修改最近一次的提交信息

# 状态查看
git status                                     # 查看仓库当前状态
git diff <文件名>                              # 查看指定文件的修改内容
git log --graph --pretty=oneline --abbrev-commit  # 以图形方式查看提交历史
```

## 4. 分支操作命令

```bash
# 分支管理
git branch                                     # 查看本地分支
git branch -a                                  # 查看所有分支(包括远程)
git checkout -b <分支名>                       # 创建并切换到新分支
git switch -c <分支名>                         # 创建并切换到新分支(新版Git推荐用法)
git checkout <分支名>                          # 切换到指定分支
git switch <分支名>                            # 切换到指定分支(新版Git推荐用法)
git merge <源分支>                             # 将指定分支合并到当前分支
git branch -d <分支名>                         # 删除本地分支
git push origin --delete <分支名>              # 删除远程分支
git checkout -b <本地分支> origin/<远程分支>    # 基于远程分支创建本地分支

# 特殊操作
git cherry-pick <commit-id>                    # 将指定提交应用到当前分支
```

## 5. 标签管理

```bash
git tag <标签名>                               # 创建标签
git tag                                       # 查看本地所有标签
git ls-remote --tags                          # 查看远程仓库所有标签
git tag -d <标签名>                           # 删除本地标签
git checkout -b <分支名> <标签名>              # 基于标签创建新分支
```

## 6. 撤销和回滚操作

```bash
git reset --hard HEAD                         # 回滚到最近一次提交(会丢失工作区修改)
git reset --soft HEAD                         # 回滚到最近一次提交(保留工作区修改)
git reset --hard <commit-id>                  # 回滚到指定提交
git checkout -- <文件名>                      # 撤销指定文件的修改
git checkout .                                # 撤销所有未提交的修改
```

## 7. 临时保存工作区

```bash
git stash                                     # 临时保存当前工作区的修改
git stash list                                # 查看所有保存的工作现场
git stash pop                                 # 恢复最近一次保存的工作现场并删除记录
git stash apply                               # 恢复指定的工作现场(不删除记录)
```

## 8. 远程同步操作

```bash
git push origin <分支名>                       # 推送本地分支到远程
git pull origin <分支名>                       # 拉取远程分支并合并
git pull --rebase origin <分支名>             # 使用rebase方式拉取并合并远程分支
git clone -b <分支名> <仓库URL>               # 克隆指定分支的代码
git clone --branch <标签名> <仓库URL>         # 克隆指定标签的代码
```

## Git之配置多SSH-KEY

现存账号类型:

1. gitee账号，国产github，码云值得拥有
2. github账号，全球最大同性交友网站
3. gitlab账号，公司内部的工作开发

解决办法很简单，新建一个config文件就可以了。

1. 本地生成SSH-Key
    生成一个gitee用的SSH-Key

    ```bash
    ssh-keygen -t rsa -C 'xxxxx@163.com' -f ~/.ssh/gitee_id_rsa
    ```

    生成一个github用的SSH-Key

    ```bash
    ssh-keygen -t rsa -C 'xxxxx@163.com' -f ~/.ssh/github_id_rsa
    ```

    生成一个公司用gitlab的SSH-Key

    ```bash
    ssh-keygen -t rsa -C 'xxxxx@company.com' -f ~/.ssh/gitlab_id_rsa
    ```

2. git平台设置中
将 .pub 文件中内容复制到相应的平台SSH设置中，
例如：gitee_id_rsa.pub中的文件内容复制到码云的SSH公钥中。

3. 新建config文件
    在 ~/.ssh 目录下新建一个config文件(无扩展名)，添加如下内容：

    ```bash
    # gitee
    Host gitee.com
    HostName gitee.com
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/gitee_id_rsa

    # github
    Host github.com
    HostName github.com
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/github_id_rsa

    # gitlab
    Host xxxx.com
    HostName xxxx.com
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/gitlab_id_rsa
    ```

    其中Host和HostName填写git服务器的域名，IdentityFile指定私钥的路径。
实在不晓得Host和HostName怎么填的呢，譬如不知道公司内部的服务器域名，可以在known_hosts文件中查看。
4. 测试是否联通
    用ssh命令分别测试，@后面跟着的是域名呀。

    ```bash
    ssh -T git@gitee.com
    ssh -T git@github.com
    ssh -T git@xxxx.com
    ```

看到界面返回欢迎标语的时候，就说明成功啦。
至此，你就可以在同一台电脑上，愉快的进行各大平台的代码提交。

## git 大文件上传和版本控制

LFS（Large File Storage）大文件存储用于改善大型二进制文件的处理方式，它通过分离较大的文件和文本差异，使版本控制更加高效。LFS的使用包括：安装Git LFS扩展、配置LFS追踪特定文件类型、提交和推送大文件至GitLab。其中，配置Git LFS追踪文件是关键步骤，这包括编辑.gitattributes文件和指定要追踪的文件类型。

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git lfs version

touch .gitattributes
*.zip filter=lfs diff=lfs merge=lfs -text

git add .gitattributes
git add your-large-file.zip
git commit -m "Add LFS tracking for zip files"
git push origin mAIn


git lfs pull
```


export GOSUMDB=off && go mod tidy    可以解决拉去gitlab仓库代码的问题。