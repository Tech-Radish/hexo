---
title: Hexo多电脑同步写作
date: 2018-03-22 21:01:20
categories:
- 技术文章
- 博客搭建
tags:
- hexo
- 多电脑同步
---
利用Hexo安装完博客之后，如何实现多电脑写作捏？下面分几步来说明

# 1.上传文件到仓库 #
前提是已经安装好Git客户端。这个应该在你安装博客的时候就已经安装好了吧。不会的话，百度下下载链接安装就好。
首先你要明白，你创建的博客通过`hexo d`命令部署到github之后,和你的本地博客根目录下的`.deploy_git`文件夹中的目录结构是一样的，所以，这只能算是个web工程，若要想实现多客户端写作的话，需要通过下面的步骤。

<!-- more -->

1. 首先在你Github账户上新建一个仓库，例如名为`hexo-blog`
2. 将本地博客根目录下的5个文件分别copy到一个新文件夹(例：hexo-blog)里面。
	1. scaffolds
	2. source
	3. themes（记得删除你下载主题的.git目录，它通常是隐藏的，需要取消隐藏之后删除，或者使用Git客户端来删除，`ls -a && rm .git`）
	4. _config.yml
	5. package.json
3.  在hexo-blog目录中执行
	1.  `git init`
	2.  `git add .`
	3.  `git remote add origin git@github@你github用户名/hexo-blog(换成你仓库名).github.io.git`（使用你新建的仓库的SSH地址）
	4.  `git commit -m 'blog source bakcup'`(commit之后才能创建分支)
	5.  `git branch hexo`创建一个hexo分支
	6.  `git checkout hexo`切换到hexo分支
	5.  `git push origin hexo`
4. 在第2步中，起始可以直接执行第3步的命令，也即可以不用复制那5个文件到新的目录中，只是因为那5个目录是必须的，其他的都是次要的。

# 2. 下载文件 #
上一步已经将你本地的博客托管到了github仓库中，接下来需要在你另一台需要写博客的电脑中，安装Node.js（这个自行百度吧，直接next安装即可）然后执行clone命令即可。

1. 进入到你放置博客的目录中，然后执行`git clone -b hexo git@github@你github用户名/hexo-blog(换成你仓库名).github.io.git`
2. `cd hexo-blog`进入此仓库目录中
3. 执行`npm install`安装所需组件
4. 使用`hexo g && hexo s -p 8080` 在本地打开浏览器输入`localhost:8080` 查看与在线的博客是否一致。
5. 使用`hexo new "page name"`新建一片博客，写完一篇博客，然后部署`hexo clean && hexo g && hexo d`,再执行以下命令来完成同步
	1. `git add .`
	2. `git commit -m 'add a new page'`
	3. `git push origin hexo`
6. 此时就可以在你原先电脑上执行`git pull origin hexo`来完成同步了。 

# 留言 #
如果还有不懂请在下面留言，我会及时回复。