---
title: 关于k阶矩的理解
date: 2018-03-22 21:44:18
categories:
- 数学基础
- 随机过程
tags:
- k阶矩
- 随机过程
- 偏度
- 峰度
---

# k阶原点矩、2阶矩、3阶矩该怎么理解？ #

下面使用语言描述和代码来讲解。
<!-- more -->

> 阶矩是用来描述随机变量的概率分布的特性.

> 一阶矩指的是随机变量的平均值,即期望值
> 
> 二阶矩指的是随机变量的方差
> 
> 三阶矩指的是随机变量的偏度
> 
> 四阶矩指的是随机变量的峰度
 
> 因此通过计算矩,则可以得出随机变量的分布形状

# 代码实现 #
使用Python2.0实现

    import numpy as np
	from scipy import stats
	
	
	def calc_statistics(x):
	n = x.shape[0]   #样本个数

	# 手动计算
	m = 0
	m2 = 0
	m3 = 0
	m4 = 0
	for t in x:
		m += t
		m2 += t*t
		m3 += t**3
		m4 += t**4
	m /= n
	m2 /= n
	m3 /= n
	m4 /= n

	mu = m    # 一阶矩
	sigma = np.sqrt(m2 - mu*mu)   # 二阶矩
	skew = (m3 - 3*mu*m2 + 2*mu**3) / sigma**3    # 三阶矩（偏度）
	kurtosis = (m4 - 4*mu*m3 + 6*mu*mu*m2 - 4*mu**3*mu + mu**4) / sigma**4 - 3	# 四阶矩（峰度）
	print "手动计算均值、标准差、偏度、峰度：", mu, sigma, skew, kurtosis

	# 使用系统函数验证
	mu = np.mean(x, axis=0)
	sigma = np.std(x, axis=0)
	skew = stats.skew(x)
	kurtosis = stats.kurtosis(x)
	return mu, sigma, skew, kurtosis

	if __name__ == '__main__':
	d = np.random.randn(10000)
	print d
	print d.shape
	mu, sigma, skew, kurtosis = calc_statistics(d)
	print "函数库计算均值、标准差、偏度、峰度：", mu, sigma, skew, kurtosis
	
执行结果:
	
	
> [-0.42751577  0.36230961  0.37899409 ...,  0.09176115 -1.38955563
 -0.57570736]


> (10000L,)


> 手动计算均值、标准差、偏度、峰度： -0.00189350820374 0.995018151945 -0.00589521484127 -0.0590604043446


> 函数库计算均值、标准差、偏度、峰度： -0.00189350820374 0.995018151945 -0.00589521484127 -0.0590604043446
