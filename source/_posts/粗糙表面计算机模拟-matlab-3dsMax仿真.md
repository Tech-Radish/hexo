---
title: 粗糙表面计算机模拟(matlab+3dsMax仿真)
date: 2018-03-23 18:29:30
categories:
- 研磨&粗糙度
- 研磨表面仿真
tags: 
- 表面粗糙度
- matlab模拟粗糙度
- 3dsMax仿真
---
本文主要使用matlab对零件表面的粗糙度进行模拟[1]，然后生成模型，导入到3dsMax中进行仿真.
采用matlab实现具有高斯分布粗糙表面的模拟，参考胡元中的论文。
# 代码 #
  
	clear
	clc
	
	N=128;%生成大小
	delta=0.05;%表面均方根粗糙度
	betax=30;%x方向的相关长度
	betay=30;%y方向的相关长度
	C=1;%功率谱密度
	
	L=0.05;
	dx=L/N;dy=dx;
	NN=-N/2:N/2-1;
	[Nx,Ny]=meshgrid(NN,NN);
	taux=dx.*Nx;tauy=dy.*Ny;
	
	%%生成具有指定自相关函数的粗糙表面
	eta=randn(N,N);%高斯分布白噪声
	A=fft2(eta);%傅里叶变换
	R=zeros(N,N);
	R=delta^2*exp(-2.3*((taux/betax).^2+(tauy/betay).^2).^0.5);%自相关函数
	Gz=1/(2*pi^2).*fft2(R);%功率谱密度函数
	H=(Gz/C).^0.5;%传递函数
	Z=H.*A;%表面高度的傅里叶变换
	z=ifft2(Z);%表面高度分布
	z = abs(z) * 1800;
	figure(1);
	mesh(z);
	% surf2stl('surf_roughness.stl',1,1,z) % 生成模型
	title('rough surface');
	axis square
	
# 实验结果 #
![matlab仿真结果128*128](http://mic-jasontang.github.io/imgs/surface_roughness.png)
![matlab仿真结果256*256](http://mic-jasontang.github.io/imgs/surface_roughness_256.png)
# 仿真结果 #
采用目标聚光灯和目标相机，相机采用50mm焦距拍摄，模拟效果还算理想。

# 参考文献 #
[1]陈辉,胡元中,王慧,王文中.粗糙表面计算机模拟[J].润滑与密封,2006(10):52-55+59.
