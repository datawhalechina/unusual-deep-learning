**生成对抗网络 Generative Adversarial Networks**

# GAN产生背景

机器学习方法有两种，**生成方法**和**判别方法**。

- 生成方法，所学到到模型称为生成式模型
  - 生成方法通过观测数据学习样本与标签的联合概率分布P(X,Y)，训练好的模型，即生成模型，能够生成符合样本分布的新数据；
  - 生成式模型在无监督深度学习方面占据主要位置，可以用于在没有目标类标签信息的情况下捕捉观测到或可见数据的高阶相关性 – 判别方法，所学到的模型称为判别式模型。

- 判别方法由数据直接学习决策函数f(X)或者条件概率分布P(Y|X)作为预 测的模型，即判别模型
  - 判别模型经常用在有监督学习方面；
  - 判别方法关心的是对给定的输入X，应该预测什么样的输出Y。

有监督学习经常比无监督学习获得更好的模型，但是有监督学习需 要大量的标注数据，从长远看无监督学习更有发展前景。但是支持无监督学习的生成式模型会遇到下面两大困难：

- 首先是人们需要大量的先验知识去对真实世界进行建模，而建模的好坏直接影响着我们的生成模型的表现；
- 真实世界的数据往往很复杂，人们要用来拟合模型的计算量往往非常庞大，甚至难以承受。

# GAN的概况

**GAN的提出**

2014年，生成对抗网络(Generative Adversarial Networks， GAN)由当时还在蒙特利尔读博士的Ian Goodfellow(导师Bengio)提出。

- 2016年，GAN热潮席卷AI领域顶级会议，从ICLR到NIPS，大量高质量论文被发表和探讨
- 2017年入选MIT评论35岁以下创新人物

**GAN的基本原理**

GAN起源于博弈论中的二人零和博弈(获胜1，失败-1)

- 由两个互为敌手的模型组成
  - 生成模型(假币制造者团队) 
  - 判别模型(警察团队)
- 竞争使得两个团队不断改进他们的方法直到无法区分假币与真币

<img src="./PIC/GAN/1.png" alt="1" style="zoom:50%;" />

参考：Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, Yoshua Bengio. Generative Adversarial Nets. NIPS 2014: 2672-2680

**GAN的评价**

Yann LeCun评价GAN：

- 我们一直在错过一个关键因素就是无监督/预 测学习，这是指：机器给真实环境建模、预测 可能的未来、并通过观察和演示来理解世界是如何运行的能力。
- GAN为创建无监督学习提供了强有力的算法 框架，有望帮助我们为AI加入常识，我们认为， 沿着这条路走下去，有不小的成功的机会能开发出更智慧的AI。

**The GAN Zoo**

- https://github.com/hindupuravinash/the-gan-zoo
- ABC-GAN, AC-GAN, ...
- BAGAN, BCGAN...
- C-GAN,CA-GAN,...
- ...α-GAN, β-GAN,...

<img src="./PIC/GAN/2.png" alt="2" style="zoom:50%;" />

# GAN模型

###  生成模型中的问题

### GAN模型示例

### GAN目标函数

### GAN损失函数

### GAN模型训练

### 优势和不足

**优势**

- 任何一个可微分函数都可以参数化D和G(如深度神经网络)
- 支持无监督方法实现数据生成，减少了数据标注工作
- 生成模型G的参数更新不是来自于数据样本本身(不是对数据的似然性进行优化)，而是来自于判别模型D的一个反传梯度

**不足**

- 无需预先建模，数据生成的自由度太大
- 得到的是概率分布，但是没有表达式，可解释性差
- D与G训练无法同步，训练难度大，会产生梯度消失问题

# GAN的优化和改进

### 限定条件优化

- CGAN:ConditionalGenerativeAdversarialNets
- Generative Adversarial Text to Image Synthesis
- InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
- Improved Techniques for Training GANs
- GP-GAN: Towards Realistic High-Resolution Image Blending

**CGAN**







### 迭代式生成优化

- –  LAPGAN:Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks
- –  StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks
- –  PPGN: “Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space

### 结构优化

- DCGAN:Unsupervised Representation Learning with Deep

  Convolutional Generative Adversarial Networks

- Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks













