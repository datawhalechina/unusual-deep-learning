# 深度生成模型

​		深度学习不仅在于其强大的学习能力，更在于它的创新能力。我们通过构建**判别模型**来**提升模型的学习能力**，通过构建**生成模型**来**发挥其创新能力**。
​		判别模型通常利用训练样本训练模型，然后利用该模型，对新样本x，进行判别或预测。而生成模型正好反过来， 根据一些规则y，来生成新样本x。 
​		生成式模型很多，本章主要介绍最常用生成式对抗网络（GAN）及其变种。GAN是基于博弈论，目的是找到达到纳什均衡的判别器网络和生成器网络。

## GAN简介

​		**生成对抗网络**（Generative Adversarial Networks，GAN）*[Goodfellow et al.,2014*] 是通过对抗训练的方式来使得生成网络产生的样本服从真实数据分布。在生成对抗网络中，有两个网络进行对抗训练。一个是**判别网络**，目标是尽量准确地判断一个样本是来自于真实数据还是由生成网络产生；另一个是**生成网络**，目标是尽量生成判别网络无法区分来源的样本。这两个目标相反的网络不断地进行交替训练。 当最后收敛时，如果判别网络再也无法判断出一个样本的来源，那么也就等价于生成网络可以生成符合真实数据分布的样本。生成对抗网络的流程图如下所示。

![](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/GAN.png)

​		GAN的直观理解，可以想象有一位莫奈风格的名画伪造者和一位莫奈风格的艺术鉴定师。一开始，伪造者是个刚入门的小白，只会在画布上画出混乱的颜色，之后他将自己的一些作品和莫奈风格的真品混在一起，请艺术鉴定师进行真实性评估，艺术鉴定师通过真实的数据集学习，一开始很容易鉴别出了赝品，并向伪造者反馈告诉他哪些看起来像真迹、哪些看起来不想真迹。

​		伪造者根据这些反馈，改进自己的赝品。随着时间的推移，伪造者技能越来越高，艺术商人也变得越来越擅长找出赝品。最后，他们手上就拥有了一些非常逼真的赝品。

因此，GAN从网络的角度来看，它由**两部分**组成。 

- **生成器网络**：它一个潜在空间的随机向量作为输入，并将其解码为一张合成图像。

- **判别器网络**：以一张图像（真实的或合成的均可）作为输入，并预测该图像来自训练集还是来自生成器网络。

  ![](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-2021042814164321110.png)

## GAN的损失函数

​		从GAN的架构图可知，控制生成器或判别器的关键是损失函数，而如何定义损失函数就成为整个GAN的关键。我们的目标很明确，既要不断提升判断器辨别是非或真假的能力，又要不断提升生成器不断提升图像质量，使判别器越来越难判别。那这些目标如何用程序体现？损失函数就能充分说明。 

### 判别网络

​		判别网络（Discriminator Network）$$𝐷(𝒙; \phi)$$ 的目标是区分出一个样本 $$𝒙$$是来自于真实分布 $$𝑝_𝑟 (x)$$ 还是来自于生成模型 $$𝑝_\theta (x)$$，因此判别网络实际上是一个二分类的分类器。用标签𝑦 = 1来表示样本来自真实分布，𝑦 = 0表示样本来自生成模型，判别网络$$𝐷(𝒙; \phi)$$​的输出为：𝒙属于真实数据分布的概率，即
$$
\begin{array}{l}
p(y=1 \mid \boldsymbol{x})=D(\boldsymbol{x} ; \phi)
\end{array}\tag {1}
$$
​		则样本来自生成模型的概率为$$p(𝑦 = 0|𝒙) = 1 − 𝐷(𝒙; \phi)$$

​		给定一个样本$$ (x, 𝑦),𝑦 = \{1, 0\}$$表示其来自于 $$𝑝_𝑟 (x)$$ 还是 $$𝑝_\theta (x)$$，判别网络的目标函数为最小化交叉熵，即

$$
\min _{\phi}-\left(\mathbb{E}_{\boldsymbol{x}}[y \log p(y=1 \mid \boldsymbol{x})+(1-y) \log p(y=0 \mid \boldsymbol{x})]\right)
\tag {2}
$$
​		假设分布 𝑝(𝒙) 是由分布 $p_𝑟(𝒙)$ 和分布 $p_𝜃(𝒙)$ 等比例混合而成，即 $p(𝒙) =
\frac{1}{2}\left(p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})\right) $，则上式等价于
$$
\max _{\phi} \mathbb{E}_{\boldsymbol{x} \sim p_{r}(\boldsymbol{x})}[\log D(\boldsymbol{x} ; \phi)]+\mathbb{E}_{\boldsymbol{x}^{\prime} \sim p_{\theta}\left(\boldsymbol{x}^{\prime}\right)}\left[\log \left(1-D\left(\boldsymbol{x}^{\prime} ; \phi\right)\right)\right] \tag{3}
$$
$$
=\max _{\phi} \mathbb{E}_{\boldsymbol{x} \sim p_{r}(\boldsymbol{x})}[\log D(\boldsymbol{x} ; \phi)]+\mathbb{E}_{\boldsymbol{z} \sim p(z)}[\log (1-D(G(\boldsymbol{z} ; \theta) ; \phi))] \tag{4}
$$
​		其中 $\theta$ 和 $\phi$ 分别是**生成网络**和**判别网络**的参数。$P(z)$是低维空间 𝒵 中的一个简单容易采样的分布，$P(z)$通常为标准多元正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{1}) $。

### 生成网络
​		生成网络（Generator Network）的目标刚好和判别网络相反，即让判别网络将自己生成的样本判别为真实样本。
$$
\begin{aligned}
& \max _{\theta}\left(\mathbb{E}_{\boldsymbol{z} \sim p(z)}[\log D(G(\boldsymbol{z} ; \theta) ; \phi)]\right) \\
=& \min _{\theta}\left(\mathbb{E}_{\boldsymbol{z} \sim p(z)}[\log (1-D(G(\boldsymbol{z} ; \theta) ; \phi))]\right)
\end{aligned}
$$
​		上面的这两个目标函数是等价的. 但是在实际训练时，一般使用前者，因为其梯度性质更好。我们知道，函数$log(𝑥)$, 𝑥 ∈ (0, 1)在𝑥 接近1时的梯度要比接近0时的梯度小很多，接近“饱和”区间。 这样，当判别网络𝐷以很高的概率认为生成网络$𝐺$产生的样本是“假”样本，即$(1 − 𝐷(𝐺(𝒛; \theta); \phi)) → 1$，这时目标函数关于𝜃 的梯度反而很小，从而不利于优化。

​		而一开始判别器是很容易鉴别仿造数据的，因此$𝐷(𝐺(𝒛;\theta);\phi)$的初始值是在靠近 0 的左端。而对于刚开始训练的模型，我们希望在初期$𝐷(𝐺(𝒛;\theta);\phi)$能够快速地更新，但不幸的是，目标函数$log(1 − D(x))$左端刚好是平缓的区域，依据梯度下降原理这会阻碍$D(x)$的快速更新。

**Tips:**

> ​	为了解决这一问题，有人提出了把$log(1 − D(x))$这个表达式换成$−logD(x)$，同样能满足判别器的目标函数要求，并且在训练初期还能更新得比较快。上述方法便是在这个非常小的地方做了改进。
>
> ​	不过后来，人们为了区分这两种 GAN，还是分别起了不同的名字。第一种 GAN 被叫做MMGAN（Minimax GAN），它也是人们常说的原始 GANs；第二种 GAN 被叫做 NSGAN（Non-saturating GAN）。

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-20210429122450146.png" alt="image-20210429122450146" style="zoom: 50%;" />

### 最小化最大化游戏

​		将判别网络和生成网络合并，整个生成对抗网络的目标函数看作是**最小化最大化游戏（Minimax Game）**。
$$
\begin{aligned}
& \min _{\theta} \max _{\phi}\left(\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}[\log D(\boldsymbol{x} ; \phi)]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}[\log (1-D(\boldsymbol{x} ; \phi))]\right) \\
=& \min _{\theta} \max _{\phi}\left(\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}[\log D(\boldsymbol{x} ; \phi)]+\mathbb{E}_{z \sim p(z)}[\log (1-D(G(\boldsymbol{z} ; \theta) ; \phi))]\right) \\
=& \min _{\theta} \max _{\phi}\left(\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}[\log D(\boldsymbol{x} ; \phi)]-\mathbb{E}_{z \sim p(z)}[\log (D(G(\boldsymbol{z} ; \theta) ; \phi))]\right)_{Non-saturating}
\end{aligned} \tag{5}
$$

​		但是如果判断器的能力过于好，$$D(G(\boldsymbol{z} ; \theta)$$趋近于0时，会导致max的值趋近一个常数。这时即使采取目标函数关于𝜃 的梯度变化较大的$\max _{\theta}\left(\mathbb{E}_{\boldsymbol{z} \sim p(z)}[\log D(G(\boldsymbol{z} ; \theta) ; \phi)]\right)$的损失函数，由于最优的判别器$D^{\star}$对所有生成的数据的输出都为0。因此生成网络的梯度消失。

---
**D表示判别器、G为生成器、real_labels、fake_labels分别表示真图像标签、假图像标签。images是** 

**真图像，z是从潜在空间随机采样的向量，通过生成器得到假图像。** 

```
# 定义判断器对真图像的损失函数 
outputs = D(images) 
d_loss_real = criterion(outputs, real_labels) 
real_score = outputs 

# 定义判别器对假图像（即由潜在空间点生成的图像）的损失函数 
z = torch.randn(batch_size, latent_size).to(device) 
fake_images = G(z) 
outputs = D(fake_images) 
d_loss_fake = criterion(outputs, fake_labels) 
fake_score = outputs 

# 得到判别器总的损失函数 
d_loss = d_loss_real + d_loss_fake 
```

生成器的损失函数如何定义，才能使其越来越向真图像靠近？以真图像为标杆或标签即可。具体代码如下： 

```
# 定义p(Z)是一个高斯分布
z = torch.randn(batch_size, latent_size).to(device) 

# 进行图片生成和判别
fake_images = G(z) 
outputs = D(fake_images)

# 得到生成器总的损失函数 
g_loss = criterion(outputs, real_labels)
```

---


## 模型训练

​		和单目标的优化任务相比，生成对抗网络的两个网络的优化目标刚好相反。因此生成对抗网络的训练比较难，往往不太稳定。 一般情况下，需要平衡两个网络的能力。对于判别网络来说，**一开始的判别能力不能太强**，否则难以提升生成网络的能力。但是，**判别网络的判别能力也不能太弱**，否则针对它训练的生成网络也不会太好。在训练时需要使用一些技巧，使得在每次迭代中，**判别网络比生成网络的能力强一些**，但又不能强太多。

​		而生成网络更新一次生成对抗网络的训练流程如下算法所示。每次迭代时，**判别网络更新 𝐾 次** ，即首先要保证判别网络足够强才能开始训练生成网络。在实践中**𝐾** 是一个超参数，其取值一般取决于具体任务。
![](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/algothorim.png)

## 手写数字生成——GAN

​		为便于说明GAN的关键环节，这里我们弱化了网络和数据集的复杂度。数据集为 MNIST、网络用全连接层。后续将用一些卷积层的实例来说明。

### 导入相关库

首先导入numpy、torch等模块。

```
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
```

### 数据加载和参数定义

pytorch内置集成了MNIST数据集。

```
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
latent_size = 64  
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'samples'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5,std=0.5)])

# MNIST dataset
mnist = torchvision.datasets.MNIST('./data',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)
```

### 判别器模型

定义判别器网络结构，这里使用LeakyReLU为激活函数，输出一个节点并经过 Sigmoid后输出，用于真假二分类。 
```
# 构建判断器 
D = nn.Sequential( 
	nn.Linear(image_size, hidden_size), 
	nn.LeakyReLU(0.2), 
	nn.Linear(hidden_size, hidden_size), 
	nn.LeakyReLU(0.2), 
	nn.Linear(hidden_size, 1), 
	nn.Sigmoid())

D = D.to(device)
```
### 生成器模型

使用nn.tanh将使数据分布 在[-1,1]之间。其输入是潜在空间的向量z，输出维度与真图像相同。

```
# 构建生成器
G = nn.Sequential(
	nn.Linear(latent_size, hidden_size), 
	nn.ReLU(), 
	nn.Linear(hidden_size, hidden_size), 
	nn.ReLU(), 
	nn.Linear(hidden_size, image_size), 
	nn.Tanh())

G = G.to(device)
```

### 训练模型

```
# 定义损失函数以及优化器
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
    
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
  for i, (images, _) in enumerate(data_loader):
    images = images.reshape(batch_size, -1).to(device) 
		
		# 定义图像是真或假的标签 			
		real_labels = torch.ones(batch_size, 1).to(device) 
		fake_labels = torch.zeros(batch_size, 1).to(device) 
    # ================================================================== # 
    #                          训练判别器                                 # 
    # ================================================================== # 
    # 定义判别器对真图像的损失函数
    outputs = D(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs 

    # 定义判别器对假图像（即由潜在空间点生成的图像）的损失函数 
    z = torch.randn(batch_size, latent_size).to(device) 
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # 得到判别器总的损失函数
    d_loss = d_loss_real + d_loss_fake 

    # 对生成器、判别器的梯度清零
    # 进行反向传播及运行判别器的优化器 
    reset_grad()
    d_loss.backward()
    d_optimizer.step() 
    # ================================================================== # 
    #                           训练生成器                                # 
    # ================================================================== # 
    # 定义生成器对假图像的损失函数
    z = torch.randn(batch_size, latent_size).to(device) 
    fake_images = G(z)
    outputs = D(fake_images)

    g_loss = criterion(outputs, real_labels) 

    # 对生成器、判别器的梯度清零 
    # 进行反向传播及运行生成器的优化器 
    reset_grad()
    g_loss.backward()
    g_optimizer.step()

    if (i+1) % 200 == 0:
      print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))

    # 保存真图像
    if (epoch+1) == 1:
      images = images.reshape(images.size(0), 1, 28, 28)
      save_image(denorm(images), os.path.join(sample_dir, 'real_images.png')) 
      
    # 保存假图像 
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28) save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1))) 

    # 保存模型
    torch.save(G.state_dict(), 'G.ckpt')
    torch.save(D.state_dict(), 'D.ckpt')
```

### 可视化结果

```
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

reconsPath = './samples/fake_images-200.png' 
Image = mpimg.imread(reconsPath) 
plt.imshow(Image)
plt.axis('off')
plt.show() 
```

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-20210428223051656.png" alt="image-20210428223051656" style="zoom:50%;" />

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430234429.png" alt="手写数字图片" style="zoom: 50%;" />

## 模型分析

​		还记得之前提到的将整个生成对抗网络的目标函数看作是**最小化最大化游戏（Minimax Game）**。
$$
\begin{aligned}
& \min _{\theta} \max _{\phi}\left(\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}[\log D(\boldsymbol{x} ; \phi)]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}[\log (1-D(\boldsymbol{x} ; \phi))]\right) \\
=& \min _{\theta} \max _{\phi}\left(\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}[\log D(\boldsymbol{x} ; \phi)]+\mathbb{E}_{z \sim p(z)}[\log (1-D(G(\boldsymbol{z} ; \theta) ; \phi))]\right)
\end{aligned} \tag{6}
$$
​		由于的生成网络梯度问题，这个最小化最大化形式的目标函数一般用来进行理论分析，并不是实际训练时的目标函数。

​		对于判别器模型，它的min**损失函数**为：
$$
\mathcal{L}(f)=\mathbb{E}_{\boldsymbol{x} \sim p\left(\boldsymbol{x} \mid c_{1}\right)}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{x} \sim p\left(\boldsymbol{x} \mid c_{2}\right)}[\log (1-D(\boldsymbol{x}))] \tag{7}
$$
​		假设$p_{r}(\boldsymbol{x})$和$p_{\theta}(\boldsymbol{x})$已知，通过数学推导，可以得到最优的判别器为
$$
D^{\star}(\boldsymbol{x})=\frac{p_{r}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})} \tag{8}
$$

​		将此时的$D^{\star}(x)$带入损失函数中，其目标函数变为
$$
\begin{aligned}
\mathcal{L}\left(G \mid D^{\star}\right) &=\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}\left[\log D^{\star}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \left(1-D^{\star}(\boldsymbol{x})\right)\right] \\
&=\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}\left[\log \frac{p_{r}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \frac{p_{\theta}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})}\right] \\
&=\mathrm{KL}\left(p_{r}, p_{a}\right)+\mathrm{KL}\left(p_{\theta}, p_{a}\right)-2 \log 2 \\
&=2 \mathrm{JS}\left(p_{r}, p_{\theta}\right)-2 \log 2
\end{aligned} \tag{9}
$$

其中$\mathrm{JS}(\cdot)$ 为 $\mathrm{JS}$ 散度, $p_{a}(\boldsymbol{x})=\frac{1}{2}\left(p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})\right)$ 为一个“平均”分布。

在生成对抗网络中，当判别网络为最优时，生成网络的优化目标是最小化真实分布$p_r$和模型分布$p_{\theta}$之间的$JS$散度。当两个分布相同时，$JS$散度为0，最优生成网络$G^{\star}$对应的损失为$\mathcal{L}\left(G^{\star} \mid D^{\star}\right)=−2log2$。

### 训练稳定性

​		使用 $JS$ 散度来训练生成对抗网络的一个问题是当两个分布没有重叠时，它们之间的$JS$散度恒等于常数$log 2$。对生成网络来说，目标函数关于参数的梯度为0，即$\frac{\partial \mathcal{L}\left(G \mid D^{\star}\right)}{\partial \theta}=0$。

​		当真实分布 $p_r $和模型分布 $p_{\theta} $没有重叠时，最优的判别器$D^{\star}$对所有生成的数据的输出都为0，而从导致生成网络的梯度消失。

​		因此，在实际训练生成对抗网络时，**一般不会将判别网络训练到最优**，只进行**一步或多步梯度**下降，使得生成网络的梯度依然存在。另外，判别网络也不能太差，否则生成网络的梯度为错误的梯度。但是，如何在梯度消失和梯度错误之间取得平衡并不是一件容易的事，这个问题使得生成对抗网络在训练时稳定性比较差。

![image-20210428204439166](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-20210428204439166.png)

### 模型坍塌（mode collapse）

对于生成器的另一种奖励形式的目标函数，将$G^{\star}$带入得到：
$$
\max _{\theta}\left(\mathbb{E}_{\boldsymbol{z} \sim p(z)}[\log D(G(\boldsymbol{z} ; \theta) ; \phi)]\right) \tag{10}
$$

$$
\begin{array}{l}
\mathcal{L}^{\prime}\left(G \mid D^{\star}\right)=\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log D^{\star}(\boldsymbol{x})\right] \\
\quad=\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \frac{p_{r}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})} \cdot \frac{p_{\theta}(\boldsymbol{x})}{p_{\theta}(\boldsymbol{x})}\right] \\
=-\mathbb{E}_{x \sim p_{\theta}(x)}\left[\log \frac{p_{\theta}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \frac{p_{\theta}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})}\right] \\
=-\mathrm{KL}\left(p_{\theta}, p_{r}\right)+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \left(1-D^{\star}(\boldsymbol{x})\right)\right] \\
=-\mathrm{KL}\left(p_{\theta}, p_{r}\right)+2 \operatorname{JS}\left(p_{r}, p_{\theta}\right)-2 \log 2-\mathbb{E}_{x \sim p_{r}(x)}\left[\log D^{\star}(\boldsymbol{x})\right]
\end{array} \tag{11}
$$

其中后两项和生成网络无关，因此：
$$
\underset{\theta}{\arg \max } \mathcal{L}^{\prime}\left(G \mid D^{\star}\right)=\underset{\theta}{\arg \min } \mathrm{KL}\left(p_{\theta}, p_{r}\right)-2 \mathrm{JS}\left(p_{r}, p_{\theta}\right) \tag{12}
$$
其中$JS$散度$JS(𝑝𝜃, 𝑝𝑟 ) ∈ [0, log 2]$为有界函数，因此生成网络的目标更多的是受逆向KL散度$KL(p_{\theta},p_r)$影响，使得生成网络更倾向于生成一些更“安全”的样本，从而造成模型坍塌（Model Collapse）问题。

下图给出数据真实分布为一个高斯混合分布，模型分布为一个单高斯分布时，使用前向和逆向 KL 散度来进行模型优化的示例。黑色曲线为真实分布$ 𝑝_𝑟$的等高线，红色曲线为模型分布$𝑝_{\theta}$的等高线.

![image-20210428205953349](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-20210428205953349.png)

- 在前向KL散度会鼓励模型分布$p_{\theta}(𝒙)$尽可能覆盖所有真实分布$p_r(𝒙)>0$的点，而不用回避$p_r(𝒙)≈0$的点；
- 逆向KL散度会鼓励模型分布$p_{\theta}(𝒙)$尽可能避开所有真实分布$p_r(𝒙)≈0$的点，而不需要考虑是否覆盖所有真实分布$p_r(𝒙)>0$的点。

**一个比较直观的演示：**

![lihungyi](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430234538.png)

​		可以看到在**生成网络**生成的图中，有一种类的图片重复出现了多次，只是变换了头发的颜色，但整体极其相似。这就是模型崩塌的典型的例子。	

​		在生成对抗网络中，JS 散度不适合衡量生成数据分布和真实数据分布的距离。由于通过优化交叉熵（JS散度）训练生成对抗网络会导致训练稳定性和模型坍塌问题，因此要改进生成对抗网络，就需要改变其损失函数。比如**W-GAN**用***Wasserstein***距离替代 JS 散度来优化训练的生成对抗网络等等。

## W-GAN

​		W-GAN 的全称是 **WassersteinGAN**，它提出了用 Wasserstein 距离（也称 EM 距离）去取代JS 距离，这样能更好的衡量两个分布之间的 Div。

​		对于真实分布$p_r$和模型分布$p_\theta$，它们的1st-Wasserstein距离为
$$
\boldsymbol{V}^{1}\left(p_{r}, p_{\theta}\right)=\inf _{\gamma \sim \Gamma\left(p_{r}, p_{\theta}\right)} \mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}) \sim \gamma}[\|\boldsymbol{x}-\boldsymbol{y}\|]
$$
​		其中$\Gamma\left(p_{r}, p_{\theta}\right)$是边际分布为$p_r$和$p_\theta$的所有可能的联合分布集合。当两个分布没有重叠或者重叠非常少时，它们之间的 KL 散度为$+\infty$，JS 散度为log 2，并不随着两个分布之间的距离而变化。而1st-Wasserstein距离依然可以衡量两个没有重叠分布之间的距离。

​		下面我们直接给出 WGAN 的判别器的目标表达式：
$$
V(G, D)=\max _{D \in 1-L i p s c h i t z}\left\{E_{x \sim P_{d a t a}}[D(x)]-E_{x \sim P_{G}}[D(x)]\right\}
$$

​		这个表达式的求解结果就是$𝑃_𝐺$与$𝑃_{𝑑𝑎𝑡𝑎}$之间的 $Wasserstein$ 距离。至于为什么会等于$Wasserstein$距离，详细证明请参阅 WGAN paper 附录当中的证明部分，因为过于繁琐，在此就不赘述。

​		关于这个表达式，值得注意的是，D 被加上了 1-Lipschitz function（如下图）的限制。
$$
\begin{array}{l}
\left\|f\left(x_{1}\right)-f\left(x_{2}\right)\right\| \leq K\left\|x_{1}-x_{2}\right\| \\
\text { K=1 for "1 - Lipschitz" }
\end{array} \tag{3}
$$

> ​												 **数学小知识**
> ​   在数学中, 对于一个实数函数 $f: \mathbb{R} \rightarrow \mathbb{R}$​, 如果满足函数曲线上任 意两点连线的斜率一致有界, 即任意两点的斜率都小于常数 $K>0$​,
> $$
> \left|f\left(x_{1}\right)-f\left(x_{2}\right)\right| \leq K\left|x_{1}-x_{2}\right|
> $$
>    则函数 $f$​ 就称为 $K$​ -Lipschitz连续函数, $K$​ 称为 Lipschitz 常数. Lipschitz 连续要求函数在无限的区间上不能有超过线性的增长. 如果一个函数可导，并满足 Lipschitz 连续，那么导数有界. 如果一个函数可导,并且导数有界,那么函数为 Lipschitz 连续.

​		先说明一下，为什么要对判别器做限制。传统 GANs 的判别器输出的结果是在(0,1)区间之内，但是在 WGAN 中输出的结果是 was 距离，was 距离是没有上下界的，这意味着，随着训练进行，$P_G$的 was 值会越来越小，$𝑃_{𝑑𝑎𝑡𝑎}$的 was 值会越来越大，判别器将永远无法收敛。

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430195420.png" alt="infinte" style="zoom:67%;" />

​		因此，为了解决这个问题，我们需要给判别器加上一些限制，让$P_G$不会持续地一直降低，让$P_{data}$也不会持续地一直升高，简言之，就是让 D 函数变得更平滑一些。但是我们知道，一般的神经网络的训练，参数都是没有限制的，而现在我们希望给判别器的参数增加一些限制，其实是不太好做的。

​		在最原始的 WGAN 中，采用的做法是 weight clipping，很简单，设定一个上限 c 与下限 -c ，如果更新参数 $w>c$，改成 $w=c$；如果更新参数 $w<-c$，改成 $w=-c$。这样 D 在$P_G$与$𝑃_{𝑑𝑎𝑡𝑎}$处的值就不会被无限拉远。但是这个方法并没有让 D 真的限制在 1-Lipschitz function 内，所以原始的 WGAN 并没有严格地给出 was 距离计算方法。

​		直到 WGAN 的增强版 WGAN-GP，以及 SNGAN 被提出，才解决了这个问题。

​		由于was 距离计算方法过于复杂，在这里我们就做一个简单的介绍。



## WGAN-GP

​		WGAN 存在的问题是，没有能够将 D 真正的限制在 1-Lipschitz function 内。由于 1-Lipschitz function可以等价于如下的表达式：
$$
D \in 1-\text { Lipschitz } \Leftrightarrow\left\|\nabla_{x} D(x)\right\| \leq 1 \text { for all } x
$$
​		即对于任意的 x，对于一个可微函数，当且仅当$D(x)$对于x的梯度的模都小于或等于 1，则该可微函数是 1-Lipschitz function。那现在我们对判别器的目标表达式增添一个条件：
$$
\begin{array}{r}
V(G, D) \approx \max _{D}\left\{E_{x \sim P_{\text {data }}}[D(x)]-E_{x \sim P_{G}}[D(x)]\right. \\
\left.-\lambda \int_{x} \max \left(0,\left\|\nabla_{x} D(x)\right\|-1\right) d x\right\}
\end{array} \tag{5}
$$
​		对于式子的第三项，实际上统计的就是所有梯度的模不满足小于或等于 1 的项，并赋予惩罚参数$\lambda$进行累加，它会拖累$\max$的取值，相当于增加了一项使模型优化变差的正则项。但是，这个增添的条件由于对所有 x 有效，会让**惩罚变得非常高**，可能会带来不必要的影响和计算开销。

​		事实上我们真正需要考虑的惩罚项，应该是对判别器产生实质影响的区域。考虑到整个WGAN 的目的是让$P_G$渐渐向$P_{data}$靠拢，那位于$P_G$和$P_{data}$之间的区域一定会对判别器产生质的影响。因此，我们将惩罚项中 x 的范围缩小为$P_{penalty}$，$P_{penalty}$是介于$P_G$和$P_{data}$之间的区域。目标表达式转化为如下式子：


$$
\begin{aligned}
V(G, D) \approx \max _{D}\left\{E_{x \sim P_{\text {data }}}[D(x)]-E_{x \sim P_{G}}[D(x)]\\
-\lambda E_{x \sim P_{\text {penalty }}}\left[\max \left(0,\left\|\nabla_{x} D(x)\right\|-1\right)\right]\right\}
\end{aligned}
$$
<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210501001329.png" alt="image-20210501001326919" style="zoom:67%;" />

​		一个有意思的一点是，在实验中作者发现在惩罚项中梯度越接近 1，训练得越快，效果也越好，于是表达式可以改为：
$$
\begin{aligned}
V(G, D) \approx \max _{D}\left\{E_{x \sim P_{\text {data }}}[D(x)]-E_{x \sim P_{G}}[D(x)]\\
-\lambda E_{x \sim P_{\text {penalty }}}\left[\left(\left\|\nabla_{x} D(x)\right\|-1\right)^{2}\right]\right\}
\end{aligned}
$$

### 梯度惩罚项的实现

```
# gradient penalty
alpha = torch.rand((self.batch_size, 1, 1, 1))
if self.gpu_mode:
    alpha = alpha.cuda()
# 进行插值得到真实图片区域和生成图片区域中间区域的值
x_hat = alpha * x_.data + (1 - alpha) * G_.data
x_hat.requires_grad = True

pred_hat = self.D(x_hat)
# 对D(X)求导
if self.gpu_mode:
    gradients = grad(
        outputs=pred_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(pred_hat.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
else:
    gradients = grad(
        outputs=pred_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(pred_hat.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
gradient_penalty = (
    self.lambda_
    * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
)

D_loss = D_real_loss + D_fake_loss + gradient_penalty
D_loss.backward()
self.D_optimizer.step()
```

## SNGAN

​		**SNGAN** 基于 **spectral normalization** 的思想，通过对W 矩阵归一化的方式，真正将梯度控制在了小于或等于 1 的范围内，使得产生的函数更平滑一些。SNGAN（频谱归一化 GAN）为了让正则化产生更明确地限制，提出了用谱范数标准化神经网络的参数矩阵 W，从而让神经网络的梯度被限制在一个范围内。

### 频谱范数正则化

​		在介绍SNGAN之前，我们先简单介绍一下什么是频谱范数正则化：

​		频谱范数正则化方法是 17 年 5 月提出来的，虽然最终的 SNGAN 没有完全采用这一方法，但是它借鉴了这个方法非常重要的思想。对于频谱范数正则化，我们可以简单理解为把传统 GANs 中的 loss 函数：
$$
\begin{aligned}
\underset{\Theta}{\operatorname{minimize}} \frac{1}{K} \sum_{i=1}^{K} L\left(f_{\Theta}\left(\boldsymbol{x}_{i}\right), \boldsymbol{y}_{i}\right)+\lambda \sum\left(w_{i}\right)^{2}
\end{aligned}
$$
​		其中的正则项替换成了谱范数：
$$
\begin{aligned}
\underset{\Theta}{\operatorname{minimize}} \frac{1}{K} \sum_{i=1}^{K} L\left(f_{\Theta}\left(\boldsymbol{x}_{i}\right), \boldsymbol{y}_{i}\right)+\frac{\lambda}{2} \sum_{\ell=1}^{L} \sigma\left(W^{\ell}\right)^{2}
\end{aligned}
$$

​		并且谱范数的计算利用了功率迭代的方法去近似。在这里我们就不对功率迭代的算法进行展开介绍，感兴趣的同学可以自行学习。

### SNGAN中的正则思想

​		频谱范数正则化固然有效，但是它不能保证把$f_{\Theta}$的梯度限制在一个确定的范围内，真正解决了这一问题，是直到 18 年 2 月才被提出的 SNGAN。

​		通常在神经网络中的每一层，先进行输入乘权重的线性运算，再将其送入激活函数，由于通常选用ReLU作为激活函数，ReLu激活函数可以用对角方阵D表示，如果$Wx$的第 i 维大于0，则D的第 i 个对角元素为1，否则为0，需要注意D的具体形式与W,x均有关系，但是D的最大奇异值必然是1。		

​		一般而言，即使神经网络的输出是非线性的，但是在x的一个足够小的邻域内，它一个表现为线性函数$Wx$，$W$的具体形式与$x$有关。真实的判别器$f(x)$的函数图像在比较小的尺度上来看应该是类似这种形式的分段函数：

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210501123156.png" alt="image-20210501123153753" style="zoom:67%;" />\

​		考虑到对于任意给定的x，均有：
$$
\begin{aligned}
\frac{\|f(x+\delta)-f(x)\|_{2}}{\|\delta\|_{2}}=\frac{\left\|W_{x} \delta\right\|_{2}}{\|\delta\|_{2}} \leq \sigma\left(W_{x}\right) \\
\end{aligned}
$$
​		整体标记判别器各层的权值、偏置项：
$$
\begin{aligned}
\Theta=\left\{W^{l}, b^{l}\right\}_{l=1}^{L}
\end{aligned}
$$
​		那么可以得到：
$$
\begin{aligned}
W_{\Theta, x}=D_{\Theta, x}^{L} W_{x}^{L} D_{\Theta, x}^{L-1} W_{x}^{L-1} \cdots D_{\Theta, x}^{1} W_{x}^{1} \\
\end{aligned}
$$
​		$D_{\Theta, x}^{l}$为对角矩阵，其中如果$x^{l−1}$中的对应元素为正，则对角线中的元素等于 1; 否则，它等于零（这是 ReLU 的定义）。

​		又注意到对于每个$l ∈ \{1,… , 𝐿\}$，有$\sigma(D^l_{\Theta,x})\leq1$，所以我们有：
$$
\begin{aligned}
\sigma\left(W_{\theta, x}\right) \leq \sigma\left(D_{\theta, x}^{L}\right) \sigma\left(W_{x}^{L}\right) \sigma\left(D_{\theta, x}^{L-1}\right) \sigma\left(W_{x}^{L-1}\right) \cdots \sigma\left(D_{\theta, x}^{1}\right) \sigma\left(W_{x}^{1}\right) \leq \prod_{\ell=1}^{L} \sigma\left(W_{x}^{\ell}\right) \\
\end{aligned}
$$
​		于是现在，我们只需要保证$\sigma(W^\ell_x)$恒等于 1，就能够让$f_\Theta$函数满足 1-lipschitz 限制。做法非常简单，只需要将 W 矩阵归一化即可：
$$
\begin{aligned}
\bar{W}_{\mathrm{SN}}^{l}\left(W^{l}\right)=\frac{W^{l}} {\sigma\left(W^{l}\right)}, \text { where } \sigma\left(W^{l}\right)=\tilde{\boldsymbol{u}}_{l}^{\mathrm{T}} W^{l} \tilde{\boldsymbol{v}}_{l}
\end{aligned}
$$

​		至此，SNGAN 通过将 W 矩阵归一为谱范数恒等于 1 的式子，进而控制$f_\Theta$的梯度恒小于等于 1，最终实现了对 D 的 1-lipschitz 限制，最后我们给出 SNGAN 中的梯度下降算法：

![image-20210501125755779](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210501125758.png)

## Condition GAN

​		在实现传统的GAN网络时，我们是不是会有这样的一个困惑，为什么模型的输入是从一个**简单的分布（高斯分布）**中**随机**抽样出来的一个张量，能不能加上**人为控制**的因素呢。比如我们想在生成新图像的时候，让**Generator**能按照用户输入的文字或者图片要求，产生出指定需求的图片。而这正是我们接下来所要介绍的：**CGAN（条件生成式对抗网络）**。

### CGAN实现的问题

​		下面我们来举一个好玩的从文本生成图像例子：

​		假设我们在模型的输入中传入一段文本："red eyes"，记作$x$，而**Generator**所作的就是将**输入的文本张量**和一个从标准正态分布中**抽样出的张量$z$**一起吃掉，吐出一张图片$y$，对于$y$，它需要满足一下两个要求：

- $y$是尽可能真实的动漫人物图片。
- $y$的特征要符合输入的文本要求，比如"red eyes"。

由于$z$是随机抽样的，因此同一个$x$，可以生成多张满足要求的$y$

![image-20210](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430234629.png)



### CGAN的原理

​		我们的目的是，既要让输出的图片真实，也要让输出的图片符合条件$x$的描述。判别器输入便被改成了同时输入$x$和$y$，输出要做两件事情，一个是判断 x 是否是真实图片，另一个是 $y$ 和 $x$ 是否是匹配的。

![image-20210429193604113](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-20210429193604113.png)

​		因此对于判别器可能会面临几种可能

- 生成的图像真实且符合条件 Good
- 生成的图像真实但不符合条件 BAD
- 生成的图像虚假 BAD

![image-20210429](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430234709.png)

### CGAN的架构

​		在GAN这种完全无监督的方式加上一个标签或一点监督信息，使整个网络就可看成半监督模型。其基本架构与GAN类似，只要添加一个条件$x$即可，$x$就是加入的监督信 息，比如说MNIST数据集可以提供某个数字的标签信息，人脸生成可以提供性别、是否微笑、年龄等信息，带某个主题的图像等标签信息。

​		在本小节的内容中，我们将条件记作符号$c（condition)$。

> In each training iteration:
> - Sample m positive examples $\left\{\left(c^{1}, x^{1}\right),\left(c^{2}, x^{2}\right), \ldots,\left(c^{m}, x^{m}\right)\right\}$ from database
> - Sample $\mathrm{m}$ noise samples $\left\{z^{1}, z^{2}, \ldots, z^{m}\right\}$ from a distribution
> - Obtaining generated data $\left\{\tilde{x}^{1}, \tilde{x}^{2}, \ldots, \tilde{x}^{m}\right\}, \tilde{x}^{i}=G\left(c^{i}, z^{i}\right)$
> - Sample m objects $\left\{\hat{x}^{1}, \hat{x}^{2}, \ldots, \hat{x}^{m}\right\}$ from database
> - Update discriminator parameters $\theta_{d}$ to maximize
> $$
> \begin{array}{l}
> \tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log D\left(c^{i}, x^{i}\right) +\frac{1}{m} \sum_{i=1}^{m^{m}} \log \left(1-D\left(c^{i}, \tilde{x}^{i}\right)\right)+\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(c^{i}, \hat{x}^{i}\right)\right),\theta_{d} \leftarrow \theta_{d}+\eta \nabla \tilde{V}\left(\theta_{d}\right)
> \end{array}
> $$
> 
> - Sample $\mathrm{m}$ noise samples $\left\{z^{1}, z^{2}, \ldots, z^{m}\right\}$ from a distribution	
>- Sample m conditions $\left\{c^{1}, c^{2}, \ldots, c^{m}\right\}$ from a database
> - Update generator parameters $\theta_{g}$ to maximize
> 
> $$
>\tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log \left(D\left(G\left(c^{i}, z^{i}\right)\right)\right), \theta_{g} \leftarrow \theta_{g}-\eta \nabla \tilde{V}\left(\theta_{g}\right)
> $$

​		因为 CGAN 是半监督学习，采样的每一项都是文字和图片的 pair。CGAN 的核心就是判断什么样的 pair 给高分，什么样的 pair 给低分。

#### 判别器

$$
\begin{array}{l}
\tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log D\left(c^{i}, x^{i}\right) \\
+\frac{1}{m} \sum_{i=1}^{m^{m}} \log \left(1-D\left(c^{i}, \tilde{x}^{i}\right)\right)+\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(c^{i}, \hat{x}^{i}\right)\right) \\
\end{array}
$$

​		第一项是正确条件与真实图片的 pair，应该给高分；第二项是正确条件与仿造图片的pair，应该给低分（于是加上了“1-”）；第三项是错误条件与真实图片的 pair，也应该给低分。可以明显的看出，CGAN 与 GANs 在判别器上的不同之处就是多出了第三项。

```
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10,10) 
        #Embedding类返回的是一个形状为[每句词个数， 词维度]的矩阵。
        self.model = nn.Sequential(
            nn.Linear(794,1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x,labels):
    	# 将图片reshape为(batch_size,784)的tensor
        x = x.view(x.size(0),784)
        # labels是用randint随机初始化到[0,9]的(batch_size,)的一维tensor。当作条件condition
        # 每一个数字分配一个长度为10的向量。所以c.shape=(batch_size,10)
        c = self.label_emb(labels)
        # x.shape=(batch_size,794)
        x = torch.cat([x,c],1)
        out = self.model(x) # out.shape=(batch_size,1)
        #可以删除数组形状中的单维度条目，即把shape中为1的维度去掉，但是对非单维的维度不起作用。
        return out.squeeze()
        
D = Discriminator().to(device)
```

#### CGAN判别器的损失函数

```
# 定义判别器的损失函数交叉熵及优化器
criterion = nn.BCELoss()

# 定义判断器对真图片的损失函数
real_validity = D(real_images,real_labels)
# 损失比较，与1
d_loss_real = criterion(real_validity,torch.ones(batch_size).to(device))
# 判别器生成的值
real_score = real_validity

# 定义判别器对假图片（即由潜在空间点生成的图片）的损失函数
### 创建batch_size行100列的随机数的tensor，随机值的分布式均值为0，方差为1
z = torch.randn(batch_size,100).to(device)
### 输入的条件，即想要生成的数字[0,9]，因此创建大小为batch_size的一维张量，其中取值范围在[0,9]
conditions = torch.randint(0, 10, (batch_size,)).to(device)
### 通过正态分布生成的特征数为100的z,以及conditions,产生一张fake_images
fake_images = G(z, conditions)
# 定义判断器对假图片的损失函数
fake_validity = D(fake_images, conditions)
# 损失比较，与0
d_loss_fake = criterion(fake_validity, torch.zeros(batch_size).to(device))
fake_score = fake_images  # 生成器生成的值

# total
d_loss = d_loss_fake + d_loss_real
```



#### 生成器

$$
\tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log \left(D\left(G\left(c^{i}, z^{i}\right)\right)\right)
$$

​		生成器的目的就是让判别器给仿造图片的得分越高越好，这与传统 GANs 本质上是一致的，只是在输入上多了一个参数 c。

```
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 每一个数字分配一个长度为10的向量，总共十个数字，产生了10*10的tensor
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        ) 
        
    def forward(self, z, labels):
    	# 定义z是个从randn取样得到的shape为(batch_size,100)的二维的tensor
        z = z.view(z.size(0), 100) 
        # labels是用randint随机初始化到[0,9]的(batch_size,)的一维tensor。当作条件condition
        # 每一个数字分配一个长度为10的向量。所以c.shape=(batch_size,10)
        c = self.label_emb(labels)
        # x.shape=(batch_size,110)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        # 将out reshape为(batch_size,28,28)的tensor
        return out.view(x.size(0), 28, 28)

G = Generator().to(device)
```

#### CGAN生成器的损失函数

```
# 定义生成器对假图片的损失函数，这里我们要求
# 判别器生成的图片越来越像真图片，故损失函数中
# 的标签改为真图片的标签，即希望生成的假图片，
# 越来越靠近真图片

### 创建batch_size行100列的随机数的tensor，随机值的分布式均值为0，方差为1
z = torch.randn(batch_size, 100).to(device)
### 输入的条件，即想要生成的数字[0,9]，因此创建大小为batch_size的一维张量，其中取值范围在[0,9]
conditions = torch.randint(0, 10, (batch_size,)).to(device)
### 通过正态分布生成的特征数为100的z,以及conditions,产生一张fake_images
fake_images = G(z, conditions)

# 定义生成器的损失函数
validity = D(fake_images, fake_labels)
g_loss = criterion(validity, torch.ones(batch_size).to(device)) #标签为1
```

#### 目标函数

$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x} \mid \boldsymbol{c})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z} \mid \boldsymbol{c})))] \tag{5}
$$

### 训练模型

```
# 定义判别器的损失函数交叉熵及优化器
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(),lr=0.0001)
g_optimizer = torch.optim.Adam(G.parameters(),lr=0.0001)

#Clamp函数x限制在区间[min, max]内
def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

#开始训练
total_step = len(data_loader)

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(data_loader):
        step = epoch*len(data_loader)+i+1
        images = images.to(device)
        labels = labels.to(device)
        # 定义图像是真或假的标签
        real_labels = torch.ones(batch_size).to(device)  #真标签全是1
        fake_labels = torch.randint(0,10,(batch_size,)).to(device) ##返回均匀分布的[0,10]之间的整数随机值
        # ================================================================== #
        #                      训练判别器                                    #
        # ================================================================== #

        # 定义判断器对真图片的损失函数
        real_validity = D(images,labels)
        d_loss_real = criterion(real_validity,real_labels)  #损失比较，与1
        real_score = real_validity   #判别器生成的值
        # 定义判别器对假图片（即由潜在空间点生成的图片）的损失函数
        z = torch.randn(batch_size,100).to(device)
        #创建batch_size行100列的随机数的tensor，随机值的分布式均值为0，方差为1
        fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
        #创建batch_size行列不指定的随机整数的tensor，随机值的区间是[low, high)[0,10]
        fake_images = G(z,fake_labels)
        fake_validity = D(fake_images,fake_labels)

        d_loss_fake = criterion(fake_validity, torch.zeros(batch_size).to(device)) #损失比较，与0
        fake_score = fake_images   #生成器生成的值
        d_loss= d_loss_fake + d_loss_real

        # 对生成器、判别器的梯度清零
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        训练生成器                                  #
        # ================================================================== #

        # 定义生成器对假图片的损失函数，这里我们要求
        # 判别器生成的图片越来越像真图片，故损失函数中
        # 的标签改为真图片的标签，即希望生成的假图片，
        # 越来越靠近真图片

        z = torch.randn(batch_size, 100).to(device)
        fake_images = G(z, fake_labels)
        validity = D(fake_images, fake_labels)
        g_loss = criterion(validity, torch.ones(batch_size).to(device)) #标签为1

        # 对生成器、判别器的梯度清零
        # 进行反向传播及运行生成器的优化器
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item() * (-1)))
    # 保存真图片
    if (epoch + 1) == 1:   #只是保存一张
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    # 保存假图片
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

# 保存模型
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
```

### 可视化结果

```import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

reconsPath = './cgan_samples/real_images.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

reconsPath = './cgan_samples/fake_images-50.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
```

![cgan_fake](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/cgan_fake.png)

### 生成自定义图片

```
from torchvision.utils import make_grid
z = torch.randn(100, 100).to(device)
labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).to(device)

images = G(z, labels).unsqueeze(1)
grid = make_grid(images, nrow=10, normalize=True)
#make_grid用于把几个图像按照网格排列的方式绘制出来
#每行的图片数量为10
#normalize如果为True，则把图像的像素值通过range指定的最大值和最小值归一化到0-1。
fig, ax = plt.subplots(figsize=(10,10))
#fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作。
#表示figure 的大小为宽、长（单位为inch）
ax.imshow(grid.permute(1, 2, 0).detach().cpu().numpy(), cmap='binary')
#grid.permute(1, 2, 0)将tensor的维度换位，原来的顺序是（0，1，2）
#当使用detach()分离tensor但是没有更改这个tensor时，并不会影响backward()
#显示设置，两端发散的色图 colormaps
ax.axis('off')
```

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-20210430122733783.png" alt="image-20210430122733783" style="zoom:67%;" />

```
def generate_digit(generator, digit):
    z = torch.randn(1, 100).to(device)
    label = torch.LongTensor([digit]).to(device)
    img = generator(z, label).detach()
    img = 0.5 * img + 0.5  #还原图像，反归一化
    return transforms.ToPILImage()(img)

generate_digit(G, 8)
```

![image-20210430122812197](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-20210430122812197.png)

### CGAN的讨论

![image-20210429202005655](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/image-20210429202005655.png)

​		大部分的 CGAN 判别器都采用上述架构，为了把图片和条件结合在一起，往往会把x丢入一个网络产生一个 embedding，condition 也丢入一个网络产生一个 embedding，然后把这两个 embedding 拼在一起丢入一个网络中，这个网络既要判断第一个 embedding 是否真实，同时也要判断两个 embedding 是否逻辑上匹配，最终给出一个分数。但是也有一种CGAN 采用了另外一种架构。

![CGAN](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/CGAN.png)

​		首先有一个网络它只负责判断输入 $x$ 是否是一个真实的图片，并且同时产生一个embedding，与 $c$ 一同传给第二个网络；然后第二个网络只需判断 $x$ 和 $c$ 是否匹配。最终两个网络的打分依据模型需求进行加权筛选即可。
​		第二种模型有一个明显的好处就是判别器能区分出为什么这样的 pair 会得低分，是因为 $c$ 不匹配还是 $x$ 不够真实；然而对第一种模型却不知道得分低的原因是什么，这会造成一种情况就是生成器产生的图片已经足够清晰了，但是因为不匹配 c 而得了低分，而生成器不知道得分低的原因是什么，依然以为是产生的图片不够清晰，那这样生成器就有可能朝着错误的方向迭代。
​		不过，目前第一种模型还是被广泛应用的，其实事实上二者的差异在实际中也不是特别明显。



## **DCGAN**

​		生成对抗网络是指一类采用对抗训练方式来进行学习的深度生成模型，其包含的判别网络和生成网络都可以根据不同的生成任务使用不同的网络结构。

​		本节介绍一个生成对抗网络的具体模型：**深度卷积生成对抗网络**（Deep Convolutional Generative Adversarial Network，DCGAN）[Radford et al., 2016]。在 DCGAN 中，**判别网络**是一个传统的深度卷积网络，但使用了带步长的卷积来实现下采样操作，不用最大汇聚（pooling）操作；**生成网络**使用一个特殊的深度卷积网络来实现，使用微步卷积来生成64 × 64大小的图像。

### 生成器模型	

​		DCGAN在GAN的基础上优化了网络结构，加入了**卷积层（Conv）**、转置卷积 （ConvTranspose）、**批量正则（Batch_norm）**等层，使得网络更容易训练，下图为使用卷积层的DCGAN的生成器网络结构示意图。 

![image-20210430135607419](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430135607.png)

​		可以看出，生成器的输入是一个 100 维的噪声，中间会通过 4 层卷积层，每通过一个卷积层通道数减半，长宽扩大一倍 ，最终产生一个 64\*64\*3 大小的图片输出。值得说明的是，在很多引用 DCGAN 的paper中，误以为这 4 个卷积层是**Wide Convolution（宽卷积）层**，但其实在DCGAN 的介绍中这 4 个卷积层是 **Fractionally Strided Convolution（微步幅度卷积）层**，二者的差别如下图所示：

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430135311.png" alt="weibu" style="zoom:67%;" />

​		上图左边是宽卷积，用 3\*3 的卷积核把 2\*2 的矩阵反卷积成 4\*4 的矩阵；而右边是微步幅度卷积，用 3\*3 的卷积核把 3\*3 的矩阵卷积成 5\*5 的矩阵，二者的差别在于，宽卷积是在整个输入矩阵周围添 0，而微步幅度卷积会把输入矩阵拆开，在每一个像素点的周围添 0。

​		上述的两种**从低维特征映射到高维特征的卷积**操作称为**转置卷积**（Transposed Convolution）[Dumoulin et al., 2016]，也称为反卷积（Deconvolution）[Zeiler et al., 2011]。

​		**转置卷积**的动图见https://nndl.github.io/v/cnn-conv-more

#### 代码示例

​		$nz$是$z$输入向量的长度，$ngf$与通过生成器传播的特征图的大小有关，$nc$是输出图像中的通道数（对于RGB图像设置为3）。 

以下是生成器的代码：

```
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```

### 判别器模型

​		判别器网络是一个二进制分类网络，该二进制分类网络将图像（3 * 64 * 64）作为输入并输出输入图像是真实的（与假的相对）的标量概率。D 可以看成是 G 结构反过来的样子，简而言之通过一系列的Conv2d，BatchNorm2d和LeakyReLU层对其进行处理，然后通过Sigmoid激活函数输出最终概率，最终得到一个 1024 * 4 * 4 的结果，再通过view(-1)进行展开成一维tensor。

​		如果需要解决此问题，则可以用更多层扩展此体系结构，但是使用**跨步卷积**，**BatchNorm**和**LeakyReLU**仍然具有重要意义。 DCGAN论文提到，使用跨步卷积而不是通过池化来进行下采样是一个好习惯，因为它可以让网络学习自己的池化功能。 **BatchNorm**和**LeakyReLU**函数还有利于梯度的传递，这对于G和D的学习过程都是至关重要的。

#### 代码示例

```
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

### 训练

​		最后，既然我们已经定义了GAN框架的所有部分，我们就可以对其进行训练。请注意，训练GAN某种程度上是一种艺术形式，因为不正确的超参数设置会导致模式崩溃，而对失败的原因几乎没有解释。在这里，我们将严格遵循Goodfellow论文中的算法，同时遵守ganhacks（https://github.com/soumith/ganhacks）中显示的一些最佳做法。即，我们将为真实和伪造构建不同的小批量图像，并调整G的目标函数以最大化$log(1 − D(G(z)))$。

​		模型的训练主要分为两个部分。第1部分**更新了判别器**，第2部分**更新了生成器**。

#### 第1部分-训练判别器

​		回想一下，训练判别器的目的是最大程度地提高将给定图片正确分类的可能性。实际上，我们要$\max \log(D(x))+\log(1-D(G(z)))$。由于**ganhacks**提出了单独的小批量建议，因此我们将分两步进行计算以至可以“通过提升随机梯度来更新鉴别器”

- 首先，我们将从训练集中构造一批真实样本，通过D，计算损失$log(D(x))$，然后再反向传播计算梯度。
- 其次，我们通过生成器构造一批假样本，将该批样本通过D，计算损失$log(1-D(G(z)))$，并通过反向传播来累积梯度。
- 现在，利用从所有真实批次和所有伪批次累积的渐变，我们将这个过程其称为“判别器”一次更新。

#### 第2部分-训练生成器

​		如原始论文所述，我们希望通过最小化$log(1-D(G(z)))$来训练生成器，以产生更好的**fake image**。如前所述，Goodfellow证明这不能提供较好的梯度，尤其是在学习过程的早期。作为解决方法，我们希望最大化$log(D(G(z))$。

​		在代码中，我们通过以下方式实现此目的：使用判别器对第1部分的Generator输出进行分类，使用真实标签作为GT计算G的损失，再反向传播计算G的梯度，最后使用优化器步骤更新G的参数。

​		使用真实标签作为损失函数的GT标签似乎有悖常理，但这允许我们使用$BCELoss$的$log(X)$部分(而不是$log(1-x)$部分)，这正是我们想要的。

​		最后，我们将进行一些统计报告，并在每个Epoch结束时，将我们的fixed_noise batch输入到到生成器中，来直观地跟踪G的训练进度。

- $Loss_D$-判别器损失，计算为所有真实批次和所有假批次的损失总和$log(D(x))+ log(1-D(G(z)))$。
- $Loss_G$-生成器损失计算为$log(D(G(z)))$
- $D(x)$-所有真实批次的判别器的平均输出。这应该**从接近1开始**，然后在G变得更好时理论上**收敛到0.5**。
- $D(G(z))$-所有假批次的判别器的平均输出。第一个数字在D更新之前，第二个数字在D更新之后。这些数字应**从0开始**，并随着G变好而**收敛至0.5**。

> 注意：此步骤可能需要一段时间，具体取决于您运行了多少个Epoch以及是否从数据集中删除了一些数据。  

```
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ################################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))  #
        ################################################################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(0)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, num_epochs, i, len(dataloader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
```

### 可视化结果

​		最后，让我们来看看我们做得怎么样。在这里，我们将查看三个不同的结果。首先，我们来看看D和G在训练中的损失是如何变化的。其次，我们将在每个时期的Fixed noise batch上可视化G的输出。第三，我们将看一批真实数据和一批来自G的生成的数据。

#### 损失与训练迭代

下面是D&G的损失与训练迭代次数的关系图。

```
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

![image-20](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430191634.png)

#### 图片生成过程的可视化

我们可以用动画来可视化G的训练过程。按播放按钮开始播放动画。

```
#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
```

![image-20210430191907098](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430191907.png)

#### 真实图片和虚假图片

```
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
```

![image-20210430192118904](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210430192119.png)

### Trick

**DCGAN** 的主要优点是通过一些经验性的网络结构设计使得对抗训练更加稳定。比如：

- 使用**带步长**的卷积（在判别网络中）和**微步卷积**（在生成网络中）来代替汇聚操作，以免损失信息；
- 使用**批量归一化**；
- **去除卷积层之后的全连接层**；
- 在生成网络中，除了**最后一层**使用 **Tanh** 激活函数外，其余层都使用**ReLU**函数；
- 在判别网络中，都使用**LeakyReLU**激活函数。

## ProGAN

​		对于之前我们实现的**CGAN**以及**DCGAN**而言，我们都会看到，生成的图片和原图还是具有一定的差异的。比如在清晰度方面，CGAN和DCGAN都无法产生高清的大图，因此**StackGAN**和**LapGAN**应运而生。但是无论是StackGAN还是LapGAN而言，如果现在我们想生成超高分辨率的图像，譬如 1024×1024 图片，我们将需要用到的 GANs 结构会非常多，这样会导致网络深度巨大，训练起来非常慢。

​		为了解决这一问题，**PGGAN（渐进式增长 GAN）**提出的想法是，我们只需要一个 GANs 就能产生 1024×1024 图片。但是一开始的时候 GANs 的网络非常浅，只能学习低分辨率（4×4）的图片生成，随着训练进行，我们会把 GANs 的网络层数逐渐加深，进而去学习更高分辨率的图片生成，最终不断的更新 GANs 从而能学习到 1024×1024 分辨率的图片生成。

### ProGAN的特点

​		ProGAN 中的 Pro 并非 Professional，而是 **Progressive**，即逐渐的意思。也就是说，PGGAN 与 StackGAN 和 LapGAN 的最大不同在于，后两者的网络结构是固定的，但是 PGGAN 随着训练进行网络会不断加深，网络结构是在不断改变的。这样做最大的好处就是，PGGAN 大部分的迭代都在较低分辨率下完成，训练速度比传统GANs提升了 2-6 倍。

![image-20210506122429443](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506142110.png)

​		可以看到在相同的训练时间下，**Progressive growing**相比**Fixed layers**能训练更多的图片。同时，这种增量学习过程大大提升了训练的稳定性，可以减少**模式坍塌（mode collapse)**发生的几率。此外，由低到高分辨率使得渐进式增长GAN网络能首先关注于**高层结构**(图像**最模糊版本**中可以辨别的模式)，在逐渐填入细节。这种方式可以降低网络完全错误陷入某种高层结构的可能性，有助于提升最终图像的质量。

### ProGAN的模型架构

![image-20210506112311317](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506112315.png)

​		从上图的模型架构中我们可以看到，训练开始于有着一个 4×4 像素的低空间分辨率的生成器和判别器。随着训练的改善，我们逐渐向生成器和判别器网络中添加层，进而增加生成图片的空间分辨率。所有现存的层在过程中保持可训练性。这里 N×N 是指卷积层在 N×N 的空间分辨率上进行操作。这个方法使得在高分辨率上也能稳定合成并且加快了训练速度。

​		右图我们展示了六张通过使用在 1024 × 1024 空间分辨率上渐进增长的方法生成的样例图片。

​		但是上述这样的做法会有一个问题，就是从 4×4 的输出变为 8×8 的输出的过程中，**网络层数的突变**会造成 GANs 的急剧不稳定，使得 GANs 需要花费额外的时间从动荡状态收敛回平稳状态，这会影响模型训练的效率。为了解决这一问题，PGGAN 提出了**平滑过渡技术（Smooth Fade in）**。(在这里可以参考Residual Network中**残差单元**的思想去理解)

![image-20210506113445987](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506113453.png)

​		如上图所示，当把生成器和判别器的分辨率加倍时，会平滑的增强新的层。我们以从16 × 16 像素的图片转换到 32 × 32 像素的图片为例。在转换（b）过程中，把在更高分辨率上操作的层视为一个**残差单元**（即ResNet中的**residual block**），权重 $\alpha$ 从 0 到 1 线性增长。当 $\alpha=0$ 的时候，相当于（a），但是生成器的输出像素和判别器的输入像素转换为了 32 × 32 ；当 $\alpha=1$ 的时候，相当（c），像素也转换为了 32 × 32 。所以生成器和判别器的整体输出为
$$
\begin{aligned}
\alpha*L_{G-new}+(1-\alpha)*upsample(L_{G-old}) \\
\alpha*L_{D-new}+(1-\alpha)*downsample(L_{D-old}) 
\end{aligned}
$$
​		所以在转换过程中，生成样本和真实样本的像素，是从 16 × 16 到 32 × 32 转换的。

​		上图中的 $2×$ 和 $0.5×$ 指利用**最近邻插值(上采样)**和**平均池化(下采样)**分别对图片分辨率加倍和折半。$toRGB$ 表示将一个层中的特征向量投射到 RGB 颜色空间中，$fromRGB$ 正好是相反的过程；这两个过程都是利用 1 × 1 卷积。

​		详细的生成器和判别器的网络结构见下图，对于细节的描述有兴趣的同学可以参考**paper[https://arxiv.org/pdf/1710.10196.pdf]。**在这里我们就简单了解一下ProGAN的主要思想，并不要求实现。（在之后的StyleGAN模型中，我们会基于这一思想，具体实现）

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506123421.png" alt="image-20210506123149198" style="zoom: 33%;" />

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506123138.png" alt="image-20210506123135598" style="zoom: 33%;" />

> **像素归一化(Pixel Normalization)**
>
> ​		相比于通用的采用 Batch Normalization(BN) 层，ProGAN 采用了像素归一化。Pixel Norm 层没有可训练的权重，其是将每个像素的特征向量的归一化到单位长度，用于在生成网络中的 conv 层之后。这样做主要是为了防止信号强度在训练过程中失控。
> $$
> \begin{aligned}
> b_{x, y}=\frac{a_{x, y}}{\sqrt{\frac{1}{C} \sum_{j=0}^{C} a_{x, y}^{j}+\epsilon}}
> \end{aligned}
> $$
> 
>
> ​		其中，C 通道的每个像素$(x, y)$值被归一化为固定长度。a 为输入 tensor，b 为输出 tensor，$\epsilon$ 是很小的值，避免分母为 0.		

​		综上，便是 PGGAN 的主要思想，PGGAN 的主要优点就是能更快的生成高质量的样本。

## StyleGAN

​		对于ProGAN模型而言，我们知道它的确能更快的生成高分辨率的图片，但是由于 ProGAN 是逐级直接生成图片，我们没有对其增添控制，我们也就无法获知它在每一级上学到的特征是什么，这就导致了它**控制所生成图像的特定特征的能力非常有限**。换句话说，这些特性是互相关联的，因此尝试调整一下输入，即使是一点儿，通常也会同时影响多个特性。而这也被称为特征之间的**相互纠缠（entanglement）**。

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506215442.png" alt="image-20210506215438183" style="zoom:50%;" />

​		比如在上图中，我们尝试稍微调整一下输入，试图将肤色进行改变。但是经过生成器生成的输出完全变样。

​		我们希望有一种更好的模型，能让我们控制住输出的图片是长什么样的，也就是在生成图片过程中每一级的特征，要能够特定决定生成图片某些方面的表象，并且相互间的影响尽可能小。于是，在 ProGAN 的基础上，StyleGAN 作出了进一步的改进与提升。

​		StyleGAN中的**"Style"**是指数据集中人脸的**主要属性**，比如人物的姿态等信息，而不是风格转换中的图像风格，这里**Style**是指**人脸的风格**，包括了**脸型**上面的**表情**、人脸**朝向**、**发型**等等，还包括**纹理**细节上的人脸**肤色**、人脸**光照**等方方面面。

### StyleGAN模型架构

​		**StyleGAN** 的网络结构包含两个部分。

- 第一个是**Mapping network**，即下图 (b)中的左部分，由隐藏变量 $z$  生成中间隐藏变量 $w$ 的过程，这个 $w$ 就是用来控制生成图像的$style$，即风格，为什么要多此一举将 $z$ 变成 $w$ 呢，后面会详细讲到。 
- 第二个是**Synthesis network**，它的作用是生成图像，创新之处在于给每一层子网络都喂了 A 和 B，A 是由 $w$ 转换得到的仿射变换，用于控制生成图像的风格，B 是转换后的随机噪声，用于丰富生成图像的细节，即每个卷积层都能根据输入的 A 来调整**"style"**。整个网络结构还是保持了 **PG-GAN(progressive growing GAN)** 的结构。最后StyleGAN论文还提供了一个高清人脸数据集**FFHQ。**

![image-20210506220504612](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506220510.png)

### 映射网络

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506220925.png" alt="image-20210506220922358" style="zoom:50%;" />

​		在详细介绍映射网络之前，我们先介绍一下**latent code**。latent code 简单理解就是，为了更好的对数据进行分类或生成，需要对数据的特征进行表示。但是数据有很多特征，这些特征之间相互关联，耦合性较高，导致模型很难弄清楚它们之间的关联，使得学习效率低下，因此需要寻找到这些**表面特征之下隐藏的深层次的关系**，将这些关系进行解耦，得到的**隐藏特征，即latent code**。由 latent code组成的空间就是 latent space。而 **Mapping network** 要做的事就是对**隐藏空间（latent space）**进行解耦，	

​		**映射网络**的目标是将输入向量 $z$ 编码为中间向量 $w$ ，中间向量 $w$ 的不同元素控制不同的视觉特征。这是一个非常重要的过程，因为**使用输入向量 $z$ 来控制视觉特征的能力是非常有限**的，因为它必须遵循训练数据的概率密度。例如，如果黑头发的人的图像在数据集中更常见，那么更多的输入值将会被映射到该特征上。因此，该模型无法将部分输入（向量中的元素）映射到特征上，这一现象被称为**特征纠缠**。

​		另一种理解是由于一般 $z$ 是符合**均匀分布**或者**高斯分布**的随机向量，但在实际情况中，并不是这样。比如特征：头发长度和男子气概。**下图（a）**中就是这两个特征的组合，左上角缺失的部分代表头发越长，男子气概越强。**下图（b）**是直接用均匀分布或者高斯分布到特征的映射，如果在纵轴上进行取值，那么无论是头发长度和男子气概都会发生变化，因为这里的实线进行了扭曲。**下图（c）**通过映射网络后对特征的扭曲进行了缓解，较好地拟合了（a）的形状。

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506222542.png" alt="image-20210506222542842" style="zoom: 50%;" />

​		因此，**映射网络**就是通过**使用另一个神经网络**，该模型可以生成一个不必遵循训练数据分布的向量，并且可以**减少特征之间的相关性**。

​		映射网络由 8 个全连接层以及Leaky relu组成，它的输出 $w$ 与输入的随机向量Latent Code（512×1）的大小相同。

### 样式模块 Style Module（AdaIN）

![image-20210506234821804](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210506234823.png)

​		**AdaIN（自适应实例标准化）**模块将映射网络创建的编码信息 $w$ 传输到生成的图像中。该模块被添加到**合成网络（Synthesis Network）**的每个分辨率级别中，并定义该级别中特征的可视化表达式：

- 卷积层输出的每个通道首先进行标准化，求出每个通道的均值 $\mu$ 和方差 $\sigma$ 以确保第三步的缩放和切换具有预期的效果；

- 中间向量 $w$ (1×512) 使用另一个全连接的网络层（标记为 $A$）转换为(2×512)的张量，最后将这个张量分割为每个通道的缩放系数 $y_{s,i}$ 和偏置量 $y_{b,i}$ ； 

- 缩放系数 $y_{s,i}$ 和偏置量 $y_{b,i}$ 的向量通过**AdaIN**方法切换卷积输出的每个通道，从而定义卷积中每个卷积核的重要性。

  这个调优操作将信息从 $w$ 转换为可视的表达方式； 

### 常数输入

<img src="C:\Users\56550\AppData\Roaming\Typora\typora-user-images\image-20210507134906372.png" alt="image-20210507134906372" style="zoom:50%;" />

​		大多数的模型以及其中的 ProGAN 使用随机输入来创建生成器的初始图像（即 4×4 级别的输入）。StyleGAN 团队发现图像特征是由 $w$ 和 AdaIN 控制的，因此可以**忽略初始输入**，**并用常量值替代**。这一操作可以有效的**减少了特征纠缠**，不依赖初始输入的向量，仅使用映射网络获得的**潜在因子 $w$** ，使得网络更容易学习。

### 随机变化（Stochastic variation）

​		人们的脸上有许多小的特征，可以看作是随机的，例如：雀斑、发髻线的准确位置、皱纹、使图像更逼真的特征以及各种增加输出的变化。如下图：

![image-20210507162725789](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507162730.png)

​		将这些小特征插入 GAN 图像的常用方法是在输入向量中添加随机噪声。然而，在许多情况下，由于上述特征的纠缠现象，控制噪声的影响是很复杂的，从而会导致图像的其它特征受到影响。		

![image-20210507155940132](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507162818.png)

​		StyleGAN 中的噪声以类似于 **AdaIN** 机制的方式添加，在 AdaIN 模块之前向每个通道添加一个缩放过的噪声，即将原始的 $image$ 变为 $image+W_{noise}*noise_{randn}$ 。其中$W_{noise}$为随机变化层需要学习的参数，$noise_{randn}$为使用标准正态分布抽样出的张量。

### 样式混合（Style Mixing）

​		StyleGAN 生成器在合成网络的每个级别中使用了中间向量 $w$ 以及线性变换后的 $A$，这有可能导致网络学习到这些级别是相关的。为了降低相关性，模型将两个不同的**latent code** $z_1$和$z_2$输入到 **mappint network** 中，分别得到 $w_1$ 和 $w_2$，代表两种不同的 style 。然后在 synthesis network 中随机选一个中间的交叉点，交叉点之前的部分使用 $w_1$ ，交叉点之后的部分使用 $w_2$ 。随机的切换确保了网络不会学习并依赖于一个合成网络级别之间的相关性，生成的图像应该同时具有 source A （对应$z_1$）和 source B （对应$z_2$）的特征。

​		根据交叉点选取位置的不同，style组合的结果也不同。下图中分为三个部分，第一部分是 **Coarse styles from source B**，分辨率(4x4 - 8x8)的网络部分使用B的style，其余使用A的style, 可以看到图像的**身份特征随souce B**，但是肤色等**细节随source A**；第二部分是 **Middle styles from source B**，分辨率(16x16 - 32x32)的网络部分使用B的style，这个时候生成图像不再具有B的身份特性，发型、姿态等都发生改变，但是肤色依然随A；第三部分 **Fine from B**，分辨率(64x64 - 1024x1024)的网络部分使用B的style，此时身份特征随A，肤色随B。

​		由此可以**大致推断**：

- **低分辨率**的style 控制姿态、脸型、配件 比如眼镜、发型等style。
- **高分辨率**的style控制肤色、头发颜色、背景色等style。

![image-20210507162254640](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507162259.png)

### 在 W 中的截取技巧（Truncation Trick）

​		Truncation Trick 不是StyleGAN提出来的，它很早就在GAN里用于图像生成了，感兴趣的可以追踪溯源。从数据分布来说，低概率密度的数据在网络中的表达能力很弱，直观理解就是，低概率密度的数据出现次数少，能影响网络梯度的机会也少，但**并不代表低概率密度的数据不重要**。可以提高数据分布的整体密度，把分布稀疏的数据点都聚拢到一起，类似于PCA，做法很简单，首先找到数据中的一个平均点，然后计算其他所有点到这个平均点的距离，对每个距离按照统一标准进行压缩，这样就能将**数据点都聚拢**了，但是又不会改变点与点之间的距离关系。

​		而在生成模型中的一个挑战，是处理在训练数据中表现不佳的地方。这导致了生成器无法学习和创建与它们类似的图像（相反，它会创建效果不好的图像）。为了避免生成较差的图像，StyleGAN 截断了中间向量 $w$，迫使它保持接近“平均”的中间向量 $\bar{w}$ 。

​		对模型进行训练之后，通过选择多个随机的输入，用映射网络生成它们的中间向量，并计算这些向量的平均值，从而生成“平均”的平均值 $\bar{w}$ 。当生成新的图像时，不用直接使用映射网络的输出，而是将值 $w$ 转换为 $w^{\prime}=\overline{w}+\psi(w-\overline{w})$，其中 $\psi$ 的值定义了图像与“平均”图像的差异量（以及输出的多样性）。有趣的是，在仿射转换块之前，通过对每个级别使用不同的 $\psi$ ，模型可以控制每个特征集与平均值的差异量。

### 微调超参数

​		StyleGAN的另外一个改进措施是更新几个网络超参数，例如训练持续时间和损失函数，并将离得最近的放大或缩小尺度替换为**双线性采样**。

​		综上，加入了一系列附加模块后得到的 StyleGAN 最终网络模型结构图如下：

![image-20210507164951115](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507164954.png)

### 存在的问题以及改进（StyleGAN2）

​		如果我们观察StyleGAN中所有特征图，就会发现从64 × 64分辨率开始都存在类似水滴的伪影。作者认为原始的**AdaIN**摧毁了层与层间传递的信息。特征图创造出强烈的信号（伪影）为了防止被摧毁。

![20210507170107](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/StyleGAN%E4%BC%AA%E5%BD%B1.png)

#### 新的网络架构

![image-20210507170210684](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507170216.png)

重点观察 (b) (c) 图的变换，我们可以看到网络结构的变化：

- 移除（简化）初期处理常数的方式

- 归一化特征时无需求均值

- 将噪声模块从风格模块中移出

#### Weight demodulation

​		对特征图的一系列操作改为对权重的操作。特征图只经过卷积处理并添加噪声。该方法在保留完全可控性的同时消除了伪影。
​		缩放特征图改为缩放卷积权重（mod）：
$$
\begin{aligned}
w_{ijk}^{\prime}=s_i*w_{ijk}
\end{aligned}
$$
​		$s_i$是第 $i$ 个输入特征图的缩放比例。

​		经过缩放和卷积后，输出激活的标准差为：
$$
\begin{aligned}
\sigma_i=\sqrt{\sum_{i,k}w_{ijk}^{\prime}{}^2}
\end{aligned}
$$
​		demod权重，旨在使输出恢复到单位标准差：
$$
\begin{aligned}
w_{i j k}^{\prime \prime}=w_{i j k}^{\prime} / \sqrt{\sum_{i, k} w_{i j k}^{\prime}{ }^{2}+\epsilon}
\end{aligned}
$$

#### 图像质量与生成器平滑度

​		通过实验发现，感知路径长度（PPL）分数低则生成图像的质量高。作者假设在训练过程中，由于判别器会对残破的图像进行惩罚，因此生成器改进的最直接方法是有效地拉伸产生良好图像的潜在空间，这将导致劣质图像被压缩到较小的变化快速的潜在空间中。虽然这可以在短期内提高平均输出质量，但累积的失真会损害训练状态，进而损害最终图像质量。所以将PPL作为正则项加到生成器上。

**Lazy regularization**

​		损失是由损失函数和正则项组成，优化的时候也是同时优化这两项的。lazy regularization就是让正则项可以减少优化的次数，比如每16个minibatch才优化一次正则项，这样可以减少计算量，同时对效果也没什么影响。

**Path length regularization**

​		在生成人脸的同时，我们希望能够控制人脸的属性，不同的latent code能得到不同的人脸，当确定latent code变化的具体方向时，该方向上不同的大小应该对应了图像上某一个具体变化的不同幅度。为了达到这个目的，设计了 Path length regularization。

​		无论 $w$ 或图像空间方向如何，这些渐变应具有接近等长度，即小位移产生相同大小的变化。表示从潜在空间到图像空间的映射是良好的。路径长度正则化不但提高了图片的生成质量，而且使得生成器更平滑，生成的图片反转回latent code更容易了。

![image-20210507173858184](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507173902.png)

​		最后从结果对比图可以看到，StyleGAN2（config F）极大地改善了PPL的分布，使之更加紧凑，生成图像的质量也更高。

#### 渐进式增长修正

​		StyleGAN使用的Progressive growth会有一些缺点，如下图，当人脸向左右偏转的时候，牙齿却没有偏转，即人脸的一些细节如牙齿、眼珠等位置比较固定，没有根据人脸偏转而变化，造成这种现象是因为采用了Progressive growth训练，Progressive growth是先训练低分辨率，等训练稳定后，再加入高一层的分辨率进行训练，训练稳定后再增加分辨率，即每一种分辨率都会去输出结果，这会导致输出频率较高的细节，如下图中的牙齿，而忽视了移动的变化。paper的解释如下：

> We believe the problem is that in progressive growing each resolution serves momentarily as the output resolution, forcing it to generate maximal frequency details, which then leads to the trained network to have excessively high frequencies in the intermediate layers, compromising shift invariance.

![image-20210507175256280](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507175256.png)

​		使用Progressive growth的原因是高分辨率图像生成需要的网络比较大比较深，当网络过深的时候不容易训练，但是skip connection可以解决深度网络的训练，因此有了下图中的三种网络结构，都采用了skip connection。

![image-20210507175555659](https://gitee.com/shenhao-stu/picgo/raw/master/CS224W/image-20210507175555659.png)

对上述三种网络结构的实验比较如下图，可以看出使用skips连接的生成器PPL最小。使用残差网络的判别器对FID有利。

![image-20210507175625519](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507175625.png)

> 代码参考：https://github.com/NVlabs/stylegan2-ada-pytorch

## CycleGAN

​		循环一致的对抗网络(cycle-consistent adversarial networks)、DiscoGAN和DualGAN是2017年提出的三个非常相似的生成对抗网络的模型。由于这三个模型非常相似，本节就详细讲解CycleGAN。

### CycleGAN解决的问题

​		假设我们现在要训练一个风格迁移的神经网络，也就是说输入一张图片，输出一张它的不同风格的图片，比如说输出一张具有梵高画风的图片。

![image-20210507183054207](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507183112.png)

​		那么我们考虑应用 GANs 技术。一个很自然的想法是给它增添一个判别器，这个判别器用来判别输入的图像是真实的还是 G 伪造的。

![image-20210507183203935](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507183207.png)

​		这个架构看似合理，但是会有一些潜在的危险。在生成器很深时，它的输出和输入差别是可能非常大的，存在一种情况是当输出图像靠近真实分布 Y 里的某一张图像时，生成器就发现了一个 BUG，只要它的输出越逼近这张真实图像，判别器给的评分就越高，于是生成器最终可以完全忽略输入长什么样，输出这张偷学到的真实图片，就能产生"高质量"图片。

![image-20210507183317588](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507183322.png)

​		为了消除这种潜在危险，CycleGAN 诞生了。

### CycleGAN的原理

​		为了防止生成器学习到具有欺骗性的造假数据，我们只需要保证生成器的输出和原图具有很高的相似性，也就是不丢失原图的特征，于是 CycleGAN 中加入了一个新的生成器，**把第一个生成器的输出当作输入丢进去**，希望能输出一个和原始输入尽可能相似的图片，如果能够比较好的还原回原始图片，证明第一个生成器的输出保留了大量原始图片的特征，输出结果是较为可靠的；而如果不能较好的还原回原始图片，意味着第一个生成器可能使用了“造假”的输出结果。

![image-20210507183958494](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507183958.png)

​		那么CycleGAN 还可以做成双向的，除了从 $X_{domain}-(G_{X\to Y})\to Y_{domain}-(G_{Y\to X})\to X_{domain}$ 的训练，同时还会有 $Y_{domain}-(G_{Y\to X})\to X_{domain}-(G_{X\to Y})\to Y_{domain}$ 这样的训练，在第二种训练中会新引入一个判别器，功能同样是保证整次训练的输入和输出尽可能相似。

![image-20210507184456584](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507184456.png)

### 损失函数

<img src="https://gitee.com/shenhao-stu/picgo/raw/master/CS224W/image-20210507185442066.png"/>

​		如上图所示，CycleGAN中有两个映射，并且分别对应了两个判别器 $D_x$ 和 $D_y$ 。模型希望 $F(G(x))=\hat{x}\approx x,G(F(y))=\hat{y}\approx y$，也就是希望（b）和（c）中的两个环为闭环。模型定义了一个名为**“cycle-consistency loss”**的损失函数，用于评定这两个映射的准确率。定义如下：
$$
\begin{aligned}
\mathcal{L}_{\text {cyc }}(G, F) &=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\|F(G(x))-x\|_{1}\right] \\
&+\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\|G(F(y))-y\|_{1}\right]
\end{aligned}
$$
​		而对于两个**生成对抗网络**的损失函数则为：
$$
\begin{aligned}
\mathcal{L}_{\mathrm{GAN}}\left(G, D_{Y}, X, Y\right) &=\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\log D_{Y}(y)\right] \\
&+\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\log \left(1-D_{Y}(G(x))\right]\right. \\
\mathcal{L}_{\mathrm{GAN}}\left(F, D_{X}, X, Y\right) &=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\log D_{X}(x)\right] \\
&+\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\log \left(1-D_{X}(G(y))\right]\right.
\end{aligned}
$$
​		**模型整体的损失函数**还要再加上两个生成对抗网络的损失函数，定义如下：
$$
\begin{aligned}
\mathcal{L}\left(G, F, D_{X}, D_{Y}\right)=& \mathcal{L}_{\mathrm{GAN}}\left(G, D_{Y}, X, Y\right) \\
&+\mathcal{L}_{\mathrm{GAN}}\left(F, D_{X}, Y, X\right) \\
&+\lambda \mathcal{L}_{\mathrm{cyc}}(G, F)\\
\end{aligned}
$$
​		**Identity Loss**：论文中的作者也发现如果引入额外的损失函数Loss function去鼓励输入和输出之间的映射网络来尽可能**保留颜色信息**，那么对于绘画生成照片的任务而言会更加有帮助。否则生成器G、F可以自由地去改变输入图片的颜色。比如，在论文中，作者提到，当训练莫奈的画作到真实照片的映射网络时，生成器往往会将白天的画作映射到日落时分的照片，因为这种映射关系可能能使得生成对抗网络的损失和循环一致的损失更加小。而对于Identity Loss的定义如下：
$$
\begin{aligned}
\mathcal{L}_{\text {identity }}(G, F) &=\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\|G(y)-y\|_{1}\right] \\
&+\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\|F(x)-x\|_{1}\right]
\end{aligned}
$$
​		即当提供目标域的真实样本作为生成器的输入时，则将生成器正则化为接近标识的映射。

### 代码实践

#### 数据集的读取

**数据集下载地址**：https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
.
├── datasets  
|   ├── <dataset_name>         # i.e. monet2photo
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images (i.e. monet)
|   |   |   └── B              # Contains domain B images (i.e. photo)
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images (i.e. monet))
|   |   |   └── B              # Contains domain B images (i.e. photo)

```
import glob
import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)  # 将几个变化整合在一起
        self.unaligned = unaligned

        # 匹配 `datasets/monet2photo/(train or test)/(A or B)` 下的所有文件并打乱
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):  # `__getitem__`, 允许用户像字典一样访问数据 : X[key] -> value

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            # 不对齐则随机出一张图片
            item_B = self.transform(
                Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            )
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        # 两者中取一张取数量大的
        return max(len(self.files_A), len(self.files_B))

import torchvision.transforms as transforms

# Dataset loader
transforms_ = [
    transforms.Resize(int(size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # PIL.Image/np.ndarray (HWC) [0, 255] -> torch.FloatTensor (CHW) [0.0, 1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]  # 将三个通道 `Normalize`
dataloader = torch.utils.data.DataLoader(
    ImageDataset(r"datasets/monet2photo", transforms_=transforms_, unaligned=True),
    batch_size=batch_size,
    shuffle=True,
)
```

#### 训练集展示

```
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

# 展示一些训练图片
real_batch = next(iter(dataloader))['B']
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
```

![image-20210509000405600](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210509000414.png)

#### 生成器部分

​		网络整体上经过一个降采样然后上采样的过程，中间是一系列残差块，数目由实际情况确定，根据论文中所说，当输入分辨率为128 × 128，采用6个残差块，当输入分辨率为256 × 256甚至更高时，采用9个残差块，其源代码如下。

```
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
```

#### 判别器部分

​		结构比生成器更加简单，经过5层卷积，通道数缩减为1，最后池化平均，尺寸也缩减为1x1，最后reshape一下，变为（batchsize, 1）

```
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])
```

#### 损失函数选择

​		损失函数包含三种：**对抗损失**，**循环一致损失**和**identity损失**。对抗损失采用LSGAN的方式，所以是MSE Loss；循环一致损失按照论文采用L1 Loss；identity损失同样采用L1 Loss：

```python3
generator_x2y = Generator(3, 3).to(device)
generator_y2x = Generator(3, 3).to(device)
discriminator_x = Discriminator(3).to(device)
discriminator_y = Discriminator(3).to(device)

loss_function_GAN = torch.nn.MSELoss().to(device)
loss_function_cycle = torch.nn.L1Loss().to(device)
loss_function_identity = torch.nn.L1Loss().to(device)
```

#### 优化器选择

​		优化时采用生成器G和生成器F同时进行，判别器$D_X$和判别器$D_Y$分开进行的优化策略，所以需要三个optimizer。优化算法采用收敛性能较好的Adam优化器，其中beta1和beta2分别为0.5和0.999，训练过程中不进行学习率的动态调整：

> ​		因为优化时采用生成器G和生成器F同时进行，所以itertools.chain 迭代器能够将多个可迭代对象合并成一个更长的可迭代对象，以便**同时进行更新**。

```python3
import itertools

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(
    itertools.chain(generator_x2y.parameters(), generator_y2x.parameters()),
    lr=lr,
    betas=(0.5, 0.999),
)
optimizer_D_A = torch.optim.Adam(discriminator_x.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(discriminator_y.parameters(), lr=lr, betas=(0.5, 0.999))
```

#### 生成器训练过程

损失函数的计算是CycleGAN最核心也是最复杂的内容，生成器的损失计算分为三个过程：

（1）对域X和域Y计算identity损失

（2）生成器计算生成样本的对抗性损失

（3）计算循环一致损失，即重构损失

```python3
# 1：计算生成器损失
optimizer_G.zero_grad()
# 1.1：Identity loss
identity_y = generator_x2y(real_y)
loss_identity_y = loss_function_identity(identity_y, real_y)*5.0
identity_x = generator_y2x(real_x)
loss_identity_x = loss_function_identity(identity_x, real_x)*5.0

# 1.2：计算生成器对伪造样本的损失
x2y = generator_x2y(real_x)
discriminator_out_x2y = discriminator_y(x2y)
loss_generator_x2y = loss_function_GAN(discriminator_out_x2y, real_label)
y2x = generator_y2x(real_y)
discriminator_out_y2x = discriminator_x(y2x)
loss_generator_y2x = loss_function_GAN(discriminator_out_y2x, real_label)

# 1.3：计算循环一致损失
recovered_A = generator_y2x(x2y)
loss_cycle_x2y2x = loss_function_cycle(recovered_A, real_x)*10.0
recovered_B = generator_x2y(y2x)
loss_cycle_y2x2y = loss_function_cycle(recovered_B, real_y)*10.0

# 1.4：计算生成器总体损失，并更新参数
loss_G = loss_identity_x + loss_identity_y + loss_generator_x2y + loss_generator_y2x + loss_cycle_x2y2x + loss_cycle_y2x2y
loss_G.backward()
optimizer_G.step()
```

#### 判别器训练过程

判别器 $D_X$ 和判别器 $D_Y$ 的训练过程是分开的，二者的训练原理相同，这里仅以 $D_X$ 为例。判别器 $D_X$ 的对抗性损失包含对真实样本的损失和对伪造的损失两个部分，也就是要分别计算：

```python3
# 2：计算判别器X损失
optimizer_DX.zero_grad()

# 2.1：判别器对真实样本的损失
pred_real = discriminator_x(real_x)
loss_D_real = loss_function_GAN(pred_real, real_label)

# 2.2：计算判别器X对伪造样本的损失
discriminator_out_y2x = discriminator_x(y2x.detach())
loss_D_fake = loss_function_GAN(discriminator_out_y2x, fake_label)

# 2.3：计算判别器X整体损失，并更新
loss_D_A = (loss_D_real + loss_D_fake)*0.5
loss_D_A.backward()
optimizer_DX.step()
```

![image-20210509001709966](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210509001715.png)

### CycleGAN的讨论

​		CycleGAN 也不是没有问题。CycleGAN: a Master of Steganography **(隐写术)** [Casey Chu, et al., NIPS workshop, 2017 ]这篇论文就指出，CycleGAN 存在一种情况，是它能**学会把输入的某些部分藏起来**，然后**在输出的时候再还原**回来。比如下面这张图：

![image-20210507184841134](https://gitee.com/shenhao-stu/picgo/raw/master/DataWhale/20210507184841.png)

​		可以看到，在经过第一个生成器的时候，屋顶的黑色斑点不见了，但是在经过第二个生成器之后，屋顶的黑色斑点又被还原回来了。这其实意味着，第一个生成器并没有遗失掉屋顶有黑色斑点这一讯息，它只是用一种人眼看不出的方式将这一讯息隐藏在输出的图片中（例如黑点数值改得非常小），而第二个生成器在训练过程中也学习到了提取这种隐藏讯息的方式。

​		那生成器隐藏讯息的目的是什么呢？其实很简单，**隐藏掉一些破坏风格相似性的“坏点”会更容易获得判别器的高分，而从判别器那拿高分是生成器实际上的唯一目的**。

​		综上，CycleGAN 所宣称的 CycleConsistency 其实是不一定能完全保持的，毕竟生成器的学习能力非常强大，即便人为地赋予它诸多限制，它也有可能学到一些 trick 去产生一些其实并不太符合人们要求的输出结果。

## 总结和深入阅读

​		生成对抗网络 [Goodfellow et al., 2014] 是一个具有开创意义的深度生成模型，突破了以往的概率模型必须通过最大似然估计来学习参数的限制。然而，生成对抗网络的训练通常比较困难。DCGAN[Radford et al., 2016]是一个生成对抗网络的成功实现，可以生成十分逼真的自然图像。[Yu et al., 2017] 进一步在文本生成任务上结合生成对抗网络和强化学习来建立文本生成模型。而对于对抗生成网络的训练不稳定问题的一种有效解决方法是W-GAN[Arjovsky et al., 2017]，通过用Wasserstein距离替代JS散度来进行训练。

​		在图像生成的应用上，从一开始的DCGAN模型，到能够生成高分辨率的StackGAN、LapGAN，以及通过学习低分辨率的图片生成，随着训练进行，网络层数逐渐加深的渐进式增长模型，通过一步步的优化，改进，最终StyleGAN2应运而生，给与了我们完全不一样的人脸生成体验。

​		虽然深度生成模型取得了巨大的成功，但是作为一种无监督模型，其主要的缺点是缺乏有效的客观评价，很难客观衡量不同模型之间的优劣。

---

**预备知识 **

> 交叉熵和散度

**理论参考来源：**

> 邱锡鹏 神经网络与深度学习 https://nndl.github.io/

--- ***By: 沈豪***


>Github：https://github.com/shenhao-stu


**关于Datawhale**：

>Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。**同时Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。**

****
