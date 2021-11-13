# Transformer 模型

谷歌2017年文章《All you need is attention》提出Transformer模型，文章链接：http://arxiv.org/abs/1706.03762。下面对几个基于Transformer的主要的模型进行简单总结。

# Bert

来自文章《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》。
整个模型可以划分为embedding、transformer、output三部分。

1. embedding部分由word embedding、position embedding、token type embedding三个模型组成，三个embedding相加形成最终的embedding输入。
2. transformer部分使用的是标准的Transformer模型encorder部分。
3. output部分由具体的任务决定。对于token级别的任务，可以使用最后一层Transformer层的输出；对于sentence级别的任务，可以使用最后一层Transformer层的第一位输出，即[CLS]对应的输出。
   文章链接：https://arxiv.org/abs/1810.04805

# GPT

来自文章《Improving Language Understanding by Generative Pre-Training》和《Language Models are Unsupervised Multitask Learners》。
GPT为生成式模型。如果说BERT使用了Transformer模型中的encoder部分，那GPT就相当于使用了Transformer模型中的deconder部分。

文章链接：
GPT:
https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf
GPT-2:
https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf

# Transformer XL

来自文章《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》。
相比于传统的transformer模型，主要有以下两点修改：

每个transformer节点除了使用本帧上层节点的数据外，还使用了上一帧上层节点的数据；
在做position embedding的时候，使用相对位置编码代替绝对位置编码。
文章链接：
http://arxiv.org/abs/1901.02860

# ALBERT

来自文章《ALBERT: A Lite BERT for Self-supervised Learning of Language Representations》。
ALBERT（A Lite BERT）即轻量级的BERT，轻量级主要体现在减少传统BERT的参数数量，提高模型的泛化能力。相比于传统BERT有以下三点区别：

1. 降低embedding层的维度，在embedding层与初始隐藏层之间增加一个全连接层，将低纬的embedding提高至高维的隐藏层纬度，相当于对原embedding层通过因式分解降维；
2. 在transformer层之间进行参数共享；
3. 使用SOP（sentence order prediction）代替NSP（next sentence prediction）对模型进行训练，ALBERT参数规模比BERT比传统BERT小18倍，但性能却超越了传统BERT。

### 模型大小对比

|        | BERT               | RoBERTa            | DistilBERT | XLNet              | ALBERT           |
| ------ | ------------------ | ------------------ | ---------- | ------------------ | ---------------- |
| Size/M | Base 110 Large 340 | Base 110 Large 340 | 66         | Base 110 Large 340 | Base 12 Large 18 |













