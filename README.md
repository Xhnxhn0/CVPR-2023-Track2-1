# CVPR 2023第一届大模型比赛Track2第2名方案
> 

## 行人检索方案概述
### 模型设计
行人检索我们采用[Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval](https://github.com/anosorae/IRRA)并在其基础上进行改进。模型仅在单卡RTX 4090 24G上训练和推理。通过clip来使文本和图像特征进行对齐，在此基础上，通过三个目标来训练模型，分别是IDloss、SDM和IRR。
- IDloss：我们根据数据给出的属性标注来进行文本图像的多标签分类，目的是使模型能够辨别图像文本的类别以及属性标签所隐含的内容。
- SDM:对齐图像和文本的特征，使用了KL divergence来关联不同模态特征，在对比过程中根据属性标注，对包含关系的图文对增加匹配概率。目的是使相关联（具有包含关系）的图片文本有相近的特征表示。
- IRR：对文本进行掩码后结合图像进行预测文本，通过crossattention来融合模态特征。目的是使文本信息与图像内容相关联。
![框架](.\\pedestrian\\images\\architecture.png)
### 数据处理
我们将行人数据分离出来，根据图文对的属性标注来分配id（正整数，从1开始编号），相同属性标注的图文对具有相同的id。并未对通过任何方式对数据做增强以及清洗。
### 优化策略
根据实验，在比赛数据集上lr=1e-5表现较好，可以适当降低lr使模型更好的收敛。在验证集上选取Max指标最高（或者附近）的epoch（最好不大于25）来作为最终结果模型。
### 复现中可能遇到的问题
- 之前没有考虑到新数据集的相关内容，假如新数据集属性标签不是21类，需要手动在model/build.py 和 model/objectives utils/metrics.py中修改所有关于21类对应的内容。（查找数字21，替换成对应类别数目即可）
- 数据从txt形式转化为data.json后，id应该从1开始编号。
- 联系方式：553193999@qq.com
## 所需文件
> 
文件存放在aistudio上，[数据、初始化预训练模型权重、结果文件以及日志](https://aistudio.baidu.com/aistudio/datasetdetail/218802)  
