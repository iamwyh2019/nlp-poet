[README in English](#Eng)

# Group project: 藏头诗生成



## 项目结构

本项目主要包含以下代码文件：

- `dataloader.py`：封装了自定义的数据库类，包括读入诗句、汉字编号、划分训练/验证/测试集、提供随机批量数据等；
- `model.py`：封装了模型，包括LSTM模型和Transformer模型，但后者由于效果不佳已被弃用；
- `main.py`：训练LSTM模型；
- `main_transformer.py`：训练Transformer模型，已被弃用；
- `interactive.py`：通过命令行与训练好的模型进行交互，由于实现了GUI，已被弃用；
- `mainWidget.py`和`mainWidget.ui`：描述GUI的窗口结构；
- `testui.py`：启动GUI；
- `uientry.py`：提供GUI与训练好的模型的接口，包括生成藏头诗和切换模式；



## 训练与测试方式

在训练之前，需要将数据稍作清洗。我们将诗句按”五言/七言”和“绝句/律诗“分为四类，删去句子长短不一致的诗（例如，”青箬笠、绿蓑衣，斜风细雨不须归“ 这首诗其中一句只有六个字），每类按每行一首诗的格式输出为txt文件，编码为`UTF-8 with BOM`。

### 训练方式

训练模型的步骤如下：

1. 调用自定义的数据库读入诗词，将标点符号转为标识符#，在开头加上标识符@，在末尾加上标识符\*；为所有符号和汉字（统称为单位）建立字库，每个单位赋予一个编号；同时，计算每个单位的tf-idf值；

2. 数据库将编码后的数据随机分为训练/验证/测试集（比例为 85%/15%/0%，我们没有在训练后独立进行测试）。以七言绝句为例，加上标识符后，每句诗为一个长为33的向量。每批数据的X为向量第0-31位，Y为第1-32位，逻辑上为”给定前a位，预测第a+1位“；

3. 每一轮训练时，`hidden_state`均初始化为全0向量；每次循环时，数据库提供一批测试数据(X,Y)，网络运算X后得到预测结果Y'，损失L采用分类交叉熵，每一个单位为一类，类的权重为这个单位的tf-idf值。框架自动计算损失L的梯度，并进行反向传播。循环直至取遍所有批次。

   对于不同批次的训练数据，`hidden_state`沿用上一批次得到的结果。循环直至取遍所有测试批次。优化器采用Adam，初始学习率为0.001。为防止梯度爆炸，每次均以参数0.1进行梯度截断；

4. 每一轮训练后，在验证数据集上进行类似的操作，但仅计算损失L，不计算梯度与反向传播；

5. 打乱测试数据；

6. 重复3-5共`num_epoch`次（项目中为300），记录每次训练和验证的平均损失，绘制曲线，并保存验证损失最小的模型。循环结束后，保存最终模型。

### 测试方式

测试阶段，我们使用GUI调用模型生成藏头诗，人工评价生成质量。

我们为每种模式预设了一首诗用于初始化隐藏层`hidden_state`。五言绝句、七言绝句、五言律诗、七言律诗的预设诗分别为：孟浩然《春晓》，杜牧《清明》，王勃《送杜少府之任蜀州》和陆游《游山西村》。这个隐藏层可以为生成的诗提供风格信息。

以七言绝句为例，我们使用杜牧《清明》，在模型加载后逐字输入模型得到`hidden_state`。对于用户输入的诗头，我们用这个隐藏层和第一个”头“输入模型中得到第一句第二个字，再用得到的隐藏层和第二个字得到第三个字，以此类推。生成第七个字后，用第七个字和分隔标识符#依次输入模型，不记录输出结果，只保存隐藏层。对于第二至四句诗重复上述操作，最终得到藏头诗。



<h1 id="Eng"> Group project: Generating Chinese poetry with heads</h1>



## Basic Definition

A Chinese poetry contains four or eight sentences, each with five or seven Chinese characters. **Jueju** means poems with four sentences, and **Lvshi** means eight; **Wuyan** means poems with five characters in each sentence, and **Qiyan** means seven. Depending on these two features, we split the poems into four kinds (of course there are exceptions). For example, a poem with four five-word sentences is **Wuyan Jueju**, and one with eight seven-word sentences is **Qiyan Lvshi**.

A **head** is the concatenation of the first character in each sentence. For example, below is a Chinese poem:

```
春眠不觉晓，
处处闻啼鸟。
夜来风雨声，
花落知多少。
```

and its head is `春处夜花`.

## Project Structure

This repository contains the following codes:

- `dataloader.py`: defines a custom `dataloader` class. It reads in the poems dataset, assigns number to each Chinese character, splits train/test/validation datasets, and feeds random batches of data;
- `model.py`: defines two models, the LSTM model and the Transformer model. The latter performs bad and is no longer used;
- `main.py`: trains the LSTM model;
- `main_transformer.py`: trains the Transformer model. No longer used;
- `interactive.py`: interacts with the trained model via command line. It's no longer used as we have implemented a GUI;
- `mainWidget.py`和`mainWidget.ui`: GUI files;
- `testui.py`: start the GUI;
- `uientry.py`: provides interface from the trained model to the GUI, including generating poems and switching between the four kinds of poetry.

## Training and Testing

Before training, we split the poems into four kinds as mentioned above (and discard poems that do not fall into these four categories). The four datasets are in the `data` folder, encoding `UTF-8 with BOM`.

### Training

We train the model in the following steps

1. Read in the dataset with the custom `dataloader` class, convert all punctuations into #, add identifiers @ and * to the beginning and ending of each poem respectively, assign number to each characters, and compute the tf-idf value for each character;

2. Split the dataset into train/validation/test datasets (at ratio 85%/15%/0%, we only test the model during training). Take Qiyan Jueju as an example, after adding identifiers, each poem is represented as a vector of length 33 (written as `a[0..32]`). The input of the model is  `a[0..31]` and output `a[1..32]`, so conceptually, given the first X characters, the model predicts the (X+1)th character;

3. before each epoch, `hidden_state` is initialized as all-zero; in each epoch, the `dataloader` provides a batch of data (X,Y), the model reads in X and predicts Y'; we compute loss L(Y,Y') with catagorical cross-entropy, with the tf-idf value of each character as its weight; the gradient of L is then back-propagated through the model;

   In each epoch, for all batches except the first one, we use the same `hidden_state` passed from the last batch; we use the Adam optimizer, and the gradient is clipped with factor 0.1 in case the gradient explodes;

4. After each round of training, we do the same thing on validation set, but we only compute L without computing or propagating gradient;

5. Shuffle the training set;

6. Repeat 3-5 for `num_epoch` times (in our project it's 300), record the average training/validation loss in each epoch, plot the curve, and save the model with minimum validation loss. Also save the final model.

### Testing

During testing, we call the model from the GUI, and judge the quality of generated poems manually.

We set a poem for each kind of poems. Prior to testing, the trained model reads in this poem and outputs the `hidden_state`. We then use this `hidden_state` to generate poems. This state encodes style information. In our project, we use "*Chuanxiao*" (by Meng Haoran) for **Wuyan Jueju**, "*Qingming*" (by Du Mu) for **Qiyan Jueju**, "*Seeing Vice-Magistrate Du Off to his Post in Sichuan*" (by Wang Bo) for **Wuyan Lvshi**, "*A Tour of the Village West of the Mountain*" (by Lu You).

Take **Qiyan Jueju**. For a given head (with four characters), we send the first character of the head and the `hidden_state` mentioned above to the model to get the second character and the new `hidden_state`. The first and second character is then concatenated and sent into the model (with the new `hidden_state`) to get the third character, so on so forth. When we have the seventh (namely, we have the first sentence), append identifier # and the second character of the head to the sentence, then send it to the model to get the second sentence. Repeat this process till we get a complete poem.

## Open-source

**If you are taking the same course, please adhere to the academic integrity code**. In other cases, you are free to use any parts of this project. I am NOT responsbile for ANY incurred results.
