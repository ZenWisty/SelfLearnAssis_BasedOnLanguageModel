# 深度学习思维链，推导过程 & DeepSeek R1 技术解读
## DeepSeek-R1 技术解读
deepseek 认为之前的上海交大包括一些实验室中沿用的O1的方法是没有必要的（因为可以看到本身OpenAI也没有开源，并且隐藏了o1的推理过程）。<br>
deepseek R1 使用的策略不包含SFT（supervised fine-tuning） ，转而采用大规模强化学习、multi-stage training、强化学习之后接着数据冷启动的方式。<br>
deepseek基于v3 模型训练的框架训练R1，即GRPO(Group Relative Policy Optimization)。几千步强化学习训练之后，模型能力增强显著，在AIME2024集上pass@1 score 15.6%上升到71.0%。值得注意的是，论文中提到，在RL training接近收敛的时候，使用了一些 SFT 数据 给到 RL checkpoint 来重新的post training。<br>
deepseek 的创新点包括：在没有任何 SFT 数据 情况下，将强化学习直接应用到基础模型，没有任何搜索树策略选择问题； 此外，在最后 post training 时，直接使用 长思维链数据（COT）进行一个 SFT； 最后是将deepseek R1 的能力提炼到小模型里面的方式。<br>
### pure RL training 部分
1. 需要注意这个过程包含了两个奖励，Accuracy Award 和 Formate rewards，Formate Rewards是针对 deepseek R1 zero存在的 output formate mixing（即输出语言、模式时常出现混淆的情况）问题加入的。这个是通过强制让大模型输出思考内容的时候必须在'<think>'和'</think>'之间，以此作为formate。<br>
在整个deepseek R1 训练过程中间没有使用针对结果或者过程的奖励模型（processing reward models）。因为在训练过程中发现 “reward hacking”现象，使得模型陷入一个被动的不断推理的过程。
2. deepseek 没有太多的公布这次R1训练的training template 。 但是讲了模型的性能和performance进化的过程。<br>
<img src="./DLReasoningDeepSeekR1/deepseek_1.png"><br>
3. 提到了模型有一个self revolution 过程，自我进化。网络模型是一开始基于一个预训练大模型进行微调的。表现在开始模型只能做多轮的回答，但是不能做推理，也不会做test time 的一个 reasoning（自我思考，自我推理）。但是在R1的训练过程中，自我思考的时间越来越长。<br>
<img src="./DLReasoningDeepSeekR1/deepseek_2.png"><br>
4. 整个模型在训练的过程中出现了一个顿悟的时刻，aha的moment.<br>
<img src="./DLReasoningDeepSeekR1/deepseek_3.png"><br>
上图意为，在推理过程中，模型自己意识到之前的推理不够细致，因此做了一些复盘。<br>
文中也提到了deepseek R1 zero 的可读性比较差，语言混杂。

###  deepseek R1 ： 冷启动 强化学习
首先聚焦于两个问题：1）冷启动时候是否能够加入少量的高质量数据提升模型的性能？ 2）如何提升zero的可阅读性？<br>
第一个问题：R1 相较于 R1 zero 搜集了很多长思维链的数据作为RL的微调过程，一方面也是为了提升可读性。<br>
第二个问题：R1加入了一些奖励，（类似于dpo的效果），这些奖励可能降低模型的性能，但是却能够增加可阅读性。<br>
在拒绝采样和监督微调的情况下，需要很多的tricks：比如R1采用的分阶段微调，在RL一个阶段的训练完成后，就会用这个ckpt断点进行拒绝采样（rejecting sampling）去搜集下一轮的数据。这会生成很多的推理轨迹和相关内容，具体数据量是600K条推理数据。当然这里还要加入很多非推理数据，否则整个模型会变得不会说话。<br>

### 不成功的尝试
1. 在整个deepseek R1 训练过程中间没有使用针对结果或者过程的奖励模型（processing reward models）。因为在训练过程中发现 “reward hacking”现象，使得模型陷入一个被动的不断推理的过程。
2. 很多论文都用 Monte Carlo Tree Search,  这个方法被发现会让模型不断的去探索，模型调起来也很难去做大规模的拓展。因为算法越复杂，越难去做一个scaling out。  只有scale law 才会使我们的模型更好。
类似bert没有gpt1，2，3好的原因，bert encoder decoder 结构越来越复杂，没有只有decoder 的结构方便。

### 训练流程：
接下来是deepseek R1 Zero 的训练方法:<br>
直接从 deepseek v3 经过 GRPO 进化到 R1-Zero。没有使用 SFT微调、蒙特卡洛搜索树。<br>
<br>
deepseek R1 训练流程：<br>
cold-start data (Reasoning Data+ Non-Reasoning data), 得到微调过后的 deepseek V3 模型；<br>
微调 V3 模型  经过 GRPO  得到 DeepSeek R1 中间模型；<br>
用DeepSeek R1 中间模型，rejecting sampling 得到 新的 Reasoning Data。然后专门搜集了一些 Non-Reasoning data .<br>
用新搜集到的 data 再对 V3 模型进行 GRPO 学习。<br>
经过好几轮的迭代之后，得到了比较好的 R1 模型。<br>
此时可以对Qwen 和 LLama 模型进行蒸馏，就得到了网上很多千问和llama相关R1 的一系列蒸馏模型。<br>

### GRPO 解析
RL+LLM : <br>
OnPolicy: 每次训练都基于自己的生成模型（Actor），通过教练（Critic）反馈奖励；效率较高，没有模型自生成，问题是模型训练后可能能力不够；<br>
OffPolicy:基于现有标注的情况进行分析，可能有训练样本与模型不匹配的情况；优势是更可能达到模型能力上限，但是效率低。<br>
PPO示意图（PPO很消耗资源）：<br>
<img src="./DLReasoningDeepSeekR1/deepseek_4.png"><br>
1. 整个算法运行起来有四个模型在运行：除了Actor Model ，和专家模型 Critic Model 之外还有 Reward Model 和Reference Model。 
2. Critic Model 通常跟 Actor Model  相当大小
3. LLM 通常只有最后一个 token 会被奖励模型打分，训练在每个token上都准确的价值函数难

GRPO 避免了像PPO那样使用额外的 Critic Model(即Value Model) 近似， 而是使用同一问题下多个采样输出的平均奖励作为 基线。<br>
<img src="./DLReasoningDeepSeekR1/deepseek_5.png"><br>
上面是对比PPO和GRPO的图，如果将PPO中的 Value Model 直接去掉，根据传统的强化学习的经验是无法保证效果的，这个想法相当于直接让Policy Model 对齐Reward Model。 Critic Model 一般是在为了强化学习更加稳定，方差比较少的情况上才加的。现在去掉Critic Model，直接退化成为 Policy Model 是不合理的。为了保证效果能够好，整个算法的关键在于找到一个baseline，使得网络整体在训练的过程中趋于稳定（特别是做过强化学习的都了解，训练是很难保证稳定的，当然这个任务能成功也得益于LLM的泛化性是非常好的的特点）。<br>
解决方案：原先的Critic Model 参与最后的求 Award。 现在没有Critic Model了，直接转变为暴力采样N次，求均值。另外现在变为，Policy Model 去跟 Reference Model 计算 KL 散度。<br>
所以，可见，GRPO 算法没有额外的价值函数，使用的是组内的平均奖励作为基线，使用的组内相对奖励优势函数，这与奖励模型通常在同一问题的不同输出之间进行比较的性质是相符的；  GRPO 直接将训练策略的 πθ 与参考策略的 πref之间计算KL散度，加入到损失函数中，而不是像PPO那样在奖励中添加KL惩罚项。<br>