# SelfLearnAssis_BasedOnLanguageModel

### 项目目的
做一个工具（包含若干小工具），1）方便快速学习所搜集、汇聚来的资料；2）方便整理处理过的资料；同时方便读者、家入快速跟进知识体系。
整理出来的资料以问答对的形式、或文档整合形式（问答太多时）呈现出来; 4）记录配置和开发的过程以及问题解决过程。 5）作为大模型、Agent这些发展迅速的领域的观察总结以及项目练手<br>
这个项目利用自建的工具所搜集的资料，其主要内容都是围绕 “Agent 的构建方式”和“使用这些Agent所需的基础设施如何构建”的。<br>
关于Agent 和构建这些Agent所需的基础设施的文档简要大纲： <br>
<br>
借用metagpt 的老图作为 单Agent 的定义架构：<br>
单智能体 = LLM + 观察 + 思考 + 行动 (+ 记忆) <br>
![image](./doc/agent_run_flowchart.6c04f3a2.png)
本工程中构建的单智能体也是参照这个架构来构建的。<br>
其中的每一个模块（观察、思考、行动、记忆）都可以依赖或不依赖大模型。
拆分说明每一个模块：<br>
1. 观察：<br> 
在原始的 metagpt 架构中各个单智能体通过接收上下游智能体发出的文字context信息+自己的记忆memory，来完成observe这个动作。<br>
主流的努力方向主要聚焦多模态输入，希望通过扩充输入模态，模拟人类的观察行为，更加全面的观察周围环境，用于后续决策。<br>
目前这个项目中的观察模块除了memory 和 context信息外，使用MiniCPM 的模型来扩充额外输入，文档见：  <br>
2. 思考：<br>
关于AI做决策和逻辑思考能力的相关总结文档：  <br>
3. 行动：<br>
本工程中，行动模块主要体现在不同的单智能体的功能，这些功能包括下文内容<###Release><###功能细节><###其他功能工具>;<br>
NOTE：11月开始学习开发RAG等相关功能，12月底时才发现豆包APP电脑版一定程度上涵盖了这些功能，且有更好的操作界面，所以推荐用豆包，少量需要定制的能力再自己开发。目前这里也有少部分文档是通过豆包APP易操作的界面汇总整理出来的。<br>
4. 记忆：<br>
记忆模块在我理解中多用RAG接外接数据库来实现。<br>
同时也有修改网络结构实现修改网络记忆能力的实现与研究，见:  <br>
5. 此外，本工程开发所基于的基础设施构建方式，见：<br>
6. 多智能体相关开发与研究：<br>

### 环境 & requirements
需要安装 metagpt ，安装方法见 https://docs.deepwisdom.ai/main/zh/guide/get_started/installation.html<br>
建议python <=3.12  >=3.9<br>
其他的见 requirements.txt<br>
另需安装： pip install 'metagpt[rag]'<br>
<br>
注意：<br>
安装时，当volcengine-python-sdk 编译出现问题时，可能原因是Windows路径长度限制：<br>
<br>
Windows系统有最长路径限制，这可能导致安装失败。可以通过修改注册表来解决这个问题。按下Win+R，输入regedit打开注册表编辑器，然后在HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem路径下的变量LongPathsEnabled设置为1<br>
<br>
然后重新运行一遍安装命令即可<br>
<br>

### Release功能
1. RAG system implementing <br>
    使用案例见 ./RAGTool/Readme.md 中的详细使用记录(基于需要文件资料库的问答被记录了下来，当问答不准确时有纠正的问答对，可用作后续reranker和LLM的训练) <br>

2. dpo learning inserting <br>
    案例见 ./RAGTool/dpo/dpo.py<br>


### 功能细节
1. RAG system: <br>
    RAG system 中向量库主要由 FAISSVectorIndex，FAISSINDEXRetriver（faiss.IndexFlatL2）构成。<br>
    其中的向量编码模型（embedding model）根据 benchmark ：https://huggingface.co/spaces/mteb/leaderboard ， 选择了'sentence-transformers/all-MiniLM-L6-v2' ；<br>
    程序启动时，自动根据给定的地址中的所有文件构建database，进而构建RAG系统以及围绕RAG系统的问答系统；<br>
    操作界面基于实时对话，可在其中询问有关资料库的问题<br>
    注意：1）因为使用LLM Reranker，如果LLM回答的不对，可能会遇到IndexError: list index out of range；这个问题可能出现且暂时无法避免；2）目前我在大多数的学习案例和问答过程中使用了mistral-7b 的模型，其编程和理解代码的能力，在牵涉到代码细节的时候不能很好的给出解答（理解和推理），可以用新的code rag rerank 或者 底层换为gpt-4o解决。<br>
    目前的问题：1）对比市面上的国产大模型对话机器人仍有召回率低的问题，在两方面：细微用词偏差仍然有概率导致返回结果不准确，预计通过rerank增强context语境理解能力的方式可以解决这个问题。目前初步满足当下的学习需求；2）对于总结性的问题表现没有对于细节问题的把控表现好。可能需要通过重构summary功能的放入来接解决。<br>

### 其他功能工具
1. 自动写代码：./CodeWriter
2. 爬虫&搜集信息：./ScrapyAssistant
3. 自动写文档：./TechDocAssistant
4. 视频下载 & 视频音频转文字：搜集资料及使用RAG时，有视频资源，如./RAGTool/file/systemDesign 。为了方便归为文档资料，使用 [ScrapVideo.py, splitMp4Audio.py, Audio2text.py] 将视频资源转化为文档资料。其中音频转文档使用的是 Whisper 的模型 'distil-whisper/distil-small.en'; 值得一提，这个模型的采样频率是16000，因此需要将输入的音频频率也重采样到 16000


### TODO

3. markdown to mind map
4. user interface for friends to visit
5. multi-tier system inserting into RAG DB(url), agent self motivated visiting url, incase lacking of information
6. bilibili input
