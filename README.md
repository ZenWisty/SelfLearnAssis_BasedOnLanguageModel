# SelfLearnAssis_BasedOnLanguageModel

### 目的
做一个app，1）方便快速学习所搜集、汇聚来的资料；2）方便整理看过、处理过的资料；同时方便家人、朋友快速跟进我的知识体系。<br>

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

### Release
1. RAG system implementing <br>
    使用案例见 ./RAGTool 中的详细使用记录(基于需要文件资料库的问答被记录了下来，当问答不准确时有纠正的问答对，可用作后续reranker(LORA)的训练金标准) <br>

2. dpo learning inserting <br>


### 功能进一步说明
1. RAG system: <br>
    RAG system 中向量库主要由 FAISSVectorIndex，FAISSINDEXRetriver（faiss.IndexFlatL2）构成。<br>
    其中的向量编码模型（embedding model）根据 benchmark ：https://huggingface.co/spaces/mteb/leaderboard ， 选择了'sentence-transformers/all-MiniLM-L6-v2' ；<br>
    程序启动时，自动根据给定的地址中的所有文件构建database，进而构建RAG系统以及围绕RAG系统的问答系统；<br>
    操作界面基于实时对话，可在其中询问有关资料库的问题<br>
    注意：因为使用LLM Reranker，如果LLM回答的不对，可能会遇到IndexError: list index out of range；这个问题可能出现且暂时无法避免<br>

### 其他工具
1. 自动写代码：./CodeWriter
2. 爬虫&搜集信息：./ScrapyAssistant
3. 自动写文档：./TechDocAssistant


### TODO

3. markdown to mind map
4. user interface for friends to visit
5. multi-tier system inserting into RAG DB(url), agent self motivated visiting url, incase lacking of information
6. bilibili input
