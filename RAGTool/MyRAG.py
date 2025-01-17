import asyncio
import os
import glob
import shutil

from pydantic import BaseModel

from pathlib import Path
DATA_PATH = Path('E:\Python_work\LLM_MetaGPT\hyx_workspace\dataBase')
EXAMPLE_DATA_PATH = Path('E:\Python_work\LLM_MetaGPT\hyx_workspace\data')
PERSIST_DATABASE_PATH = Path('E:\Python_work\LLM_MetaGPT\hyx_workspace\dataBase')/ "rag"

from metagpt.logs import logger
from metagpt.rag.engines import SimpleEngine
from metagpt.rag.schema import (
    ChromaIndexConfig,
    ChromaRetrieverConfig,
    ElasticsearchIndexConfig,
    ElasticsearchRetrieverConfig,
    ElasticsearchStoreConfig,
    FAISSRetrieverConfig,
    LLMRankerConfig,
)
from metagpt.utils.exceptions import handle_exception

HUGGINGFACEHUB_API_TOKEN = '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'
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embed_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L6-v2")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
# tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')

# RAG example of MetaGPT
DEFAULT_INIT_RAG_FILE = r'E:\Python_work\LLM_MetaGPT\hyx_workspace\data\rag\init_rag_engine.txt'
LLM_TIP = "If you not sure, just answer I don't know."

class RAGTool:
    """Show how to use RAG."""

    def __init__(self, engine: SimpleEngine = None, 
                 use_llm_ranker: bool = True, 
                 embed_mod = None,
                 init_input_files = None):
        '''
        Args:
            engine: 传入的engine，没有即默认None，会在初次调用的时候自动构建,建议传入
            use_llm_ranker: 
            embed_mod: 现成构建好的embedding模型传入，没有即默认None，会在内部根据 config 构建
        '''
        self._engine = engine
        self._use_llm_ranker = use_llm_ranker
        self._embed_mod = embed_mod
        self._init_input_files = init_input_files

    @property
    def engine(self):
        if not self._engine:
            ranker_configs = [LLMRankerConfig(llm=rerank_model)] if self._use_llm_ranker else None

            self._engine = SimpleEngine.from_docs(
                input_files=[DEFAULT_INIT_RAG_FILE] if not self._init_input_files else self._init_input_files,
                retriever_configs=[FAISSRetrieverConfig(dimensions=384)],
                # retriever_configs=[ChromaRetrieverConfig(persist_path=PERSIST_DATABASE_PATH)],
                ranker_configs=ranker_configs,
                embed_model=embed_model if embed_model else None
            )
        return self._engine

    @engine.setter
    def engine(self, value: SimpleEngine):
        self._engine = value

    @staticmethod
    def _print_retrieve_result(result):
        """Print retrieve result."""
        logger.info("Retrieve Result:")

        for i, node in enumerate(result):
            logger.info(f"{i}. {node.text[:10]}..., {node.score}")

        logger.info("")
    
    @staticmethod
    def _print_query_result(result):
        """Print query result."""
        logger.info("Query Result:")

        logger.info(f"{result}\n")


if __name__ == '__main__':
    # 初始化一个RAGTool对象，用于查询和检索。 需要给出初始的文档，用于构建 RAG 模型的文档库`。
    
    # 是否删除原有的db #########
    Refresh_database = True
    # database 的路径 #########
    this_file_path = os.path.abspath(__file__)
    db_path = os.path.join(os.path.dirname(this_file_path), 'database')
    # 需要读入的 files 的dir路径 #########
    folder_path = os.path.join(os.path.dirname(this_file_path), 'files/design_pattern')


    # 开始执行
    if Refresh_database:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            os.makedirs(db_path)
    file_paths = glob.glob(os.path.join(folder_path, '*'))  # 使用 glob.glob() 获取所有文件的路径 ;模式 * 表示匹配任意数量的字符

    # TODO: 要添加的功能：这里RAGTool 的构建需要考虑 调用 from persist path，或者在构建时就先删除(clear) 原路径下的数据库
    ragpool = RAGTool(use_llm_ranker=False, embed_mod=embed_model, 
                      init_input_files=file_paths)

    # TODO: persist 函数上移， 显示初始化db 的path
    ragpool.engine.persist(db_path)
    
    # # add relative docs
    # TRAVEL_QUESTION = f"请用中文回答，Agent AI 这篇论文讲了什么内容，可以引用他的相关abstract。 {LLM_TIP}"
    # # travel_filepath = r'E:\Python_work\LLM_MetaGPT\MetaGPT-main\examples\yxh_localworkspace\AgentMaterials\2401.03568v2_Note.txt'
    # # ragpool.engine.add_docs([travel_filepath])

    # # query again
    # nodes = ragpool.engine.retrieve(TRAVEL_QUESTION)
    # ragpool._print_retrieve_result(nodes)
    # answer = ragpool.engine.query(TRAVEL_QUESTION)
    # ragpool._print_query_result(answer)
    

    while True:
        # 使用 input() 函数等待用户输入
        user_input = input(">>> ")
        
        # 退出命令
        if user_input.lower() == 'exit':
            print("退出程序。")
            break
        
        # 正经回答
        TRAVEL_QUESTION = f"请用中文回答:{user_input}.{LLM_TIP}"
        # 举例： nodes[0] 中 有 text, filepath 属性，然后就可以了
        nodes = ragpool.engine.retrieve(TRAVEL_QUESTION)
        ragpool._print_retrieve_result(nodes)
        print("=====================>")
        print("part 1:")
        print(nodes[0].text)
        print("part 2:")
        print(nodes[1].text)
        print("=====================>")
        answer = ragpool.engine.query(TRAVEL_QUESTION)
        ragpool._print_query_result(answer)
        