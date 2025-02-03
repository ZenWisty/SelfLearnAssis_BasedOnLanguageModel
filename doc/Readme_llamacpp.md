# llama.cpp 简单实践&源码解析
llama.cpp 是一个简洁、强大的大语言模型（不止llama，支持多种）部署框架，它基于ggml工程搭建，包含了量化、分布式部署等技术栈。llama.cpp 项目在github上十分活跃，更新速度很快。因此这里采用的学习路径是借用一个直观的例子脚本，即llama.cpp/examples/simple.cpp 来解读源码和说明。这个例子包含了llamacpp部署推理时大部分的主要功能。

## 环境配置
llama.cpp 的环境配置相对来说简单：我拿的是github上的 afa8a9ec9b520137bbd1ca6838cda93ee39baf20 版本。
删除根目录下的 CMakePresets.json；<br>
我默认是使用GPU backend的。切换到ggml目录下的CMakeLists.txt，将其中的第143行：<br>
option(GGML_CUDA                            "ggml: use CUDA"                                  OFF)<br>
改为:<br>
option(GGML_CUDA                            "ggml: use CUDA"                                  ON)<br>

需要使用这个例子还得从huggingface 上下载书生大模型：https://huggingface.co/internlm/internlm2-1_8b/tree/main?library=transformers <br>
为方便，将simple中的main函数中的model路径修改成我们下载的路径：<br>
```cpp
int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path = "<你的model地址目录，如我的是**\AppData\huggingface\hub\models--internlm--internlm2-1_8b\snapshots\e96c608609a487e5674dffb18f7fa135d6c4b171>";
    // ...
```
然后切到根目录用Cmake build即可。<br>
## 源码解读
