# llama.cpp 简单实践&源码解析
llama.cpp 是一个简洁、强大的大语言模型（不止llama，支持多种）部署框架，它基于ggml工程搭建，包含了量化、分布式部署等技术栈。llama.cpp 项目在github上十分活跃，更新速度很快。因此这里采用的学习路径是借用一个直观的例子脚本，即llama.cpp/examples/simple.cpp 来解读源码和说明。这个例子包含了llamacpp部署推理时大部分的主要功能。

## 环境配置
llama.cpp 的环境配置相对来说简单：我拿的是github上的 afa8a9ec9b520137bbd1ca6838cda93ee39baf20 版本；<br>
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
llama.cpp  simple.cpp 中分为几大步：
1. ggml_backend_load_all（） dll读入所有需要使用的 ggml backend的函数；
2. llama_model_default_params（）返回一个刚初始化的 llama_model_params 类型 参数。其中包含一些超参，比如放到gpu上的layer的总数：n_gpu_layers，在llama模型中，所有219个layer中有99个被放到 gpu中。
3. llama_model_load_from_file  初始化模型得到 模型类对象
4. llama_model_get_vocab 初始化 vocab 表，相当于python代码中的 tokenizer 的部分功能
5. 转换 input 和 prompt 作为输入，并根据初始化的模型类对象调用llama_init_from_model 初始化 模型context 。
6. 利用context 调用模型，并得到输出结果
对上面的步骤择重点来解析：
### llama_model_default_params（）
#### 所返回的 llama_model_params 的参数:
```cpp
struct llama_model_params {
ggml_backend_dev_t * devices; // list 存放设备，因为可能有多张卡，所以需要list
int32_t n_gpu_layers; // 有多少个 layers 放在 gpu 中
enum llama_split_mode split_mode; // 多个 GPU时如何分layer到gpu上，解析见下面“llama_split_mode”
int32_t main_gpu;	// 多gpu时，且用的是分发制时，主分发gpu用的是哪一块
const float * tensor_split;	// 分配给多gpu的比例
llama_progress_callback progress_callback;	// 进度条（不重要）
void * progress_callback_user_data;	//用于计算进度条的内容（不重要）
const struct llama_model_kv_override * kv_overrides;	// model 的kv对，超参

bool vocab_only;    // 这个根据具体实现，这里还不知道不读入 weight 影响什么
bool use_mmap;      // 是否使用内存映射（重要）
bool use_mlock;     // 是否锁页内存
bool check_tensors; // （不重要）
};
```
```cpp 
//llama_split_mode 定义
LLAMA_SPLIT_MODE_NONE  = 0, // 单 GPU
LLAMA_SPLIT_MODE_LAYER = 1, // 将 layers 和 KV cache 在 GPU 之间 分布式部署
LLAMA_SPLIT_MODE_ROW   = 2, // 将 layers 和 KV cache 在 GPU 之间 分布式部署，如果系统支持tensor split，就采用
```
#### llama_model_load_from_file 函数中新建的 llama_model以及 llama_model中的impl类的内容
llama_model_load_from_file 比较重要，其中会新建一个 llama_model 类，参数都是初始化的，tensor赋值和参数赋值会留到之后；但是llama_model 依然包含一些重要的内容： llama_model 中包含了一个 impl 类：
<br>llamacpp是先建立context存放上下文和需要开辟的空间的信息，然后再分配、赋值存储空间的。<br>
先逐步看一下 llama_model中的 impl类的内容（后文再解释内存/显存分配的模式）：
```cpp
// impl 类
struct llama_model::impl {
// ... 
uint64_t n_elements = 0;	
size_t n_bytes = 0;		// 记录总 bytes
std::string desc_str;	// 描述（不重要）

llama_mmaps mappings;	// 内存映射记录, llama_mmaps 见下面

// 这些是强制把他们放到RAM里的方式（锁页内存？）
llama_mlocks mlock_bufs;
llama_mlocks mlock_mmaps;

// ggml_context_ptr 和 ggml_backend_buffer_ptr，后面会拆解
std::vector<ggml_context_ptr> ctxs;	// tensors 的 context，超参（重要类）
std::vector<ggml_backend_buffer_ptr> bufs; // 存放 context中记录的tensor的数据的buffers

buft_list_t cpu_buft_list;	// 当cpu backends 时，这个类被使用
// gpu backends 时，记录gpu的list，多个gpu的每一个devices，对应一个buffer list
std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list; 

// 记录每个layers的devies 和 buffers列表
// 关于这个layer_dev中牵涉到的各种类，包括 ggml_backend_dev_t
// 是一个复杂的相互引用，见后文详细分解
struct layer_dev {		
ggml_backend_dev_t dev;		
buft_list_t * buft_list;
};

layer_dev dev_input = {};	// 记录 input 和 output 的层单独复制一遍
layer_dev dev_output = {};
std::vector<layer_dev> dev_layer;
};
```

#### impl类中的ggml_context_ptr 
ggml_context_ptr 结构体:
```cpp
// 本质上是一个链表新，begin 和 end都是 ggml_object
struct ggml_context {
size_t mem_size;
void * mem_buffer;		//存这些 object 的 buffer指针
bool   mem_buffer_owned;// 这些 buffer是否是这个device所owned，如果是则要负责释放这些buf
bool   no_alloc;

int    n_objects;

struct ggml_object * objects_begin;
struct ggml_object * objects_end;
};
```
ggml_context_ptr 中的 ggml_object:
```cpp
struct ggml_object {
size_t offs;
size_t size;

struct ggml_object * next;

enum ggml_object_type type;  //三类：GGML_OBJECT_TYPE_TENSOR,GGML_OBJECT_TYPE_GRAPH,GGML_OBJECT_TYPE_WORK_BUFFER

char padding[4];
};
```
#### impl类中的ggml_backend_buffer_ptr相关内容
顺着impl 类的成员 ggml_backend_buffer_ptr 会找到ggml_backend_buffer 结构体。
ggml_backend_buffer 结构体:
```cpp
struct ggml_backend_buffer {
struct ggml_backend_buffer_i  iface;// 这个结构里存放的都是 interface 接口的函数指针
// buft里面包含一个ggml_backend_buffer_type_i 的interface和 一个 ggml_backend_reg_t
ggml_backend_buffer_type_t    buft;
void * context;	 	// 这里注意多态，只有cpu backend和gpu backend的context不一样
size_t size;
enum ggml_backend_buffer_usage usage;
};
```

<br>
这些 ggml_backend 和  registry 相关的东西都有相互之间的引用。这里借用图来分解一下llamacpp中复杂的ggml backend 相互引用关系：<br>
<img src="./llamacpp/ggml1.png" alt="引用图" width="750" height="327"><br>

```cpp 
struct ggml_backend_reg_i {
const char * (*get_name)(ggml_backend_reg_t reg);

// enumerate available devices
size_t             (*get_device_count)(ggml_backend_reg_t reg);
ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

// (optional) get a pointer to a function in the backend
// backends can add custom functions that are not part of the standard ggml-backend interface
void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
};
```

void ggml_backend_load_all_from_path(const char * dir_path)   中会调用：<br>
- ggml_backend_load_best("blas", silent, dir_path);<br>
    - ggml_backend_load_best("cann", silent, dir_path);<br>
    ggml_backend_load_best("cuda", silent, dir_path);<br>
        - get_reg().load_backend(path, silent);	<br>
<br>

get_reg() 获取的是 register ，其中调用：<br>
```cpp
static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}
```
创建 ggml_backend_registry 全局对象，在构建该对象时：<br>
```cpp
struct ggml_backend_registry {
std::vector<ggml_backend_reg_entry> backends;
std::vector<ggml_backend_dev_t> devices;
```
会创建这两个，这里device 在cpubackends的时候是一对一的，但是在用gpu时，devices不与backends一一对应，因为可能多个devices。（在用gpu时，应该是两个backends分别代表 cpu资源和gpu卡，多张卡就是多个devices）。<br>
<br>
前面的很多buffer 可以之后说，注意在private 中有需要私有的实现，比如：<br>
std::unique_ptr<impl> pimpl;    其中 impl中有上文提到的 ggml_context_ptr， 还有  ggml_backend_buffer_ptr，
这里的buffer分cpu的buffer，和使用cuda时的buffer. 我们先聚焦于cuda backend的buffer结构:<br>
<img src="./llamacpp/ggml_2.png" alt="引用图" width="750" height="402"><br>
值得注意的是，最后llama_model param里计算出的总字节数大小，是遍历ctx也就是context 里的总字节数存下来，
ctx->size += GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);<br>
<br>
关于 ggml.c 中的  内存对齐这个问题后面再讨论：const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);<br>
<br>


### llama_model_load_from_file()
llama_model_load_from_file会调用 llama_model_load，其中会创建 llama_model_loader 类：const int status = llama_model_load(path_model, splits, *model, params);<br>
```cpp
try{
    llama_model_loader ml(fname, params.use_mmap, params.check_tensors, params.kv_overrides);
    // ...

    try {
            model.load_hparams(ml);   // 这个 load_hparams 的最后一句是 
            //hparams.rope_type = llama_model_rope_type(this); 是用来设置旋转位置编码的
    try {	// 获取分词词表
            model.load_vocab(ml);   
            // load_vocab 中 有调用 impl 的load，将tokenize的类型 等从 ml中load进来，然后初始化
            // 题外话：llama 用的 tokenizer 是 SPM， 同BPE一样
            // 这个load 中包括添加起始和结束的 字符 ：
            // if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) {add_bos = temp;}
            // if (ml.get_key(LLM_KV_TOKENIZER_ADD_EOS, temp, false)) {add_eos = temp;}
            // 和特殊字符：
                //if (special_eot_id == LLAMA_TOKEN_NULL) {
                //if (false
                //        || t.first == "<|eot_id|>"
                //        || t.first == "<|im_end|>"
                //        || t.first == "<|end|>"
                //        || t.first == "<end_of_turn>"
                //        || t.first == "<|endoftext|>"
                //        || t.first == "<EOT>"
                //        || t.first == "<｜end▁of▁sentence｜>" // DeepSeek
                //   ) {
        // ...
    // 然后开始 load_tensor, 这个是重点。 这里ml已经快要销毁了，因此context这里会将权重加载到
    // gpu 里了， 加载到 model 里面（llama_model 类的对象）
    // 我们单独来讲这个函数
        if (!model.load_tensors(ml)) {
            // ...
        }
        // ...
```
首先说调用完成之后ml中会有哪些成员:(mappings 当下刚初始化，还什么都没用，所以size是0，weights_map 是键值对，里面是每个层 的名字，contexts 是链表ggml_obj+tensors头的图，meta是存gguf的context)<br>
<img src="./llamacpp/ggml_3.png" alt="引用图" width="434" height="391"><br>

#### llama_model_loader 类
源码：llama_model_loader 和 他的构造函数

```cpp
struct llama_model_loader{
// ... 
std::vector<ggml_context_ptr> contexts;
// ...

llama_model_loader::llama_model_loader(
const std::string & fname,
std::vector<std::string> & splits,
bool use_mmap,
bool check_tensors,
const struct llama_model_kv_override * param_overrides_p) {
    // ...

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
    /*.no_alloc = */ true,
    /*.ctx      = */ &ctx,
};
    meta.reset(gguf_init_from_file(fname.c_str(), params)); //gguf 的部分被存放到meta中

    // ...

    files.emplace_back(new llama_file(fname.c_str(), "rb")); 
    // ggml 的ctx 被存放到 contexts里
    contexts.emplace_back(ctx);     // 注意这里其实将所构建的ctx加到了contexts的结尾
    // emplace_back可以传入参数在结尾构建一个， push_back只在结尾添加一个已经构建好的对象

    // ... 下面是一些打印 kv 对信息的操作，和内部统计赋值简单操作，略过

    this->use_mmap = use_mmap;		// 最后设置“是否使用内存映射”
    this->check_tensors = check_tensors;		// 是否检查 tensors
}
```
其中gguf_init_from_file 调用 gguf_init_from_file_impl 函数：<br>
```cpp
struct gguf_context * gguf_init_from_file_impl(FILE * file, struct gguf_init_params params) {
    const struct gguf_reader gr(file);
    struct gguf_context * ctx = new gguf_context;
    // ...

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;
    if (ok && gr.read(ctx->version)) {	// 从 gguf 文件读入 version
        // ....
    }
    // ...
    if (ok && gr.read(n_tensors)) {	// 从 gguf 读 n_tensors，视频例子中是219个
        // ...
    }
    // ...
    if (ok && gr.read(n_kv)) {  // 从 gguf 读 n_kv 键值对，视频例子中是26个
        // ...
    }

    //...
    // KV pairs
    {
        for (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            gguf_type   type     = gguf_type(-1);
            bool        is_array = false;
            uint64_t    n        = 1;
            // ...
            // ?: 这里不知道是在检查什么
            for (size_t j = 0; ok && j < ctx->kv.size(); ++j) {
                if (key == ctx->kv[j].key) {
                    fprintf(stderr, "%s: duplicate key '%s' for tensors %zu and %" PRIi64 " \n", __func__, key.c_str(), j, i);
                    ok = false;
                }
            }
            // ...
            // 把键值对读入，总共26个，循环中每次读一个
            switch (type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, key, is_array, n); break;
                //...
            }
        }
    }
    // read the tensor info 这里要读取 tensor 的 info 了
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        struct gguf_tensor_info info;
        // ...
        ggml_set_name(&info.t, name.c_str());  // 把名字付给 tensor
        // ...
    }
    // tensor shape
    {
        uint32_t n_dims = -1;
        ok = ok && gr.read(n_dims);
        // ...
    }
    // ...
    // 到这里 tensor 的名字和维度还有类型已经读取完了 

    // ...
    // compute the total size of the data section, taking into account the alignment
    // 这里作了一个操作：将所有的Pad过后的align好的 tensor 的字节 累加，记录到 ctx中
    {
        ctx->size = 0;
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const gguf_tensor_info & ti = ctx->info[i];
            if (ti.offset != ctx->size) {
                fprintf(stderr, "%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                fprintf(stderr, "%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
            ctx->size += GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
        }
    }
    // 只有有需要时，才load tensor data
    if (params.ctx != nullptr) {
        // gguf_context 里面指明了是 no_alloc 时,只创建 "empty" tensors 不 read binary blob。否则 load 进 binary blob 到 ggml_context 之后 将  ggml_tensor 结构体与 binary blob 中的相应位置联系起来指明
        const size_t mem_size =
            params.no_alloc ?
            (n_tensors    )*ggml_tensor_overhead() :
            (n_tensors + 1)*ggml_tensor_overhead() + ctx->size;
        // 这里对应两种方式，第一种是backend只存多个object+tensor头
        //（图见下面章节“gpubackend 和 cpu 模式下的tensor 内存分配情况”https://www.yuque.com/huangyuxiang-8hx5j/wb31rp/lv9gnrqmoz2yyp07#bJo3x）
        // 第二种是cpu形式，存一个object+tensor头和weight data，以及多个 object+tensor头
        // （图见下面章节“gpubackend 和 cpu 模式下的tensor 内存分配情况”https://www.yuque.com/huangyuxiang-8hx5j/wb31rp/lv9gnrqmoz2yyp07#bJo3x）

        struct ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };
        // 注意这里 ggml_init 逻辑与ggml项目中的一样，
        // 但是这里llamacpp工程中不同点是其中没有创建 g_state 这个全局变量了
        *params.ctx = ggml_init(pdata);
        // 到这里是把 context 创建完
        // ...
        // 下面要把tensor 都放进去 
        // 我们只考虑 gpu backend 的场景：
        ggml_set_no_alloc(ctx_data, true); // 由于gpu backend，所以no_alloc 为True

        // create the tensors
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const struct gguf_tensor_info & info = ctx->info[i];
            // 这里要把刚刚创建的 context 传进去，来创建这个 tensor
            struct ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);
            // ...
        }
        // 可以想到，执行219遍创建tensor,连接入context，
        // 下面章节“gpubackend 和 cpu 模式下的tensor 内存分配情况”中那个219长度的链表就创建出来了
```
其中初始化的 gguf_context 是这样的（其中包含kv）：<br>
```cpp
struct gguf_context {
uint32_t version = GGUF_VERSION;

std::vector<struct gguf_kv> kv;
std::vector<struct gguf_tensor_info> info;

size_t alignment = GGUF_DEFAULT_ALIGNMENT;
size_t offset    = 0; // offset of `data` from beginning of file
size_t size      = 0; // size of `data` in bytes

void * data = nullptr;
};

// 其中 gguf_kv 的定义是这样的
struct gguf_kv {
std::string key;

bool is_array;
enum gguf_type type;	// 这个定义了下面这个 data 是以什么形式解读的
// 比如视频例子中 是GGUF_TYPE_INT32（4） 
// 意思是 data中虽然存了4个 int8类型，但是要按 int32来解析
std::vector<int8_t>      data;
std::vector<std::string> data_string;
// ...

}
```
读取 tensor info 时的 gguf_tensor_info 结构是这样的：<br>
gguf_tensor_info <br>
```cpp
struct gguf_tensor_info {
struct ggml_tensor t; // for holding the equivalent info
uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};
```
#### gpubackend 和 cpu 模式下的tensor 内存分配情况
这里还是要分清一下cpu backend 和 gpu backend 情况下的 内存/显存分配方式的不同:<br>
cpu only：比gpu模式，多放一个头+data<br>
<img src="./llamacpp/ggml_4.png" alt="引用图" width="992" height="70"><br>
gpubackend：<br>
<img src="./llamacpp/ggml_5.png" alt="引用图" width="365" height="82"><br>
读完219 个tensor放在context里面的时候（链表）：<br>
<img src="./llamacpp/ggml_6.png" alt="引用图" width="992" height="70"><br>

#### load_hparams 函数
llama_model_loader 创建完成之后会调用 load_hparams 函数，这个函数会将ml中的kv对 读到 hparams 中<br>

#### load_tensors 函数
load_hparams 之后会调用 load_tensors 函数。十分重要，单独分析:<br>
llama_model_load 中的 load_tensors 函数 ， 用来将ml 中的 context 中的内容读到 llama_model 和 gpu中<br>
有关cpu gpu 的 buf list,我看过一些博文，cpu部分的buflist 确实会创建3个，第一个是cpu的dev，第二个是gpu的dev，第三个又是cpu的dev，不清楚这样的设计是为了规避什么，但是这里执行时在有gpu的情况下，会不用第一个cpu的dev的buflist。:<br>
<img src="./llamacpp/ggml_7.png" alt="引用图" width="980" height="410"><br>
源码：<br>
```cpp
// 这里注意，我把 llama_model 的成员 impl 的结构放出来， 这里有个cpu_buft_list， 还有
// 一个map类型的 gpu_buft_list
struct llama_model::impl {
    impl() {}
    ~impl() {}
    // ...
    buft_list_t cpu_buft_list;
    // 这是因为 cpu只有一个，但是gpu可能有多个，因此需要用map记录分配
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list; 
    // 从这里也能看出，layer_dev 中，因为可能layer是放在不同device gpu上的，
    // 因此这里也带了一个 ggml_backend_dev_t 用于记录需要放到的device
    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };
    // ...
}

bool llama_model::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & n_gpu_layers = params.n_gpu_layers;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer = hparams.n_layer;	// 共有24个layer

    const bool use_mmap_buffer = true;      // 是否使用内存映射
    
    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices);
    for (auto * dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev, std::move(buft_list));
    }
```

