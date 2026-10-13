# 私有知识库 RAG + Agent 问答系统

一个基于 **Streamlit + DeepSeek** 的问答系统，包含两个子系统：

- **RAG 检索链路** —— 文档知识库、语义向量检索、rerank 精排、答案可溯源
- **Agent 工具调度链路** —— 由模型自主决定是否调用工具（知识库检索 / 数学计算 / 当前时间查询）

## ✨ 功能特性

- 多会话流式对话（支持自定义昵称与人格设定）
- 上传文档（txt / md / pdf / docx）构建私有知识库
- 语义向量检索（BGE 中文 embedding + 余弦相似度）
- Rerank 重排（bge-reranker 精排，提升检索精度）
- 答案可溯源（回答下方展示命中的知识库片段）
- 多轮对话记忆（历史发言向量化，检索相关记忆融入回答）
- **Agent 工具调用** —— 模型自主决定调不调工具、调哪个，支持一次提问并行调用多个工具
- **工具调用轨迹可见** —— 界面可折叠面板 + JSONL 结构化落盘，决策链路全程可复盘
- **双对话模式** —— 侧边栏可切换「知识库模式」（每问必检索）与「Agent 模式」（按需调用）
- **HTTP 接口服务** —— 以 FastAPI 暴露 `POST /chat`、`POST /knowledge` 等接口，同一套调度逻辑可供任意前端调用

## 🧠 核心流程

### ① RAG 检索链路（知识库模式）

```
上传文档 → 解析 → 切块 → 向量化 → 存入知识库
提问 → 向量化 → 相似度检索 → rerank 精排 → 挑出最相关片段
片段注入提示词 → DeepSeek 生成回答 → 流式显示 + 展示引用来源
```

### ② Agent 工具调度链路（Agent 模式）

```
下发工具声明 → 模型判断是否需要工具
  ├─ 不需要 → 直接生成回答（不检索、不调工具）
  └─ 需要   → 模型返回结构化调用请求
              → 本地代码执行工具
              → 结果回填上下文 → 再交给模型
              → 循环，直到模型给出最终答复
```

> **关键点：执行工具的始终是本地 Python 代码，模型只输出"调用请求"文本。**
> 单轮最多允许 5 次循环（`MAX_ROUNDS` 护栏），防止模型不收敛导致请求失控。

## 🔧 Agent 工具清单

| 工具               | 作用                         | 说明                                  |
|--------------------|------------------------------|---------------------------------------|
| `search_knowledge` | 检索私有知识库的相关原文片段 | 复用 RAG 检索链路，返回原文而非答案   |
| `calculate`        | 四则运算计算                 | 正则白名单 + 剥离内置函数，防代码注入 |
| `get_current_time` | 获取当前日期与时间           | 模型无法凭记忆回答"今天几号"          |

**工具的两张表**（职责分离）：

- `TOOL_SCHEMAS` —— **声明表**，给模型看。只描述工具用途，模型据此判断该不该调、调哪个。
- `TOOLS` —— **注册表**，给代码用。工具名 → 真实函数的映射。

工具名、Python 参数名、Schema 键名三处强制保持一致。

## 🛡 安全设计

`calculate` 若直接对模型生成的表达式求值，模型可能被诱导产出读写文件、执行系统命令的恶意载荷。采用三级防护：

1. **正则整串白名单** —— `re.fullmatch(r"[0-9+\-*/().\s]+", expr)`，只放行数字与 `+ - * / ( )`
2. **剥离全部内置函数** —— `eval(expr, {"__builtins__": {}}, {})`
3. **异常兜底** —— 捕获异常并返回可读错误字符串，不中断主循环

实测可拦截 `__import__("os").system(...)` 类注入载荷。

## 📈 可观测性

每次工具调用都会结构化为一行 JSON 追加写入 `agent_trace.jsonl`（轮次 / 工具名 / 参数 / 结果），并在界面回答下方的可折叠面板中实时展示。

```jsonl
{"time": "2026-10-12 00:39:09", "question": "现在几点了？顺便帮我算一下 (1234+5678)*3 等于多少。", "round": 1, "tool": "get_current_time", "args": {}, "result": "2026-10-12 00:39:08"}
{"time": "2026-10-12 00:39:09", "question": "现在几点了？顺便帮我算一下 (1234+5678)*3 等于多少。", "round": 1, "tool": "calculate", "args": {"expression": "(1234+5678)*3"}, "result": "20736"}
```

选 JSONL 是因为它是**追加式**格式：可以逐行 append，无需把整个文件读进内存重写。

## 🛠 技术栈

| 层        | 技术                       |
|-----------|----------------------------|
| 界面      | Streamlit                  |
| 大模型    | DeepSeek API（OpenAI SDK） |
| 工具调用  | DeepSeek Function Calling  |
| Embedding | BAAI/bge-small-zh-v1.5     |
| Rerank    | BAAI/bge-reranker-base     |
| 相似度    | numpy（余弦相似度）        |
| 轨迹落盘  | JSONL（追加式结构化日志）  |
| 接口服务  | FastAPI + Uvicorn          |

## 📊 评测结果

自建 10 题测试集，检索命中率（Top3）：

| 检索方法      | 命中率   |
|---------------|----------|
| 关键词检索    | 80%      |
| 向量检索      | **100%** |
| 向量 + rerank | **100%** |

## 🚀 安装与运行

```bash
# 1. 安装依赖（sentence-transformers 会一并安装 PyTorch，体积较大，请耐心等待）
pip install -r requirements.txt

# 2. 下载模型（走国内 ModelScope 镜像，合计约 500MB）
python download_model.py       # 下载 embedding 模型
python download_reranker.py    # 下载 rerank 模型

# 3. 配置 DeepSeek API Key（环境变量）
#    Windows:  set DEEPSEEK_API_KEY=你的key
#    其他系统: export DEEPSEEK_API_KEY=你的key
#    变量名可参考 .env.example

# 4. 运行 Web 界面
streamlit run app.py

# 5. 或运行 HTTP 接口服务（FastAPI）
uvicorn api:app --host 127.0.0.1 --port 8000
```

运行后在侧边栏「对话模式」中切换：

- **知识库模式** —— 每次提问都先检索知识库（原有行为）
- **Agent 模式** —— 模型自己判断要不要检索、要不要算数、要不要看时间

### HTTP 接口

`uvicorn` 启动后访问 `http://127.0.0.1:8000/docs` 可打开自动生成的交互式 API 文档，直接在页面上调用接口。

| 方法     | 路径         | 说明                                        |
|----------|--------------|---------------------------------------------|
| `GET`    | `/health`    | 健康检查                                    |
| `POST`   | `/chat`      | 执行一轮 Agent 调度，返回答复与工具调用轨迹 |
| `POST`   | `/knowledge` | 上传文本资料，切块向量化后加入服务端知识库  |
| `GET`    | `/knowledge` | 查看知识库当前状态                          |
| `DELETE` | `/knowledge` | 清空知识库                                  |

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "帮我算一下 (1234+5678)*3"}'
```

> 说明：HTTP 模式下知识库存于服务端进程内存（`tools.py` 的 `_KNOWLEDGE`），重启服务即清空；界面模式仍使用独立的会话状态。

## ✅ 测试

```bash
pytest
```

全部用例**不联网、不加载 embedding 模型**，约 4 秒跑完。

| 测试文件              | 覆盖内容                                                                                      |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `tests/test_tools.py` | 四则运算、代码注入载荷拦截、异常兜底、工具名在两张表间的一致性、知识库存取                    |
| `tests/test_agent.py` | 无工具直答、多轮工具调用、上下文消息顺序、未知工具、工具异常、`MAX_ROUNDS` 主动停止、轨迹落盘 |
| `tests/test_api.py`   | `/health`；`/chat` 的正常分支与 422 / 502 / 503 错误分支；`/knowledge` 的增删查               |

## 🐳 Docker 运行（可选）

```bash
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
docker build -t rag-agent .
docker run --rm -p 8000:8000 --env-file .env rag-agent
```

首次构建需安装依赖并下载约 500MB 模型，耗时较长。若本地已有 `models/`，可加 `--build-arg DOWNLOAD_MODELS=false` 跳过下载，改用 `-v "$(pwd)/models:/app/models"` 挂载。

> 注：Docker 配置为最近补充，尚未在作者本机完成整轮构建验证。

## 🧩 架构分层

调度逻辑层（`agent.py`）**不依赖任何界面框架**：

```
app.py（界面层）    run_agent_turn()  → 渲染回答 / 存会话 / 落盘 / 轨迹面板
api.py（接口层）    chat()            → 校验请求 / 返回 JSON
                            ↓ 两者调用同一个函数
agent.py（逻辑层）  run_agent()       → 多轮循环 / 派发工具 / 两级兜底
                            ↓ 调用
tools.py（工具层）  TOOLS / TOOL_SCHEMAS / 知识库来源
```

因为逻辑层不引用 Streamlit，所以同一套调度逻辑：

- 可在终端直接单元测试（无需启动界面）
- 已被 Streamlit 界面与 FastAPI 接口同时复用，接入时未改动 `agent.py` 一行

## 📁 目录结构

```
ai/
├── app.py                    # 程序入口（页面配置 + AI 调用 + 主程序）
├── config.py                 # 所有常量
├── session.py                # 会话管理
├── rag.py                    # RAG：模型加载 + 切块 + 检索 + 注入
├── ui.py                     # 界面（聊天记录、知识库面板、侧边栏）
├── api.py                    # FastAPI 服务：HTTP 接口（/chat、/knowledge）
├── evaluate.py               # 评测脚本（对比三种检索的命中率）
├── agent.py                  # Agent 主循环（工具调度 + 轨迹落盘）
├── tools.py                  # Agent 工具：函数本体 + 注册表 + 声明表
├── download_model.py         # 下载 embedding 模型
├── download_reranker.py      # 下载 rerank 模型
├── tests/                    # 单元测试（pytest）
│   ├── test_tools.py         #   工具行为 + 安全防护 + 注册表一致性
│   ├── test_agent.py         #   调度循环 + 兜底路径 + 轨迹落盘
│   └── test_api.py           #   HTTP 接口状态码与校验
├── pytest.ini                # pytest 配置
├── Dockerfile                # 容器构建
├── .dockerignore
├── .env.example              # 环境变量模板
├── requirements.txt          # 依赖清单
├── models/                   # 模型文件（下载后生成）
├── sessions/                 # 会话记录（运行后生成）
├── agent_trace.jsonl         # 工具调用轨迹（运行后生成，已 gitignore）
└── resources/                # logo 图片
```
