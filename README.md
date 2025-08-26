> [!NOTE]
> 本项目由 Claude 4 Sonnet 编写。

# LLM API Recorder

一个用于 LLM API 请求中转的服务器，支持请求记录、Web 界面展示、全文搜索和流式请求转发。

## 功能特性

- 🔄 **API请求中转**: 支持将请求转发到指定的 LLM API 服务器
- 📊 **请求记录**: 记录所有请求的详细信息（时间、内容、响应、token 使用等）
- 🌐 **Web界面**: 美观的Web界面展示请求历史和统计信息
- 🔍 **全文搜索**: 支持对请求内容、响应内容、模型等进行全文搜索
- 🌊 **流式支持**: 完整支持流式请求的转发和记录
- 📈 **统计分析**: 实时显示请求数量、成功率、token 使用等统计信息

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置设置

编辑 `config.ini` 文件，设置目标API服务器：

### 3. 启动服务器

```bash
python main.py
```

或直接使用uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问服务

- **Web界面**: http://localhost:8000
- **API端点**: http://localhost:8000/v1/chat/completions

## 使用方法

### API调用

将你的LLM客户端的API地址指向中转服务器：

```python
import openai

client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"  # 指向中转服务器
)

# 正常使用，请求会被自动记录和转发
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)
```

### 流式请求

```python
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "讲一个故事"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### 获取模型列表

```python
models = client.models.list()
print("可用模型:", [model.id for model in models.data])
```

### 文本嵌入

```python
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="要计算嵌入向量的文本"
)
embedding = response.data[0].embedding
```

### Web界面功能

1. **请求历史**: 查看所有API请求的详细记录
2. **统计信息**: 查看总请求数、成功率、token使用量等
3. **搜索功能**: 搜索请求内容、模型名称、错误信息等
4. **请求详情**: 点击任意请求查看完整的请求和响应内容
5. **分页浏览**: 高效浏览大量请求记录

## 项目结构

```
LLM-API-Recorder/
├── main.py              # 主应用文件
├── config.py            # 配置管理
├── database.py          # 数据库模型和操作
├── utils.py             # 工具函数
├── config.ini           # 配置文件
├── requirements.txt     # Python依赖
├── templates/           # HTML模板
│   └── dashboard.html   # 主界面模板
└── README.md           # 项目说明
```

## 配置说明

### LLM配置
- `api_base_url`: 目标LLM API服务器地址

### 服务器配置
- `host`: 服务器监听地址
- `port`: 服务器端口
- `debug`: 是否启用调试模式
- `access_password`: Web界面访问密码（留空则不需要密码）
- `web_title`: Web界面标题

### 数据库配置
- `db_path`: SQLite数据库文件路径

## API接口

### 主要端点

- `GET /`: Web界面主页
- `POST /v1/chat/completions`: 聊天完成API（兼容OpenAI格式）
- `POST /v1/completions`: 文本完成API（兼容OpenAI格式）
- `POST /v1/embeddings`: 嵌入向量API（兼容OpenAI格式）
- `GET /v1/models`: 获取模型列表API（兼容OpenAI格式）
- `GET /api/requests/{id}`: 获取特定请求的详细信息

### 查询参数

Web界面支持以下查询参数：
- `page`: 页码（默认为1）
- `search`: 搜索关键词

## 数据存储

使用SQLite数据库存储请求记录，包含以下字段：

- `id`: 请求ID
- `timestamp`: 请求时间
- `model`: 使用的模型
- `app_name`: 请求头中的 X-Title 参数
- `request_content`: 请求内容（JSON）
- `response_content`: 响应内容（JSON）
- `input_tokens`: 输入token数量
- `output_tokens`: 输出token数量
- `total_tokens`: 总token数量
- `duration`: 请求耗时
- `status_code`: HTTP状态码
- `error_message`: 错误信息（如有）

## 开发说明

### 添加新的API端点

在 `main.py` 中添加新的路由：

```python
@app.post("/v1/your-endpoint")
async def your_endpoint(request: Request):
    # 实现逻辑
    pass
```

### 自定义token计数

修改 `utils.py` 中的 `count_tokens_estimate` 函数：

```python
def count_tokens_estimate(text: str) -> int:
    # 使用更精确的token计数库，如tiktoken
    pass
```

### 扩展数据库模型

在 `database.py` 中修改 `APIRequest` 模型：

```python
class APIRequest(Base):
    # 添加新字段
    new_field = Column(String(100))
```

## 注意事项

1. **安全性**: 确保API密钥安全，不要在公网暴露服务器
2. **性能**: 对于高并发场景，考虑使用PostgreSQL替代SQLite
3. **监控**: 生产环境建议添加日志和监控
4. **备份**: 定期备份数据库文件

## 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **数据库权限问题**
   确保应用有写入数据库文件的权限

3. **API请求失败**
   检查目标API服务器地址和密钥是否正确

4. **流式请求中断**
   检查网络连接和目标服务器是否支持流式响应