from fastapi import FastAPI, Request, HTTPException, Query, Depends, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import time
import logging
import random
from typing import Optional, Dict, Any
import asyncio
from contextlib import asynccontextmanager

from config import config
from database import DatabaseFilter, db, APIRequest
from utils import (
    extract_tokens_from_request, 
    extract_tokens_from_response, 
    clean_response_for_storage,
    clean_request_for_storage,
    format_duration,
    format_tokens,
    truncate_text
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="LLM API Recorder", description="LLM API请求中转服务器")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 设置模板（启用空白控制，减少HTML输出中的多余空行）
templates = Jinja2Templates(directory="templates")
templates.env.trim_blocks = True
templates.env.lstrip_blocks = True

# HTTP基本认证
security = HTTPBasic()

last_list_load_ms: Optional[float] = None


def get_estimated_load_ms() -> float:
    """Return previous load duration plus random network jitter."""
    base = last_list_load_ms if last_list_load_ms else 800.0
    jitter = random.randint(500, 1000)
    return base + jitter

def verify_password(credentials: HTTPBasicCredentials = Depends(security)):
    """验证访问密码"""
    if not config.access_password:
        return True  # 如果没有设置密码，则不需要验证
    
    if credentials.password != config.access_password:
        raise HTTPException(
            status_code=401,
            detail="访问密码错误",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, page: int = Query(1, ge=1), search: str = Query(None),
                   model_filter: str = Query(None), app_filter: str = Query(None),
                   request_id: int = Query(None, alias="request"),
                   lazy: bool = Query(True),
                   authenticated: bool = Depends(verify_password)):
    """主页面 - 显示请求记录"""
    try:
        start_time = time.perf_counter()
        # 如果有request参数或开启lazy模式，跳过获取请求列表（延迟加载）
        skip_list = request_id is not None or lazy
        
        if skip_list:
            # 返回空列表，前端关闭详情后再加载
            result = {'total': 0, 'page': 1, 'page_size': 20, 'pages': 0, 'requests': []}
            stats = {'total_requests': 0, 'total_input_tokens': 0, 'total_output_tokens': 0, 'total_tokens': 0, 'avg_duration': 0}
            all_models = []
            all_apps = []
            load_time_ms = 0
        else:
            with DatabaseFilter(db, search=search, model_filter=model_filter, app_filter=app_filter) as df:
                result = df.get_requests(page=page, page_size=20)
                stats = df.get_statistics()
            
            # 获取筛选选项
            all_models = db.get_unique_models()
            all_apps = db.get_unique_apps()
            
            # 格式化数据用于显示
            for req in result['requests']:
                req['formatted_duration'] = format_duration(req['duration'] or 0)
                req['formatted_input_tokens'] = format_tokens(req['input_tokens'])
                req['formatted_output_tokens'] = format_tokens(req['output_tokens'])
                req['formatted_total_tokens'] = format_tokens(req['total_tokens'])
                
                # 截断请求内容用于预览
                if req['request_content'] and req['request_content'].get('messages'):
                    last_message = req['request_content']['messages'][-1]
                    req['preview_content'] = truncate_text(last_message.get('content', ''))
                else:
                    req['preview_content'] = "No content"

            load_time_ms = (time.perf_counter() - start_time) * 1000
            # 记录本次加载时间用于下次预估
            global last_list_load_ms
            last_list_load_ms = load_time_ms

        estimated_next_ms = get_estimated_load_ms()
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "result": result,
            "stats": stats,
            "all_models": all_models,
            "all_apps": all_apps,
            "search": search or "",
            "model_filter": model_filter or "",
            "app_filter": app_filter or "",
            "current_page": page,
            "web_title": config.web_title,
            "skip_list": skip_list,
            "initial_request_id": request_id,
            "lazy": lazy,
            "load_time_ms": load_time_ms,
            "estimated_next_ms": estimated_next_ms
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/requests")
async def get_requests_api(page: int = Query(1, ge=1), search: str = Query(None),
                          model_filter: str = Query(None), app_filter: str = Query(None),
                          authenticated: bool = Depends(verify_password)):
    """获取请求列表API"""
    try:
        start_time = time.perf_counter()
        with DatabaseFilter(db, search=search, model_filter=model_filter, app_filter=app_filter) as df:
            result = df.get_requests(page=page, page_size=20)
            stats = df.get_statistics()
        
        all_models = db.get_unique_models()
        all_apps = db.get_unique_apps()
        
        # 格式化数据
        for req in result['requests']:
            req['formatted_duration'] = format_duration(req['duration'] or 0)
            req['formatted_input_tokens'] = format_tokens(req['input_tokens'])
            req['formatted_output_tokens'] = format_tokens(req['output_tokens'])
            req['formatted_total_tokens'] = format_tokens(req['total_tokens'])

        load_time_ms = (time.perf_counter() - start_time) * 1000
        global last_list_load_ms
        last_list_load_ms = load_time_ms
        estimated_next_ms = get_estimated_load_ms()
        
        return {
            "result": result,
            "stats": stats,
            "all_models": all_models,
            "all_apps": all_apps,
            "load_time_ms": load_time_ms,
            "estimated_next_ms": estimated_next_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/requests/{request_id}")
async def get_request_detail(request_id: int, authenticated: bool = Depends(verify_password)):
    """获取请求详情API"""
    try:
        request_detail = db.get_request_by_id(request_id)
        if not request_detail:
            raise HTTPException(status_code=404, detail="Request not found")
        return request_detail
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def proxy_models(request: Request):
    """代理模型列表API请求"""
    try:
        # 构建目标URL
        target_url = f"{config.llm_api_base_url}/models"
        
        # 设置请求头 - 直接转发客户端的Authorization头
        headers = {}
        
        # 转发客户端的所有相关头部
        for key, value in request.headers.items():
            if key.lower() not in ['host', 'content-length']:
                headers[key] = value
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(target_url, headers=headers)
            return response.json()
    
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def proxy_completions(request: Request):
    """代理文本完成API请求"""
    start_time = time.time()
    request_id = None
    
    try:
        # 读取请求体
        request_body = await request.body()
        request_data = json.loads(request_body)
        
        # 提取模型信息
        model = request_data.get('model', 'unknown')
        
        # 提取应用名称（从X-Title头）
        app_name = request.headers.get('X-Title', None)
        
        # 估算输入tokens
        input_tokens = extract_tokens_from_request(request_data)
        
        # 清理请求数据（替换图片为占位符）
        cleaned_request_data = clean_request_for_storage(request_data)
        
        # 记录请求开始
        request_id = db.add_request(
            model=model,
            app_name=app_name or "",
            request_content=cleaned_request_data,
            input_tokens=input_tokens,
            status_code=0  # 0表示正在处理
        )
        
        # 检查是否为流式请求
        is_streaming = request_data.get('stream', False)
        
        # 构建目标URL
        target_url = f"{config.llm_api_base_url}/completions"
        
        # 设置请求头 - 直接转发客户端的Authorization头
        headers = {
            "Content-Type": "application/json"
        }
        
        # 转发客户端的所有相关头部
        for key, value in request.headers.items():
            if key.lower() not in ['host', 'content-length']:
                headers[key] = value
        
        if is_streaming:
            # 处理流式请求 (类似于chat/completions的流式处理)
            return await handle_streaming_request(
                target_url, headers, request_data, 
                request_id, model, input_tokens, start_time
            )
        else:
            # 处理非流式请求
            async with httpx.AsyncClient(timeout=300.0) as client:
                return await handle_non_streaming_request(
                    client, target_url, headers, request_data,
                    request_id, model, input_tokens, start_time
                )
    
    except Exception as e:
        # 记录错误
        duration = time.time() - start_time
        if request_id is not None:
            # 更新请求记录
            try:
                session = db.get_session()
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    request_record.duration = duration
                    request_record.status_code = 500
                    request_record.error_message = str(e)
                    session.add(request_record)
                    session.commit()
                session.close()
            except:
                pass
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
async def proxy_embeddings(request: Request):
    """代理嵌入向量API请求"""
    start_time = time.time()
    request_id = None
    
    try:
        # 读取请求体
        request_body = await request.body()
        request_data = json.loads(request_body)
        
        # 提取模型信息
        model = request_data.get('model', 'unknown')
        
        # 提取应用名称（从X-Title头）
        app_name = request.headers.get('X-Title', None)
        
        # 估算输入tokens
        input_tokens = extract_tokens_from_request(request_data)
        
        # 清理请求数据（替换图片为占位符）
        cleaned_request_data = clean_request_for_storage(request_data)
        
        # 记录请求开始
        request_id = db.add_request(
            model=model,
            app_name=app_name or "",
            request_content=cleaned_request_data,
            input_tokens=input_tokens,
            status_code=0  # 0表示正在处理
        )
        
        # 构建目标URL
        target_url = f"{config.llm_api_base_url}/embeddings"
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json"
        }
        
        # 转发客户端的所有相关头部
        for key, value in request.headers.items():
            if key.lower() not in ['host', 'content-length']:
                headers[key] = value
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(target_url, json=request_data, headers=headers)
            duration = time.time() - start_time
            
            response_data = response.json()
            
            # 提取token使用信息
            req_input_tokens, output_tokens = extract_tokens_from_response(response_data)
            if req_input_tokens == 0:
                req_input_tokens = input_tokens
            
            # 更新数据库记录
            session = db.get_session()
            try:
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    request_record.response_content = json.dumps(clean_response_for_storage(response_data), ensure_ascii=False)
                    request_record.input_tokens = req_input_tokens
                    request_record.output_tokens = output_tokens
                    request_record.total_tokens = req_input_tokens + output_tokens
                    request_record.duration = duration
                    request_record.status_code = response.status_code
                    session.add(request_record)
                    session.commit()
            finally:
                session.close()
            
            return response_data
    
    except httpx.HTTPError as e:
        duration = time.time() - start_time
        # 更新错误记录
        session = db.get_session()
        try:
            request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
            if request_record:
                request_record.duration = duration
                request_record.status_code = 500
                request_record.error_message = str(e)
                session.add(request_record)
                session.commit()
        finally:
            session.close()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # 记录错误
        duration = time.time() - start_time
        if request_id is not None:
            try:
                session = db.get_session()
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    request_record.duration = duration
                    request_record.status_code = 500
                    request_record.error_message = str(e)
                    session.add(request_record)
                    session.commit()
                session.close()
            except:
                pass
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """代理聊天完成API请求"""
    start_time = time.time()
    request_id = None
    
    try:
        # 读取请求体
        request_body = await request.body()
        request_data = json.loads(request_body)
        
        logger.info(f"收到聊天完成请求: model={request_data.get('model', 'unknown')}")
        
        # 提取模型信息
        model = request_data.get('model', 'unknown')
        
        # 提取应用名称（从X-Title头）
        app_name = request.headers.get('X-Title', None)
        
        # 估算输入tokens
        input_tokens = extract_tokens_from_request(request_data)
        
        # 清理请求数据（替换图片为占位符）
        cleaned_request_data = clean_request_for_storage(request_data)
        
        # 记录请求开始
        request_id = db.add_request(
            model=model,
            app_name=app_name or "",
            request_content=cleaned_request_data,
            input_tokens=input_tokens,
            status_code=0  # 0表示正在处理
        )
        
        # 检查是否为流式请求
        is_streaming = request_data.get('stream', False)
        
        # 构建目标URL
        target_url = f"{config.llm_api_base_url}/chat/completions"
        logger.info(f"转发请求到: {target_url}")
        
        # 设置请求头 - 直接转发客户端的Authorization头
        headers = {
            "Content-Type": "application/json"
        }
        
        # 转发客户端的所有相关头部
        for key, value in request.headers.items():
            if key.lower() not in ['host', 'content-length']:
                headers[key] = value
        
        if is_streaming:
            # 处理流式请求 - 不能在这里使用client context manager
            return await handle_streaming_request(
                target_url, headers, request_data, 
                request_id, model, input_tokens, start_time
            )
        else:
            # 处理非流式请求
            async with httpx.AsyncClient(timeout=300.0) as client:
                return await handle_non_streaming_request(
                    client, target_url, headers, request_data,
                    request_id, model, input_tokens, start_time
                )
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"聊天完成请求处理失败: {e}", exc_info=True)
        # 记录错误
        duration = time.time() - start_time
        if request_id is not None:
            # 更新请求记录
            try:
                session = db.get_session()
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    request_record.duration = duration
                    request_record.status_code = 500
                    request_record.error_message = str(e)
                    session.add(request_record)
                    session.commit()
                session.close()
            except Exception as db_error:
                logger.error(f"数据库更新失败: {db_error}")
        
        raise HTTPException(status_code=500, detail=str(e))

async def handle_non_streaming_request(client, target_url, headers, request_data, 
                                     request_id, model, input_tokens, start_time):
    """处理非流式请求"""
    try:
        logger.info(f"发送非流式请求到: {target_url}")
        response = await client.post(target_url, json=request_data, headers=headers)
        duration = time.time() - start_time
        
        logger.info(f"收到响应: status={response.status_code}, duration={duration:.3f}s")
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"目标服务器返回错误: {response.status_code} - {error_text}")
            raise HTTPException(status_code=response.status_code, detail=error_text)
        
        response_data = response.json()
        
        # 提取token使用信息
        req_input_tokens, output_tokens = extract_tokens_from_response(response_data)
        if req_input_tokens == 0:
            req_input_tokens = input_tokens
        
        # 更新数据库记录
        session = db.get_session()
        try:
            request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
            if request_record:
                request_record.response_content = json.dumps(clean_response_for_storage(response_data), ensure_ascii=False)
                request_record.input_tokens = req_input_tokens
                request_record.output_tokens = output_tokens
                request_record.total_tokens = req_input_tokens + output_tokens
                request_record.duration = duration
                request_record.status_code = response.status_code
                session.add(request_record)
                session.commit()
                logger.info(f"数据库记录更新成功: request_id={request_id}")
        except Exception as db_error:
            logger.error(f"数据库更新失败: {db_error}")
        finally:
            session.close()
        
        return response_data
    
    except httpx.HTTPError as e:
        logger.error(f"HTTP请求错误: {e}")
        duration = time.time() - start_time
        # 更新错误记录
        session = db.get_session()
        try:
            request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
            if request_record:
                request_record.duration = duration
                request_record.status_code = 500
                request_record.error_message = str(e)
                session.add(request_record)
                session.commit()
        except Exception as db_error:
            logger.error(f"数据库错误记录失败: {db_error}")
        finally:
            session.close()
        raise HTTPException(status_code=500, detail=str(e))

async def handle_streaming_request(target_url, headers, request_data,
                                 request_id, model, input_tokens, start_time):
    """处理流式请求"""
    
    async def generate_stream():
        accumulated_content = ""
        output_tokens = 0
        
        try:
            # 在生成器内部创建和管理httpx客户端
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream('POST', target_url, json=request_data, headers=headers) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"流式请求失败: {response.status_code} - {error_text.decode()}")
                        raise HTTPException(status_code=response.status_code, detail=error_text.decode())
                    
                    logger.info("流式响应连接成功")
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            chunk_str = chunk.decode('utf-8')
                            yield chunk
                            
                            # 解析流式响应以提取内容
                            lines = chunk_str.strip().split('\n')
                            for line in lines:
                                if line.startswith('data: '):
                                    try:
                                        data_str = line[6:]  # 移除 'data: ' 前缀
                                        if data_str.strip() == '[DONE]':
                                            continue
                                        
                                        data = json.loads(data_str)
                                        choices = data.get('choices', [])
                                        for choice in choices:
                                            delta = choice.get('delta', {})
                                            content = delta.get('content', '')
                                            if content:
                                                accumulated_content += content
                                                output_tokens += len(content.split())  # 简单估算
                                    except json.JSONDecodeError:
                                        continue
        
        except Exception as e:
            logger.error(f"流式请求处理异常: {e}")
            # 记录流式请求错误
            duration = time.time() - start_time
            session = db.get_session()
            try:
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    request_record.duration = duration
                    request_record.status_code = 500
                    request_record.error_message = str(e)
                    session.add(request_record)
                    session.commit()
            except Exception as db_error:
                logger.error(f"数据库错误记录失败: {db_error}")
            finally:
                session.close()
            raise
        
        finally:
            # 更新流式请求的最终记录
            duration = time.time() - start_time
            session = db.get_session()
            try:
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    # 构建模拟的响应数据用于存储
                    simulated_response = {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": accumulated_content
                            }
                        }],
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens
                        }
                    }
                    
                    request_record.response_content = json.dumps(simulated_response, ensure_ascii=False)
                    request_record.input_tokens = input_tokens
                    request_record.output_tokens = output_tokens
                    request_record.total_tokens = input_tokens + output_tokens
                    request_record.duration = duration
                    request_record.status_code = 200
                    session.add(request_record)
                    session.commit()
                    logger.info(f"流式请求记录更新成功: request_id={request_id}")
            except Exception as db_error:
                logger.error(f"数据库最终更新失败: {db_error}")
            finally:
                session.close()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/plain; charset=utf-8"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.server_host,
        port=config.server_port,
        reload=config.server_debug
    )