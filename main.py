from fastapi import FastAPI, Request, HTTPException, Query, Depends, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import time
import logging
import random
from typing import Optional, Dict, Any, List
import asyncio
from contextlib import asynccontextmanager
from pydantic import BaseModel

from config import config
from database import db, APIRequest
from utils import (
    extract_tokens_from_request, 
    extract_tokens_from_response, 
    clean_response_for_storage,
    clean_request_for_storage,
    format_duration,
    format_tokens,
    truncate_text
)


class SearchRequest(BaseModel):
    """搜索请求模型"""
    page: int = 1
    search: Optional[str] = None
    model_filter: Optional[str] = None
    app_filter: Optional[str] = None
    status_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_tokens: Optional[str] = None
    max_tokens: Optional[str] = None
    min_duration: Optional[str] = None
    max_duration: Optional[str] = None
    search_field: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None
    id_from: Optional[str] = None
    id_to: Optional[str] = None
    search_mode: Optional[str] = None
    cond: Optional[str] = None

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
                   status_filter: str = Query(None), date_from: str = Query(None),
                   date_to: str = Query(None), min_tokens: str = Query(None),
                   max_tokens: str = Query(None), min_duration: str = Query(None),
                   max_duration: str = Query(None), search_field: str = Query(None),
                   sort_by: str = Query(None), sort_order: str = Query(None),
                   id_from: str = Query(None), id_to: str = Query(None),
                   search_mode: str = Query(None),
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
            # 安全转换数值参数
            min_tokens_int = safe_int(min_tokens)
            max_tokens_int = safe_int(max_tokens)
            min_duration_float = safe_float(min_duration)
            max_duration_float = safe_float(max_duration)
            id_from_int = safe_int(id_from)
            id_to_int = safe_int(id_to)
            
            # 使用优化的合并查询方法，一次获取所有数据
            data = db.get_requests_with_stats(
                page=page, page_size=20, search=search,
                model_filter=model_filter, app_filter=app_filter,
                status_filter=status_filter, date_from=date_from,
                date_to=date_to, min_tokens=min_tokens_int,
                max_tokens=max_tokens_int, min_duration=min_duration_float,
                max_duration=max_duration_float, search_field=search_field,
                sort_by=sort_by, sort_order=sort_order,
                id_from=id_from_int, id_to=id_to_int, search_mode=search_mode
            )
            
            result = data['result']
            stats = data['stats']
            all_models = data['all_models']
            all_apps = data['all_apps']
            
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
        
        # 构建筛选条件摘要
        filter_summary = build_filter_summary(
            search=search, model_filter=model_filter, app_filter=app_filter,
            status_filter=status_filter, date_from=date_from, date_to=date_to,
            min_tokens=min_tokens, max_tokens=max_tokens,
            min_duration=min_duration, max_duration=max_duration, search_field=search_field,
            sort_by=sort_by, id_from=id_from, id_to=id_to, search_mode=search_mode
        )
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "result": result,
            "stats": stats,
            "all_models": all_models,
            "all_apps": all_apps,
            "search": search or "",
            "model_filter": model_filter or "",
            "app_filter": app_filter or "",
            "status_filter": status_filter or "",
            "date_from": date_from or "",
            "date_to": date_to or "",
            "min_tokens": min_tokens or "",
            "max_tokens": max_tokens or "",
            "min_duration": min_duration or "",
            "max_duration": max_duration or "",
            "search_field": search_field or "all",
            "sort_by": sort_by or "time",
            "sort_order": sort_order or "desc",
            "id_from": id_from or "",
            "id_to": id_to or "",
            "search_mode": search_mode or "and",
            "current_page": page,
            "web_title": config.web_title,
            "skip_list": skip_list,
            "initial_request_id": request_id,
            "lazy": lazy,
            "load_time_ms": load_time_ms,
            "estimated_next_ms": estimated_next_ms,
            "filter_summary": filter_summary
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def safe_int(value) -> Optional[int]:
    """安全转换为整数，空字符串或无效值返回None"""
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def safe_float(value) -> Optional[float]:
    """安全转换为浮点数，空字符串或无效值返回None"""
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_conditions(cond_str: str) -> Optional[list]:
    """解析高级条件参数，支持base64编码和直接JSON"""
    if not cond_str:
        return None
    
    import base64
    from urllib.parse import unquote
    
    # 首先尝试base64解码
    try:
        decoded = base64.b64decode(cond_str).decode('utf-8')
        decoded = unquote(decoded)
        result = json.loads(decoded)
        if isinstance(result, list):
            return result
    except:
        pass
    
    # 尝试直接JSON解析
    try:
        result = json.loads(cond_str)
        if isinstance(result, list):
            return result
    except:
        pass
    
    # 尝试URL解码后JSON解析
    try:
        result = json.loads(unquote(cond_str))
        if isinstance(result, list):
            return result
    except:
        pass
    
    return None


def build_filter_summary(search=None, model_filter=None, app_filter=None,
                        status_filter=None, date_from=None, date_to=None,
                        min_tokens=None, max_tokens=None, min_duration=None,
                        max_duration=None, search_field=None, sort_by=None,
                        id_from=None, id_to=None, search_mode=None, conditions=None):
    """构建筛选条件的摘要文本"""
    parts = []
    
    if search:
        field_names = {
            'all': '全部字段',
            'request': '请求内容',
            'response': '响应内容',
            'error': '错误信息'
        }
        field_name = field_names.get(search_field, '全部字段')
        mode_name = 'AND' if search_mode == 'and' else 'OR'
        if ' ' in search or '-' in search:
            parts.append(f'搜索"{search}"于{field_name}({mode_name}模式)')
        else:
            parts.append(f'搜索"{search}"于{field_name}')
    
    if model_filter:
        parts.append(f'模型: {model_filter}')
    
    if app_filter:
        parts.append(f'应用: {app_filter}')
    
    if status_filter:
        status_names = {'success': '成功', 'error': '失败', 'pending': '处理中'}
        parts.append(f'状态: {status_names.get(status_filter, status_filter)}')
    
    if date_from or date_to:
        if date_from and date_to:
            parts.append(f'时间: {date_from} 至 {date_to}')
        elif date_from:
            parts.append(f'时间: {date_from} 起')
        else:
            parts.append(f'时间: 至 {date_to}')
    
    if min_tokens or max_tokens:
        if min_tokens and max_tokens:
            parts.append(f'Token: {min_tokens}-{max_tokens}')
        elif min_tokens:
            parts.append(f'Token: ≥{min_tokens}')
        else:
            parts.append(f'Token: ≤{max_tokens}')
    
    if min_duration or max_duration:
        if min_duration and max_duration:
            parts.append(f'耗时: {min_duration}s-{max_duration}s')
        elif min_duration:
            parts.append(f'耗时: ≥{min_duration}s')
        else:
            parts.append(f'耗时: ≤{max_duration}s')
    
    if id_from or id_to:
        if id_from and id_to:
            parts.append(f'ID: {id_from}-{id_to}')
        elif id_from:
            parts.append(f'ID: ≥{id_from}')
        else:
            parts.append(f'ID: ≤{id_to}')
    
    if sort_by and sort_by != 'time':
        sort_names = {'tokens': 'Token数', 'duration': '耗时', 'time': '时间'}
        parts.append(f'排序: {sort_names.get(sort_by, sort_by)}')
    
    # 高级条件摘要
    if conditions and isinstance(conditions, list):
        for cond in conditions:
            field = cond.get('field', '')
            operator = cond.get('operator', '')
            value = cond.get('value', '')
            role = cond.get('role', '')
            negate = cond.get('negate', False)
            
            if field == 'role':
                op_text = '包含' if operator == 'exists' else '不包含'
                role_names = {'system': '系统消息', 'user': '用户消息', 'assistant': '助手消息', 'tool': '工具消息'}
                parts.append(f'{op_text}{role_names.get(value, value)}')
            
            elif field == 'role_content':
                role_names = {'system': '系统', 'user': '用户', 'assistant': '助手', 'tool': '工具'}
                op_text = '包含' if operator == 'contains' else '不包含'
                parts.append(f'{role_names.get(role, role)}消息{op_text}"{value}"')
            
            elif field == 'any_content':
                op_text = '包含' if operator == 'contains' else '不包含'
                parts.append(f'内容{op_text}"{value}"')
            
            elif field == 'has_feature':
                feature_names = {
                    'tools': '工具定义', 'tool_calls': '工具调用', 'images': '图片',
                    'functions': '函数', 'stream': '流式', 'error': '错误',
                    'system_prompt': '系统提示词'
                }
                prefix = '无' if negate else '有'
                parts.append(f'{prefix}{feature_names.get(value, value)}')
            
            elif field == 'model_contains':
                op_text = '包含' if operator == 'contains' else '不包含'
                parts.append(f'模型名{op_text}"{value}"')
            
            elif field == 'app_contains':
                op_text = '包含' if operator == 'contains' else '不包含'
                parts.append(f'应用名{op_text}"{value}"')
            
            elif field == 'error_contains':
                op_text = '包含' if operator == 'contains' else '不包含'
                parts.append(f'错误信息{op_text}"{value}"')
            
            elif field == 'param':
                param_name = cond.get('param_name', '')
                if operator == 'exists':
                    parts.append(f'有参数{param_name}')
                elif operator == 'not_exists':
                    parts.append(f'无参数{param_name}')
                else:
                    parts.append(f'参数{param_name}={value}')
    
    return ' | '.join(parts) if parts else None

@app.get("/api/requests")
async def get_requests_api(page: int = Query(1, ge=1), search: str = Query(None),
                          model_filter: str = Query(None), app_filter: str = Query(None),
                          status_filter: str = Query(None), date_from: str = Query(None),
                          date_to: str = Query(None), min_tokens: str = Query(None),
                          max_tokens: str = Query(None), min_duration: str = Query(None),
                          max_duration: str = Query(None), search_field: str = Query(None),
                          sort_by: str = Query(None), sort_order: str = Query(None),
                          id_from: str = Query(None), id_to: str = Query(None),
                          search_mode: str = Query(None), cond: str = Query(None),
                          authenticated: bool = Depends(verify_password)):
    """获取请求列表API (GET)"""
    return await _search_requests(
        page=page, search=search, model_filter=model_filter, app_filter=app_filter,
        status_filter=status_filter, date_from=date_from, date_to=date_to,
        min_tokens=min_tokens, max_tokens=max_tokens, min_duration=min_duration,
        max_duration=max_duration, search_field=search_field, sort_by=sort_by,
        sort_order=sort_order, id_from=id_from, id_to=id_to, search_mode=search_mode,
        cond=cond
    )


@app.post("/search")
async def post_search(request: Request, authenticated: bool = Depends(verify_password)):
    """POST搜索接口 - 支持表单提交"""
    form_data = await request.form()
    
    return await _search_requests(
        page=safe_int(form_data.get('page')) or 1,
        search=form_data.get('search') or None,
        model_filter=form_data.get('model_filter') or None,
        app_filter=form_data.get('app_filter') or None,
        status_filter=form_data.get('status_filter') or None,
        date_from=form_data.get('date_from') or None,
        date_to=form_data.get('date_to') or None,
        min_tokens=form_data.get('min_tokens') or None,
        max_tokens=form_data.get('max_tokens') or None,
        min_duration=form_data.get('min_duration') or None,
        max_duration=form_data.get('max_duration') or None,
        search_field=form_data.get('search_field') or None,
        sort_by=form_data.get('sort_by') or None,
        sort_order=form_data.get('sort_order') or None,
        id_from=form_data.get('id_from') or None,
        id_to=form_data.get('id_to') or None,
        search_mode=form_data.get('search_mode') or None,
        cond=form_data.get('cond') or None
    )


@app.post("/api/search")
async def post_search_api(body: SearchRequest, authenticated: bool = Depends(verify_password)):
    """POST搜索API - 支持JSON请求体"""
    return await _search_requests(
        page=body.page, search=body.search, model_filter=body.model_filter,
        app_filter=body.app_filter, status_filter=body.status_filter,
        date_from=body.date_from, date_to=body.date_to,
        min_tokens=body.min_tokens, max_tokens=body.max_tokens,
        min_duration=body.min_duration, max_duration=body.max_duration,
        search_field=body.search_field, sort_by=body.sort_by,
        sort_order=body.sort_order, id_from=body.id_from, id_to=body.id_to,
        search_mode=body.search_mode, cond=body.cond
    )


async def _search_requests(page=1, search=None, model_filter=None, app_filter=None,
                           status_filter=None, date_from=None, date_to=None,
                           min_tokens=None, max_tokens=None, min_duration=None,
                           max_duration=None, search_field=None, sort_by=None,
                           sort_order=None, id_from=None, id_to=None,
                           search_mode=None, cond=None):
    """搜索请求的核心逻辑"""
    try:
        start_time = time.perf_counter()
        
        # 安全转换数值参数
        min_tokens_int = safe_int(min_tokens)
        max_tokens_int = safe_int(max_tokens)
        min_duration_float = safe_float(min_duration)
        max_duration_float = safe_float(max_duration)
        id_from_int = safe_int(id_from)
        id_to_int = safe_int(id_to)
        
        # 解析高级条件（支持base64编码和直接JSON）
        conditions_list = parse_conditions(cond)
        
        # 使用优化的合并查询方法，一次获取所有数据
        data = db.get_requests_with_stats(
            page=page, page_size=20, search=search,
            model_filter=model_filter, app_filter=app_filter,
            status_filter=status_filter, date_from=date_from,
            date_to=date_to, min_tokens=min_tokens_int,
            max_tokens=max_tokens_int, min_duration=min_duration_float,
            max_duration=max_duration_float, search_field=search_field,
            sort_by=sort_by, sort_order=sort_order,
            id_from=id_from_int, id_to=id_to_int, search_mode=search_mode,
            conditions=conditions_list
        )
        
        result = data['result']
        stats = data['stats']
        all_models = data['all_models']
        all_apps = data['all_apps']
        
        # 格式化数据
        for req in result['requests']:
            req['formatted_duration'] = format_duration(req['duration'] or 0)
            req['formatted_input_tokens'] = format_tokens(req['input_tokens'])
            req['formatted_output_tokens'] = format_tokens(req['output_tokens'])
            req['formatted_total_tokens'] = format_tokens(req['total_tokens'])
            
            # 如果有搜索词，提取匹配的片段
            if search:
                req['match_snippets'] = extract_match_snippets(req, search)

        load_time_ms = (time.perf_counter() - start_time) * 1000
        global last_list_load_ms
        last_list_load_ms = load_time_ms
        estimated_next_ms = get_estimated_load_ms()
        
        # 构建筛选条件摘要
        filter_summary = build_filter_summary(
            search=search, model_filter=model_filter, app_filter=app_filter,
            status_filter=status_filter, date_from=date_from, date_to=date_to,
            min_tokens=min_tokens, max_tokens=max_tokens,
            min_duration=min_duration, max_duration=max_duration, search_field=search_field,
            sort_by=sort_by, id_from=id_from, id_to=id_to, search_mode=search_mode,
            conditions=conditions_list
        )
        
        return {
            "result": result,
            "stats": stats,
            "all_models": all_models,
            "all_apps": all_apps,
            "load_time_ms": load_time_ms,
            "estimated_next_ms": estimated_next_ms,
            "filter_summary": filter_summary,
            "search_keyword": search
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_match_snippets(req: dict, search: str, context_chars: int = 50) -> list:
    """从请求/响应内容中提取匹配搜索词的片段"""
    snippets = []
    search_lower = search.lower()
    
    # 搜索请求内容
    if req.get('request_content'):
        content = json.dumps(req['request_content'], ensure_ascii=False)
        snippets.extend(find_snippets_in_text(content, search_lower, context_chars, 'request'))
    
    # 搜索响应内容
    if req.get('response_content'):
        content = json.dumps(req['response_content'], ensure_ascii=False)
        snippets.extend(find_snippets_in_text(content, search_lower, context_chars, 'response'))
    
    # 搜索错误信息
    if req.get('error_message'):
        snippets.extend(find_snippets_in_text(req['error_message'], search_lower, context_chars, 'error'))
    
    # 最多返回3个片段
    return snippets[:3]


def find_snippets_in_text(text: str, search: str, context_chars: int, source: str) -> list:
    """在文本中查找匹配片段"""
    import re
    snippets = []
    text_lower = text.lower()
    
    # 查找所有匹配位置
    start = 0
    while True:
        pos = text_lower.find(search, start)
        if pos == -1:
            break
        
        # 提取上下文
        snippet_start = max(0, pos - context_chars)
        snippet_end = min(len(text), pos + len(search) + context_chars)
        
        snippet = text[snippet_start:snippet_end]
        
        # 添加省略号
        if snippet_start > 0:
            snippet = '...' + snippet
        if snippet_end < len(text):
            snippet = snippet + '...'
        
        snippets.append({
            'source': source,
            'text': snippet,
            'match_start': pos - snippet_start + (3 if snippet_start > 0 else 0),
            'match_length': len(search)
        })
        
        start = pos + 1
        
        # 最多找3个匹配
        if len(snippets) >= 3:
            break
    
    return snippets

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
            app_name=app_name,
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
        if request_id:
            # 更新请求记录
            try:
                session = db.get_session()
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    request_record.duration = duration
                    request_record.status_code = 500
                    request_record.error_message = str(e)
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
            app_name=app_name,
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
                session.commit()
        finally:
            session.close()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # 记录错误
        duration = time.time() - start_time
        if request_id:
            try:
                session = db.get_session()
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    request_record.duration = duration
                    request_record.status_code = 500
                    request_record.error_message = str(e)
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
            app_name=app_name,
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
        if request_id:
            # 更新请求记录
            try:
                session = db.get_session()
                request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
                if request_record:
                    request_record.duration = duration
                    request_record.status_code = 500
                    request_record.error_message = str(e)
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