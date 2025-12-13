from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, inspect, text, func, and_, or_, not_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import json
import re
import os
from config import config

Base = declarative_base()

# 定义UTC+8时区
UTC8 = timezone(timedelta(hours=8))


def get_current_time():
    """获取当前时间，兼容不同数据库"""
    return datetime.now(UTC8)


class APIRequest(Base):
    """API请求记录表"""
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=get_current_time, index=True)
    model = Column(String(100), index=True)
    app_name = Column(String(100), index=True, nullable=True)  # 应用名称，来自X-Title头
    request_content = Column(Text)  # JSON格式的请求内容
    response_content = Column(Text)  # JSON格式的响应内容
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    duration = Column(Float)  # 请求耗时（秒）
    status_code = Column(Integer)
    error_message = Column(Text, nullable=True)
    
    def to_dict(self, include_content: bool = True):
        """转换为字典格式
        
        Args:
            include_content: 是否包含请求/响应内容，列表查询时可设为False以提高性能
        """
        result = {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'model': self.model,
            'app_name': self.app_name,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'duration': self.duration,
            'status_code': self.status_code,
            'error_message': self.error_message
        }
        
        if include_content:
            result['request_content'] = json.loads(self.request_content) if self.request_content else None
            result['response_content'] = json.loads(self.response_content) if self.response_content else None
        else:
            # 只解析请求内容用于预览
            result['request_content'] = json.loads(self.request_content) if self.request_content else None
            result['response_content'] = None
        
        return result

class Database:
    """数据库管理类，支持 SQLite 和 PostgreSQL"""
    
    def __init__(self, db_url: str = None):
        """
        初始化数据库连接
        
        Args:
            db_url: 数据库连接字符串，支持以下格式：
                - SQLite: sqlite:///path/to/db.db 或直接文件路径
                - PostgreSQL: postgresql://user:password@host:port/dbname
        """
        if db_url is None:
            db_url = config.db_url
        
        # 解析数据库URL
        self.db_type, self.connection_string = self._parse_db_url(db_url)
        
        print(f"数据库类型: {self.db_type}")
        print(f"连接字符串: {self._mask_password(self.connection_string)}")
        
        # 创建数据库引擎，根据数据库类型设置不同参数
        engine_kwargs = {'echo': False}
        
        if self.db_type == 'postgresql':
            # PostgreSQL 连接池配置
            engine_kwargs.update({
                'pool_size': 5,
                'max_overflow': 10,
                'pool_pre_ping': True,  # 连接健康检查
                'pool_recycle': 3600,   # 1小时回收连接
            })
        
        self.engine = create_engine(self.connection_string, **engine_kwargs)
        
        # 创建表（如果不存在）
        try:
            Base.metadata.create_all(bind=self.engine)
            print("数据库连接成功，表结构已同步")
            # 检查并执行数据库升级
            self._upgrade_database()
        except Exception as e:
            print(f"数据库初始化失败: {e}")
            raise
            
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _parse_db_url(self, db_url: str) -> Tuple[str, str]:
        """
        解析数据库URL
        
        Returns:
            (db_type, connection_string) 元组
        """
        # 如果是完整的数据库URL
        if '://' in db_url:
            parsed = urlparse(db_url)
            scheme = parsed.scheme.lower()
            
            if scheme in ('postgresql', 'postgres'):
                # 统一使用 postgresql 协议
                if scheme == 'postgres':
                    db_url = db_url.replace('postgres://', 'postgresql://', 1)
                return 'postgresql', db_url
            elif scheme == 'sqlite':
                return 'sqlite', db_url
            else:
                raise ValueError(f"不支持的数据库类型: {scheme}")
        
        # 如果是文件路径，视为 SQLite 数据库
        db_path = db_url
        
        # 如果是相对路径，使用当前工作目录
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)
        
        # 标准化路径
        db_path = os.path.normpath(db_path)
        
        return 'sqlite', f"sqlite:///{db_path}"
    
    def _mask_password(self, url: str) -> str:
        """隐藏连接字符串中的密码"""
        try:
            parsed = urlparse(url)
            if parsed.password:
                masked = url.replace(f':{parsed.password}@', ':****@')
                return masked
        except:
            pass
        return url
    
    def _upgrade_database(self):
        """检查并升级数据库结构"""
        # 暂时不需要升级功能，直接返回
        return
        
        try:
            # 检查表结构
            inspector = inspect(self.engine)
            if not inspector.has_table('api_requests'):
                return
                
            columns = inspector.get_columns('api_requests')
            column_names = [col['name'] for col in columns]
            
            # 检查是否缺少app_name字段
            if 'app_name' not in column_names:
                print("检测到旧版数据库，正在添加app_name字段...")
                
                # 使用ALTER TABLE添加新字段（兼容SQLite和PostgreSQL）
                with self.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE api_requests ADD COLUMN app_name VARCHAR(100)'))
                    conn.commit()
                
                print("数据库升级完成：已添加app_name字段")
            else:
                print("数据库结构已是最新版本")
                
        except Exception as e:
            print(f"数据库升级检查失败: {e}")
            # 升级失败不影响程序运行，只是记录错误
    
    def get_session(self):
        """获取数据库会话"""
        return self.SessionLocal()
    
    def add_request(self, model: str, request_content: dict, response_content: dict = None, 
                   input_tokens: int = 0, output_tokens: int = 0, duration: float = 0,
                   status_code: int = 200, error_message: str = None, app_name: str = None):
        """添加API请求记录"""
        session = self.get_session()
        try:
            request_record = APIRequest(
                model=model,
                app_name=app_name,
                request_content=json.dumps(request_content, ensure_ascii=False),
                response_content=json.dumps(response_content, ensure_ascii=False) if response_content else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                duration=duration,
                status_code=status_code,
                error_message=error_message
            )
            session.add(request_record)
            session.commit()
            return request_record.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def _build_filter_query(self, query, search: str = None, 
                             model_filter: str = None, app_filter: str = None,
                             status_filter: str = None, date_from: str = None,
                             date_to: str = None, min_tokens: int = None,
                             max_tokens: int = None, min_duration: float = None,
                             max_duration: float = None, search_field: str = None,
                             id_from: int = None, id_to: int = None,
                             search_mode: str = None, conditions: List[Dict] = None):
        """
        构建通用的过滤查询条件
        
        Args:
            query: SQLAlchemy 查询对象
            search: 简单搜索关键词（支持空格分隔多关键词，-前缀排除）
            model_filter: 模型筛选
            app_filter: 应用筛选
            status_filter: 状态筛选 (success/error/pending)
            date_from: 开始日期 (YYYY-MM-DD)
            date_to: 结束日期 (YYYY-MM-DD)
            min_tokens: 最小token数
            max_tokens: 最大token数
            min_duration: 最小耗时（秒）
            max_duration: 最大耗时（秒）
            search_field: 搜索字段 (all/request/response/error)
            id_from: 起始ID
            id_to: 结束ID
            search_mode: 搜索模式 (and/or)
            conditions: 高级条件列表，每个条件是一个字典，包含:
                - field: 字段名 (role, content, model, app, param, feature等)
                - operator: 操作符 (contains, not_contains, equals, not_equals, exists, not_exists)
                - value: 值
                - role: 针对哪个角色 (用于 content 字段)
        
        Returns:
            添加了过滤条件的查询对象
        """
        # 简单搜索（保留原有逻辑）
        if search:
            # 解析搜索词：支持空格分隔，-前缀表示排除
            include_terms = []
            exclude_terms = []
            
            # 处理引号内的短语
            tokens = re.findall(r'"([^"]+)"|(\S+)', search)
            for quoted, unquoted in tokens:
                term = quoted if quoted else unquoted
                if term.startswith('-') and len(term) > 1:
                    exclude_terms.append(term[1:])
                elif term and term != '-':
                    include_terms.append(term)
            
            def build_term_filter(term, exclude=False):
                """构建单个词的搜索条件"""
                search_filter = f"%{term}%"
                
                if search_field == 'request':
                    if self.db_type == 'postgresql':
                        condition = APIRequest.request_content.ilike(search_filter)
                    else:
                        condition = APIRequest.request_content.like(search_filter)
                elif search_field == 'response':
                    if self.db_type == 'postgresql':
                        condition = APIRequest.response_content.ilike(search_filter)
                    else:
                        condition = APIRequest.response_content.like(search_filter)
                elif search_field == 'error':
                    if self.db_type == 'postgresql':
                        condition = APIRequest.error_message.ilike(search_filter)
                    else:
                        condition = APIRequest.error_message.like(search_filter)
                else:
                    # 搜索所有字段
                    if self.db_type == 'postgresql':
                        condition = or_(
                            APIRequest.model.ilike(search_filter),
                            APIRequest.app_name.ilike(search_filter),
                            APIRequest.request_content.ilike(search_filter),
                            APIRequest.response_content.ilike(search_filter),
                            APIRequest.error_message.ilike(search_filter)
                        )
                    else:
                        condition = or_(
                            APIRequest.model.like(search_filter),
                            APIRequest.app_name.like(search_filter),
                            APIRequest.request_content.like(search_filter),
                            APIRequest.response_content.like(search_filter),
                            APIRequest.error_message.like(search_filter)
                        )
                
                return not_(condition) if exclude else condition
            
            # 构建包含条件
            if include_terms:
                if search_mode == 'or':
                    include_conditions = [build_term_filter(term) for term in include_terms]
                    query = query.filter(or_(*include_conditions))
                else:
                    for term in include_terms:
                        query = query.filter(build_term_filter(term))
            
            # 构建排除条件
            for term in exclude_terms:
                query = query.filter(build_term_filter(term, exclude=True))
        
        # 高级条件筛选
        if conditions:
            for cond in conditions:
                query = self._apply_condition(query, cond)
        
        # 模型筛选
        if model_filter:
            query = query.filter(APIRequest.model == model_filter)
        
        # 应用筛选
        if app_filter:
            query = query.filter(APIRequest.app_name == app_filter)
        
        # 状态筛选
        if status_filter:
            if status_filter == 'success':
                query = query.filter(APIRequest.status_code == 200)
            elif status_filter == 'error':
                query = query.filter(APIRequest.status_code != 200, APIRequest.status_code != 0)
            elif status_filter == 'pending':
                query = query.filter(APIRequest.status_code == 0)
        
        # ID范围筛选
        if id_from is not None:
            query = query.filter(APIRequest.id >= id_from)
        
        if id_to is not None:
            query = query.filter(APIRequest.id <= id_to)
        
        # 时间范围筛选
        if date_from:
            try:
                from_date = datetime.strptime(date_from, '%Y-%m-%d').replace(tzinfo=UTC8)
                query = query.filter(APIRequest.timestamp >= from_date)
            except ValueError:
                pass
        
        if date_to:
            try:
                to_date = datetime.strptime(date_to, '%Y-%m-%d').replace(tzinfo=UTC8) + timedelta(days=1)
                query = query.filter(APIRequest.timestamp < to_date)
            except ValueError:
                pass
        
        # Token范围筛选
        if min_tokens is not None and min_tokens > 0:
            query = query.filter(APIRequest.total_tokens >= min_tokens)
        
        if max_tokens is not None and max_tokens > 0:
            query = query.filter(APIRequest.total_tokens <= max_tokens)
        
        # 耗时范围筛选
        if min_duration is not None and min_duration > 0:
            query = query.filter(APIRequest.duration >= min_duration)
        
        if max_duration is not None and max_duration > 0:
            query = query.filter(APIRequest.duration <= max_duration)
        
        return query
    
    def _apply_condition(self, query, cond: Dict):
        """
        应用单个高级条件
        
        条件类型:
        1. role_filter: 筛选包含/不包含特定角色的请求
           - field: "role", value: "system/user/assistant/tool", operator: "exists/not_exists"
        
        2. role_content: 筛选特定角色消息内容
           - field: "role_content", role: "user/assistant/system", value: "关键词", operator: "contains/not_contains"
        
        3. message_count: 消息数量筛选
           - field: "message_count", operator: "gte/lte/eq", value: 数字
        
        4. has_feature: 特征筛选
           - field: "has_feature", value: "tools/images/functions/stream/error"
        
        5. param: 请求参数筛选
           - field: "param", param_name: "temperature/max_tokens/...", operator: "eq/gte/lte", value: 值
        
        6. first_message / last_message: 首条/末条消息内容
           - field: "first_message/last_message", operator: "contains/not_contains", value: "关键词"
        """
        field = cond.get('field', '')
        operator = cond.get('operator', 'contains')
        value = cond.get('value', '')
        negate = cond.get('negate', False)
        
        like_func = APIRequest.request_content.ilike if self.db_type == 'postgresql' else APIRequest.request_content.like
        resp_like_func = APIRequest.response_content.ilike if self.db_type == 'postgresql' else APIRequest.response_content.like
        
        condition = None
        
        if field == 'role':
            # 角色存在性筛选: 检查 messages 中是否有特定角色
            # JSON 中角色格式: "role": "system" 或 "role":"system"
            role_pattern = f'%"role"%:%"{value}"%'
            if operator == 'exists':
                condition = like_func(role_pattern)
            else:  # not_exists
                condition = not_(like_func(role_pattern))
        
        elif field == 'role_content':
            # 特定角色的消息内容筛选
            # 构建匹配模式：匹配 "role": "xxx" 后面跟着的 "content": "...value..."
            role = cond.get('role', 'user')
            content_value = value
            
            # 简化实现：检查同时包含指定角色和内容关键词
            role_pattern = f'%"role"%:%"{role}"%'
            content_pattern = f'%{content_value}%'
            
            if operator == 'contains':
                # 同时满足角色存在和内容包含
                condition = and_(like_func(role_pattern), like_func(content_pattern))
            else:  # not_contains
                # 有该角色但内容不包含，或者没有该角色
                condition = or_(
                    not_(like_func(role_pattern)),
                    and_(like_func(role_pattern), not_(like_func(content_pattern)))
                )
        
        elif field == 'any_content':
            # 任意消息内容包含
            content_pattern = f'%{value}%'
            if operator == 'contains':
                condition = or_(like_func(content_pattern), resp_like_func(content_pattern))
            else:
                condition = and_(not_(like_func(content_pattern)), not_(resp_like_func(content_pattern)))
        
        elif field == 'request_content':
            # 仅请求内容
            content_pattern = f'%{value}%'
            if operator == 'contains':
                condition = like_func(content_pattern)
            else:
                condition = not_(like_func(content_pattern))
        
        elif field == 'response_content':
            # 仅响应内容
            content_pattern = f'%{value}%'
            if operator == 'contains':
                condition = resp_like_func(content_pattern)
            else:
                condition = not_(resp_like_func(content_pattern))
        
        elif field == 'has_feature':
            # 特征存在性
            if value == 'tools':
                # 检查是否有工具定义或工具调用
                pattern = '%"tools"%:%[%'
                call_pattern = '%"tool_calls"%:%[%'
                condition = or_(like_func(pattern), like_func(call_pattern), resp_like_func(call_pattern))
            elif value == 'tool_calls':
                # 只检查工具调用（响应中）
                pattern = '%"tool_calls"%:%[%'
                condition = resp_like_func(pattern)
            elif value == 'images':
                # 检查是否有图片
                pattern = '%"image_url"%:%'
                condition = like_func(pattern)
            elif value == 'functions':
                # 检查是否有函数定义
                pattern = '%"functions"%:%[%'
                condition = like_func(pattern)
            elif value == 'stream':
                # 检查是否是流式请求
                pattern = '%"stream"%:%true%'
                condition = like_func(pattern)
            elif value == 'error':
                # 有错误
                condition = and_(
                    APIRequest.status_code != 200,
                    APIRequest.status_code != 0
                )
            elif value == 'system_prompt':
                # 有系统提示词
                pattern = '%"role"%:%"system"%'
                condition = like_func(pattern)
            elif value == 'long_context':
                # 长上下文（消息数 > 10，通过多个 role 出现来估算）
                # 简化：检查是否有多个 user 角色
                condition = APIRequest.request_content.like('%"role"%"user"%"role"%"user"%')
            
            if negate and condition is not None:
                condition = not_(condition)
        
        elif field == 'param':
            # 请求参数筛选
            param_name = cond.get('param_name', '')
            if param_name:
                # 构建 JSON 键值匹配模式
                if operator == 'eq':
                    pattern = f'%"{param_name}"%:%{value}%'
                    condition = like_func(pattern)
                elif operator == 'exists':
                    pattern = f'%"{param_name}"%:%'
                    condition = like_func(pattern)
                elif operator == 'not_exists':
                    pattern = f'%"{param_name}"%:%'
                    condition = not_(like_func(pattern))
        
        elif field == 'model_contains':
            # 模型名包含
            if operator == 'contains':
                if self.db_type == 'postgresql':
                    condition = APIRequest.model.ilike(f'%{value}%')
                else:
                    condition = APIRequest.model.like(f'%{value}%')
            else:
                if self.db_type == 'postgresql':
                    condition = not_(APIRequest.model.ilike(f'%{value}%'))
                else:
                    condition = not_(APIRequest.model.like(f'%{value}%'))
        
        elif field == 'app_contains':
            # 应用名包含
            if operator == 'contains':
                if self.db_type == 'postgresql':
                    condition = APIRequest.app_name.ilike(f'%{value}%')
                else:
                    condition = APIRequest.app_name.like(f'%{value}%')
            else:
                if self.db_type == 'postgresql':
                    condition = not_(APIRequest.app_name.ilike(f'%{value}%'))
                else:
                    condition = not_(APIRequest.app_name.like(f'%{value}%'))
        
        elif field == 'error_contains':
            # 错误信息包含
            err_like = APIRequest.error_message.ilike if self.db_type == 'postgresql' else APIRequest.error_message.like
            if operator == 'contains':
                condition = err_like(f'%{value}%')
            else:
                condition = not_(err_like(f'%{value}%'))
        
        # 应用条件
        if condition is not None:
            query = query.filter(condition)
        
        return query
    
    def get_requests_with_stats(self, page: int = 1, page_size: int = 50, search: str = None, 
                                model_filter: str = None, app_filter: str = None,
                                status_filter: str = None, date_from: str = None,
                                date_to: str = None, min_tokens: int = None,
                                max_tokens: int = None, min_duration: float = None,
                                max_duration: float = None, search_field: str = None,
                                sort_by: str = None, sort_order: str = None,
                                id_from: int = None, id_to: int = None,
                                search_mode: str = None, conditions: List[Dict] = None) -> Dict[str, Any]:
        """
        一次查询获取请求列表、统计信息和筛选选项
        
        Args:
            sort_by: 排序字段 (time/tokens/duration)
            sort_order: 排序方向 (asc/desc)
            conditions: 高级筛选条件列表
        
        Returns:
            包含 result, stats, all_models, all_apps 的字典
        """
        session = self.get_session()
        try:
            # 构建基础查询
            base_query = session.query(APIRequest)
            filtered_query = self._build_filter_query(
                base_query, search, model_filter, app_filter,
                status_filter, date_from, date_to, min_tokens,
                max_tokens, min_duration, max_duration, search_field,
                id_from, id_to, search_mode, conditions
            )
            
            # 1. 获取分页数据
            offset = (page - 1) * page_size
            
            # 获取总数
            count_query = filtered_query.with_entities(func.count(APIRequest.id))
            total = count_query.scalar()
            
            # 确定排序方式
            if sort_by == 'tokens':
                order_column = APIRequest.total_tokens
            elif sort_by == 'duration':
                order_column = APIRequest.duration
            else:
                order_column = APIRequest.timestamp
            
            # 确定排序方向
            if sort_order == 'asc':
                ordered_query = filtered_query.order_by(order_column.asc())
            else:
                ordered_query = filtered_query.order_by(order_column.desc())
            
            # 获取分页记录
            requests = ordered_query.offset(offset).limit(page_size).all()
            
            # 2. 获取统计数据（使用同一个过滤条件）
            stats_query = self._build_filter_query(
                session.query(APIRequest), search, model_filter, app_filter,
                status_filter, date_from, date_to, min_tokens,
                max_tokens, min_duration, max_duration, search_field,
                id_from, id_to, search_mode, conditions
            )
            stats = stats_query.with_entities(
                func.count(APIRequest.id).label('total_requests'),
                func.coalesce(func.sum(APIRequest.input_tokens), 0).label('total_input_tokens'),
                func.coalesce(func.sum(APIRequest.output_tokens), 0).label('total_output_tokens'),
                func.coalesce(func.sum(APIRequest.total_tokens), 0).label('total_tokens'),
                func.coalesce(func.avg(APIRequest.duration), 0).label('avg_duration')
            ).first()
            
            # 3. 获取所有不同的模型和应用（全局，不受筛选条件影响）
            all_models = [m[0] for m in session.query(APIRequest.model).distinct().all() if m[0]]
            all_apps = [a[0] for a in session.query(APIRequest.app_name).distinct().all() if a[0]]
            
            return {
                'result': {
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'pages': (total + page_size - 1) // page_size if total > 0 else 0,
                    'requests': [req.to_dict(include_content=True) for req in requests]
                },
                'stats': {
                    'total_requests': stats.total_requests or 0,
                    'total_input_tokens': int(stats.total_input_tokens or 0),
                    'total_output_tokens': int(stats.total_output_tokens or 0),
                    'total_tokens': int(stats.total_tokens or 0),
                    'avg_duration': round(float(stats.avg_duration or 0), 3)
                },
                'all_models': all_models,
                'all_apps': all_apps
            }
        finally:
            session.close()
    
    def get_requests(self, page: int = 1, page_size: int = 50, search: str = None, 
                    model_filter: str = None, app_filter: str = None):
        """获取请求记录列表（保留向后兼容）"""
        session = self.get_session()
        try:
            query = session.query(APIRequest)
            query = self._build_filter_query(query, search, model_filter, app_filter)
            
            # 按时间倒序排列
            query = query.order_by(APIRequest.timestamp.desc())
            
            # 分页
            offset = (page - 1) * page_size
            total = query.with_entities(func.count(APIRequest.id)).scalar()
            requests = query.offset(offset).limit(page_size).all()
            
            return {
                'total': total,
                'page': page,
                'page_size': page_size,
                'pages': (total + page_size - 1) // page_size if total > 0 else 0,
                'requests': [req.to_dict() for req in requests]
            }
        finally:
            session.close()
    
    def get_statistics(self, search: str = None, model_filter: str = None, app_filter: str = None):
        """获取统计信息（保留向后兼容）"""
        session = self.get_session()
        try:
            query = session.query(APIRequest)
            query = self._build_filter_query(query, search, model_filter, app_filter)
            
            # 计算统计数据
            stats = query.with_entities(
                func.count(APIRequest.id).label('total_requests'),
                func.coalesce(func.sum(APIRequest.input_tokens), 0).label('total_input_tokens'),
                func.coalesce(func.sum(APIRequest.output_tokens), 0).label('total_output_tokens'),
                func.coalesce(func.sum(APIRequest.total_tokens), 0).label('total_tokens'),
                func.coalesce(func.avg(APIRequest.duration), 0).label('avg_duration')
            ).first()
            
            return {
                'total_requests': stats.total_requests or 0,
                'total_input_tokens': int(stats.total_input_tokens or 0),
                'total_output_tokens': int(stats.total_output_tokens or 0),
                'total_tokens': int(stats.total_tokens or 0),
                'avg_duration': round(float(stats.avg_duration or 0), 3)
            }
        finally:
            session.close()
    
    def get_unique_models(self) -> List[str]:
        """获取所有不同的模型"""
        session = self.get_session()
        try:
            models = session.query(APIRequest.model).distinct().all()
            return [model[0] for model in models if model[0]]
        finally:
            session.close()
    
    def get_unique_apps(self) -> List[str]:
        """获取所有不同的应用"""
        session = self.get_session()
        try:
            apps = session.query(APIRequest.app_name).distinct().all()
            return [app[0] for app in apps if app[0]]
        finally:
            session.close()
    
    def get_models_and_apps(self) -> Tuple[List[str], List[str]]:
        """
        一次查询获取所有模型和应用列表
        
        优化：合并两次查询为一次会话
        
        Returns:
            (models, apps) 元组
        """
        session = self.get_session()
        try:
            models = [m[0] for m in session.query(APIRequest.model).distinct().all() if m[0]]
            apps = [a[0] for a in session.query(APIRequest.app_name).distinct().all() if a[0]]
            return models, apps
        finally:
            session.close()
    
    def get_request_by_id(self, request_id: int):
        """根据ID获取请求记录"""
        session = self.get_session()
        try:
            request_record = session.query(APIRequest).filter(APIRequest.id == request_id).first()
            return request_record.to_dict() if request_record else None
        finally:
            session.close()

# 全局数据库实例
db = Database()