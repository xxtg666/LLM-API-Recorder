from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, inspect, text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import json
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
                             model_filter: str = None, app_filter: str = None):
        """
        构建通用的过滤查询条件
        
        Args:
            query: SQLAlchemy 查询对象
            search: 搜索关键词
            model_filter: 模型筛选
            app_filter: 应用筛选
        
        Returns:
            添加了过滤条件的查询对象
        """
        # 全文搜索
        if search:
            search_filter = f"%{search}%"
            if self.db_type == 'postgresql':
                # PostgreSQL 使用 ILIKE 进行大小写不敏感搜索
                query = query.filter(
                    (APIRequest.model.ilike(search_filter)) |
                    (APIRequest.app_name.ilike(search_filter)) |
                    (APIRequest.request_content.ilike(search_filter)) |
                    (APIRequest.response_content.ilike(search_filter)) |
                    (APIRequest.error_message.ilike(search_filter))
                )
            else:
                # SQLite 使用 LIKE
                query = query.filter(
                    (APIRequest.model.like(search_filter)) |
                    (APIRequest.app_name.like(search_filter)) |
                    (APIRequest.request_content.like(search_filter)) |
                    (APIRequest.response_content.like(search_filter)) |
                    (APIRequest.error_message.like(search_filter))
                )
        
        # 模型筛选
        if model_filter:
            query = query.filter(APIRequest.model == model_filter)
        
        # 应用筛选
        if app_filter:
            query = query.filter(APIRequest.app_name == app_filter)
        
        return query
    
    def get_requests_with_stats(self, page: int = 1, page_size: int = 50, search: str = None, 
                                model_filter: str = None, app_filter: str = None) -> Dict[str, Any]:
        """
        一次查询获取请求列表、统计信息和筛选选项
        
        优化：合并了原来的 get_requests、get_statistics、get_unique_models、get_unique_apps 四次查询为一次会话
        
        Returns:
            包含 result, stats, all_models, all_apps 的字典
        """
        session = self.get_session()
        try:
            # 构建基础查询
            base_query = session.query(APIRequest)
            filtered_query = self._build_filter_query(base_query, search, model_filter, app_filter)
            
            # 1. 获取分页数据（只查询需要的列，避免加载大文本字段）
            offset = (page - 1) * page_size
            
            # 使用子查询优化：先获取ID，再获取详细数据
            count_query = filtered_query.with_entities(func.count(APIRequest.id))
            total = count_query.scalar()
            
            # 获取分页记录
            requests = filtered_query.order_by(APIRequest.timestamp.desc()).offset(offset).limit(page_size).all()
            
            # 2. 获取统计数据（使用同一个过滤条件）
            stats_query = self._build_filter_query(session.query(APIRequest), search, model_filter, app_filter)
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