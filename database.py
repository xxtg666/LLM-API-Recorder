from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone, timedelta
import json
import os
from config import config

Base = declarative_base()

# 定义UTC+8时区
UTC8 = timezone(timedelta(hours=8))

class APIRequest(Base):
    """API请求记录表"""
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC8), index=True)
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
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'model': self.model,
            'request_content': json.loads(self.request_content) if self.request_content else None,
            'response_content': json.loads(self.response_content) if self.response_content else None,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'duration': self.duration,
            'status_code': self.status_code,
            'error_message': self.error_message
        }

class Database:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = config.db_path
        
        # 如果是相对路径，使用当前工作目录
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)
        
        # 标准化路径
        db_path = os.path.normpath(db_path)
        
        # 检查数据库文件是否已存在
        db_exists = os.path.exists(db_path)
        
        if db_exists:
            print(f"使用现有数据库: {db_path}")
        else:
            print(f"创建新数据库: {db_path}")
        
        # 创建数据库引擎
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        
        # 创建表（如果不存在）
        try:
            Base.metadata.create_all(bind=self.engine)
            if not db_exists:
                print("数据库表创建成功")
            else:
                print("数据库连接成功")
                # 检查并执行数据库升级
                self._upgrade_database()
        except Exception as e:
            print(f"数据库初始化失败: {e}")
            raise
            
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _upgrade_database(self):
        """检查并升级数据库结构"""
        try:
            # 检查表结构
            inspector = inspect(self.engine)
            columns = inspector.get_columns('api_requests')
            column_names = [col['name'] for col in columns]
            
            # 检查是否缺少app_name字段
            if 'app_name' not in column_names:
                print("检测到旧版数据库，正在添加app_name字段...")
                
                # 使用ALTER TABLE添加新字段
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
    
    def get_requests(self, page: int = 1, page_size: int = 50, search: str = None, 
                    model_filter: str = None, app_filter: str = None):
        """获取请求记录列表"""
        session = self.get_session()
        try:
            query = session.query(APIRequest)
            
            # 全文搜索
            if search:
                search_filter = f"%{search}%"
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
            
            # 按时间倒序排列
            query = query.order_by(APIRequest.timestamp.desc())
            
            # 分页
            offset = (page - 1) * page_size
            total = query.count()
            requests = query.offset(offset).limit(page_size).all()
            
            return {
                'total': total,
                'page': page,
                'page_size': page_size,
                'pages': (total + page_size - 1) // page_size,
                'requests': [req.to_dict() for req in requests]
            }
        finally:
            session.close()
    
    def get_statistics(self, search: str = None, model_filter: str = None, app_filter: str = None):
        """获取统计信息"""
        session = self.get_session()
        try:
            # 构建查询条件
            query = session.query(APIRequest)
            
            # 全文搜索
            if search:
                search_filter_text = f"%{search}%"
                query = query.filter(
                    (APIRequest.model.like(search_filter_text)) |
                    (APIRequest.app_name.like(search_filter_text)) |
                    (APIRequest.request_content.like(search_filter_text)) |
                    (APIRequest.response_content.like(search_filter_text)) |
                    (APIRequest.error_message.like(search_filter_text))
                )
            
            # 模型筛选
            if model_filter:
                query = query.filter(APIRequest.model == model_filter)
            
            # 应用筛选
            if app_filter:
                query = query.filter(APIRequest.app_name == app_filter)
            
            # 计算统计数据
            from sqlalchemy import func
            stats = query.with_entities(
                func.count(APIRequest.id).label('total_requests'),
                func.sum(APIRequest.input_tokens).label('total_input_tokens'),
                func.sum(APIRequest.output_tokens).label('total_output_tokens'),
                func.sum(APIRequest.total_tokens).label('total_tokens'),
                func.avg(APIRequest.duration).label('avg_duration')
            ).first()
            
            return {
                'total_requests': stats.total_requests or 0,
                'total_input_tokens': stats.total_input_tokens or 0,
                'total_output_tokens': stats.total_output_tokens or 0,
                'total_tokens': stats.total_tokens or 0,
                'avg_duration': round(stats.avg_duration or 0, 3)
            }
        finally:
            session.close()
    
    def get_unique_models(self):
        """获取所有不同的模型"""
        session = self.get_session()
        try:
            from sqlalchemy import func
            models = session.query(APIRequest.model).distinct().all()
            return [model[0] for model in models if model[0]]
        finally:
            session.close()
    
    def get_unique_apps(self):
        """获取所有不同的应用"""
        session = self.get_session()
        try:
            from sqlalchemy import func
            apps = session.query(APIRequest.app_name).distinct().all()
            return [app[0] for app in apps if app[0]]
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