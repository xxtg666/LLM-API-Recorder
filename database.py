from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
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
        except Exception as e:
            print(f"数据库初始化失败: {e}")
            raise
            
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self):
        """获取数据库会话"""
        return self.SessionLocal()
    
    def add_request(self, model: str, request_content: dict, response_content: dict = None, 
                   input_tokens: int = 0, output_tokens: int = 0, duration: float = 0,
                   status_code: int = 200, error_message: str = None):
        """添加API请求记录"""
        session = self.get_session()
        try:
            request_record = APIRequest(
                model=model,
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
    
    def get_requests(self, page: int = 1, page_size: int = 50, search: str = None):
        """获取请求记录列表"""
        session = self.get_session()
        try:
            query = session.query(APIRequest)
            
            # 全文搜索
            if search:
                search_filter = f"%{search}%"
                query = query.filter(
                    (APIRequest.model.like(search_filter)) |
                    (APIRequest.request_content.like(search_filter)) |
                    (APIRequest.response_content.like(search_filter)) |
                    (APIRequest.error_message.like(search_filter))
                )
            
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