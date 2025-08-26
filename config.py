import configparser
import os
from typing import Optional

class Config:
    def __init__(self, config_file: str = "config.ini"):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')
        else:
            # 创建默认配置
            self.create_default_config()
    
    def create_default_config(self):
        """创建默认配置文件"""
        self.config.add_section('llm')
        self.config.set('llm', 'api_base_url', 'https://api.openai.com/v1')
        
        self.config.add_section('server')
        self.config.set('server', 'host', '0.0.0.0')
        self.config.set('server', 'port', '8000')
        self.config.set('server', 'debug', 'true')
        self.config.set('server', 'access_password', '')
        self.config.set('server', 'web_title', 'LLM API Recorder')
        
        self.config.add_section('database')
        self.config.set('database', 'db_path', 'llm_requests.db')
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)
    
    @property
    def llm_api_base_url(self) -> str:
        return self.config.get('llm', 'api_base_url')
    
    @property
    def server_host(self) -> str:
        return self.config.get('server', 'host')
    
    @property
    def server_port(self) -> int:
        return self.config.getint('server', 'port')
    
    @property
    def server_debug(self) -> bool:
        return self.config.getboolean('server', 'debug')
    
    @property
    def access_password(self) -> str:
        return self.config.get('server', 'access_password', fallback='')
    
    @property
    def web_title(self) -> str:
        return self.config.get('server', 'web_title', fallback='LLM API 监控面板')
    
    @property
    def db_path(self) -> str:
        return self.config.get('database', 'db_path')

# 全局配置实例
config = Config()