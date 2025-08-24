import json
import re
from typing import Dict, Any

def clean_request_for_storage(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理请求数据，将base64图片替换为占位符以节省存储空间
    """
    if not request_data:
        return {}
    
    # 深拷贝数据，避免修改原始数据
    import copy
    cleaned_data = copy.deepcopy(request_data)
    
    # 处理messages中的图片
    if 'messages' in cleaned_data:
        for message in cleaned_data['messages']:
            if 'content' in message and isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_url = item.get('image_url', {}).get('url', '')
                        if image_url.startswith('data:image/'):
                            # 获取图片格式信息
                            format_match = re.match(r'data:image/([^;]+);base64,', image_url)
                            image_format = format_match.group(1) if format_match else 'unknown'
                            # 计算原始大小（base64编码后的大小约为原始大小的4/3）
                            encoded_size = len(image_url)
                            estimated_size = int(encoded_size * 3 / 4)
                            size_kb = estimated_size / 1024
                            
                            # 替换为占位符
                            item['image_url']['url'] = f"[图片占位符: {image_format}格式, 约{size_kb:.1f}KB]"
    
    return cleaned_data

def count_tokens_estimate(text) -> int:
    """
    简单的token计数估算
    实际项目中建议使用tiktoken等专业库
    """
    if not text:
        return 0
    
    # 如果是列表（多模态内容），处理每个元素
    if isinstance(text, list):
        total = 0
        for item in text:
            if isinstance(item, dict):
                # 处理字典格式的内容（如 {"type": "text", "text": "content"}）
                if 'text' in item:
                    total += count_tokens_estimate(item['text'])
                elif 'content' in item:
                    total += count_tokens_estimate(item['content'])
            elif isinstance(item, str):
                total += count_tokens_estimate(item)
        return total
    
    # 如果不是字符串，转换为字符串
    if not isinstance(text, str):
        text = str(text)
    
    # 简单估算: 1个token约等于4个字符（英文）或1-2个中文字符
    # 这里使用保守估算
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    
    return chinese_chars + (other_chars // 4)

def extract_tokens_from_response(response_data: Dict[str, Any]) -> tuple[int, int]:
    """
    从API响应中提取token使用信息
    返回: (input_tokens, output_tokens)
    """
    if not response_data:
        return 0, 0
    
    usage = response_data.get('usage', {})
    if usage:
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        return input_tokens, output_tokens
    
    # 如果没有usage信息，尝试估算
    input_tokens = 0
    output_tokens = 0
    
    # 估算输出tokens
    choices = response_data.get('choices', [])
    for choice in choices:
        message = choice.get('message', {})
        content = message.get('content', '')
        output_tokens += count_tokens_estimate(content)
    
    return input_tokens, output_tokens

def extract_tokens_from_request(request_data: Dict[str, Any]) -> int:
    """
    从API请求中估算输入token数量
    """
    if not request_data:
        return 0
    
    total_tokens = 0
    
    try:
        # 处理messages
        messages = request_data.get('messages', [])
        for message in messages:
            content = message.get('content', '')
            total_tokens += count_tokens_estimate(content)
        
        # 处理prompt（某些API格式）
        prompt = request_data.get('prompt', '')
        if prompt:
            total_tokens += count_tokens_estimate(prompt)
    
    except Exception as e:
        # 如果解析失败，返回0而不是抛出异常
        print(f"Token计数失败: {e}")
        return 0
    
    return total_tokens

def clean_response_for_storage(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理响应数据，移除不必要的字段以节省存储空间
    """
    if not response_data:
        return {}
    
    # 保留主要字段
    cleaned = {}
    important_fields = ['id', 'object', 'created', 'model', 'choices', 'usage']
    
    for field in important_fields:
        if field in response_data:
            cleaned[field] = response_data[field]
    
    return cleaned

def format_duration(duration: float) -> str:
    """格式化持续时间显示"""
    if duration < 1:
        return f"{duration*1000:.0f}ms"
    else:
        return f"{duration:.2f}s"

def format_tokens(tokens: int) -> str:
    """格式化token数量显示"""
    if tokens >= 1000:
        return f"{tokens/1000:.1f}k"
    return str(tokens)

def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本用于显示"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."