import json, re
from datetime import datetime
def _extract_gt_retention(completion: str):
    """
    从模型输出中提取gt_retention字段值
    
    Args:
        completion: 模型的输出字符串
        
    Returns:
        gt_retention的值，如果解析失败返回None
    """
    
    json_data = None
    gt_retention, gt_retention_code, reason = -1, '', ''
    
    try:
        # 尝试从文本中提取JSON部分
        # 查找可能的JSON格式内容
        json_pattern = r'\{[^{}]*"gt_retention"[^{}]*\}'
        matches = re.findall(json_pattern, completion, re.DOTALL)
        for match in matches:
            try:
                json_data = json.loads(match)
                # if 'gt_retention' in json_data:
                #     return json_data['gt_retention']
                if json_data is not None and 'gt_retention' in json_data:
                    break
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    # 第一种解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
            fixed_json = completion.split('```json\n', 1)[-1]
            fixed_json = fixed_json.replace('```', '')
            # 仅转义字符串值内的换行符（非结构换行）
            fixed_json = re.sub(
                r'("[^"]*")', 
                lambda m: m.group(0).replace('\n', '\\\\n'), 
                fixed_json
            )
            json_data = json.loads(fixed_json, strict=False) #
    except Exception as e:
        print(f'[Error-1] {datetime.now()} fail to convert completion to json_data "{json_data}"')

    # 第二种解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
            fixed_json = re.sub(r'^```json\s*|\s*```$', '', completion, flags=re.DOTALL).strip()
            # 修复字符串值内的特殊字符
            def fix_json_string(match):
                s = match.group(1)
                # 转义双引号（但保留已转义的双引号）
                s = re.sub(r'(?<!\\)"', r'\\"', s)
                # 转义换行符
                s = s.replace('\n', '\\n')
                # 转义反斜杠（但保留已转义的）
                s = re.sub(r'(?<!\\)\\', r'\\\\', s)
                return f'"{s}"'
            # 处理所有字符串值
            fixed_json = re.sub(
                r'"((?:[^"\\]|\\.)*)"', 
                fix_json_string, 
                fixed_json,
                flags=re.DOTALL
            )
            json_data = json.loads(fixed_json) # # strict=False
        
    except Exception as e:
        print(f'[Error-2] {datetime.now()} fail to convert completion to json_data "{json_data}"')
    
    # 第三种解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
            json_data = sanitize_json_string(completion)
    except Exception as e:
        print(f'[Error-3] {datetime.now()} fail to convert completion to json_data "{json_data}"')
    
    # 第四种解析方法
    try:
        if json_data is None or 'gt_retention' not in json_data or 'gt_retention_code' not in json_data or 'reason' not in json_data:
            json_data = sanitize_json_string2(completion)
    except Exception as e:
        print(f'[Error-4] {datetime.now()} fail to convert completion to json_data "{json_data}"')
    
    if json_data is not None and 'gt_retention' in json_data and 'gt_retention_code' in json_data and 'reason' in json_data:
        gt_retention = json_data['gt_retention']
        gt_retention_code = json_data['gt_retention_code']
        reason = json_data['reason']
    
    if gt_retention==-1:
        print(f'[Error-5] {datetime.now()} finally fail to convert completion "{[completion]}" to json_data="{json_data}"')
    
    return gt_retention

s='{\n  "gt_retention": 0,\n  "gt_retention_code": "int checkGap = 0;\\n\\nwhile (!this->shouldStop()) {\\n    if (this->slmHandle.load() == 0) {\\n        std::this_thread::sleep_for(std::chrono::seconds(1));\\n        continue;\\n    }\\n    if (getCurrentUTCTimestamp() > expiring && expiring > 0) {\\n        vaild.store(false);\\n        ERROR_MSG(\\"License has expired \\", 0x22000012, \\"error...\\");\\n    }\\n\\n    // 非服务器联网授权模式，也间隔10秒判断\\n    if (this->licMode != control::LicenseMode::LICENSE_MODE_SERVER_CENTRAL && this->licMode != control::LicenseMode::LICENSE_MODE_SINGLE_MACHINE) {\\n        ++checkGap;\\n        if (checkGap < 10) {\\n            std::this_thread::sleep_for(std::chrono::seconds(1));\\n            continue;\\n        }\\n        else {\\n            // 正确位置应为循环外，避免重复初始化\\n            // 移除循环内的 checkGap 重置逻辑\\n            // 原代码中的 checkGap 初始化应仅在循环外部进行一次\\n            getDevLicenseID(licenseID);\\n            if (licenseStatus >= ENUM_LICENSE_STATUS_EXPIRED) {\\n                vaild.store(false);\\n            }\\n        }\\n    }\\n}",\n  "reason": "补全代码在循环内部将 checkGap 重置为 0，导 致计数器无法累积，无法实现每 10 次循环后执行一次检查的业务逻辑。正确的做法是将 checkGap 的初始化放在循环外部，避免在循环内部重置。此外，剪切板信息中的代码与补全代码无关，不应被错误引用。"'

print(_extract_gt_retention(s))