# upload_detection_config.py
"""
模块3配置文件
定义敏感资源、黑名单应用和白名单应用
"""

class UploadDetectionConfig:
    """上传检测配置类"""
    
    def __init__(self):
        """初始化配置"""
        
        # ========== 敏感资源列表 ==========
        # TODO用户应根据实际情况修改此列表
        self.sensitive_files = [
            # 示例敏感文件路径（请根据实际情况修改）
            "/Users/tujiali/Documents/项目3prd设计.docx", # 60 重命名 45 截屏
            "/Users/tujiali/Documents/项目1详细规划.docx", # 43 压缩zip
            "/Users/tujiali/Documents/项目2需求分析.docx", # 42 格式转换：导出为pdf
            "D:\\dingxinyao\\desktop\\智能图像增强系统技术设计文档.docx", # 64 重命名(windows) 61 复制粘贴
            
            "D:\\dingxinyao\\desktop\\公司合作合同.docx", # 46 蓝牙发送文件
            "D:\\dingxinyao\\desktop\\2025年第一季度内部财务报表.docx", # 47 U盘拷贝
            "D:\\dingxinyao\\desktop\\公司机密\\公司内部资料\\公司高层秘密会议纪要.docx", #61 复制粘贴
            "D:\\dingxinyao\\desktop\\重点客户信息汇总表.docx", #62 截图泄露 65 云端上传泄露

            "/Users/tujiali/Documents/需求设计.docx", # 41 重命名
            "/Users/tujiali/Documents/AAA公司员工守则.docx", # 44 复制粘贴 57 OCR识别文字 58 截图泄露
        ]
        
        # ========== 黑名单应用 ==========
        # 这些应用如果检测到上传行为，将触发报警
        self.blacklist_apps = [
            "夸克网盘", "quark.exe", "quark", 
            "微信", "WeChat.exe", "WeChat", 
            "钉钉", "dingtalk.exe", "dingtalk",
            "腾讯会议", "TencentMeeting.exe", "TencentMeeting",
            "钉钉会议", "Meeting.exe", "Meeting",
            "Cherry Studio", "CherryStudio.exe", "CherryStudio",
            "Chatbox", "Chatbox.exe", "Chatbox",
            "豆包", "Doubao.exe", "Doubao",
            "元宝", "Yuanbao.exe", "Yuanbao",
            "飞书", "Feishu.exe", "Feishu",
            "Lark", "Lark.exe", "Lark",
            "QQ", "QQ.exe", "QQ",
            "TIM", "TIM.exe", "TIM",
            "邮箱大师", "Youxiangdashi.exe", "Youxiangdashi",

            "网易邮箱", "163邮箱", "mail.163.com",
            "QQ邮箱", "mail.qq.com"

            "百度网盘", "BaiduNetdisk.exe", "BaiduNetdisk", "pan.baidu.com"
            "GitHub", "github.com",
            "CSDN", "csdn.net",
            "有道云笔记", "note.youdao.com",
            "微信网页版", "wx.qq.com", "wexin.qq.com",
            "Zoom", "zoom.us",
            "通义千问", "tongyi.aliyun.com",
            "Kimi", "kimi.cn", "kimi.com",
            "文心一言", "yiyan.baidu.com",
            "豆包网盘", "doubao.com",

            "元宝网页版", "yuanbao.com",
            "audio2edit.com",
            "chatGPT", "chat.openai.com",
            "DeepSeek", "deepseek.ai", "deepseek.com",
            "阿里云云盘", "aliyundrive.com", "aliyundrive",
            "Gitee", "gitee.com",
            "GitLab", "gitlab.com",
            "smallpdf", "smallpdf.com",
            "ilovepdf", "ilovepdf.com",
            "online-convert", "online-convert.com",
            "腾讯微云", "Weiyun.exe", "Weiyun", "weiyun.com",
            "115网盘", "115.exe", "115",
            "坚果云", "Nutstore.exe", "Nutstore", "jianguoyun.com",
            "gemini", "gemini.google.com",
            "WPS云盘", 

        ]
        
        # ========== 白名单应用 ==========
        # 这些应用的上传行为被认为是安全的，不触发报警
        self.whitelist_apps = [
            "企业微信",
            "WeCom",
            "Slack",
            "Microsoft Teams",
            "企业邮箱",
            "Outlook",
            "企业网盘",
            "企业云存储",
            "WPS", "wpsdesktop.exe", "wps.exe", 
        ]
        
        # ========== 检测规则 ==========
        self.detection_rules = {
            # 直接外发行为关键词
            "upload_keywords": [
                "上传", "发送", "分享", "转发", "附件",
                "upload", "send", "share", "forward", "attach"
            ],
            
            # 疑似外发操作类型
            "upload_operations": [
                "邮件附件外发",
                "聊天转发",
                "云盘上传",
                "网页上传",
                "文件分享",
                "拖拽上传",
            ],
            
            # 报警级别
            "alert_levels": {
                "critical": "严重",    # 黑名单应用上传
                "warning": "警告",     # 可疑行为
                "info": "信息",        # 非黑名单应用上传（仅记录）
            }
        }
    
    def is_sensitive_file(self, file_path: str) -> bool:
        """
        判断文件是否为敏感文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为敏感文件
        """
        if not file_path:
            return False
        
        # 精确匹配
        if file_path in self.sensitive_files:
            return True
        
        # 文件名匹配（不区分路径）
        import os
        file_name = os.path.basename(file_path)
        for sensitive_file in self.sensitive_files:
            if os.path.basename(sensitive_file) == file_name:
                return True
        
        return False
    
    def get_app_category(self, app_name: str) -> str:
        """
        获取应用的类别（黑名单/白名单/未知）
        
        Args:
            app_name: 应用名称
            
        Returns:
            "blacklist" | "whitelist" | "unknown"
        """
        if not app_name:
            return "unknown"
        
        app_name_lower = app_name.lower()
        
        for blacklist_app in self.blacklist_apps:
            if blacklist_app.lower() in app_name_lower:# or app_name_lower in blacklist_app.lower():
                return "blacklist"
        
        for whitelist_app in self.whitelist_apps:
            if whitelist_app.lower() in app_name_lower:# or app_name_lower in whitelist_app.lower():
                return "whitelist"
        
        return "unknown"
    
    def should_alert(self, app_category: str, behavior_category: str) -> tuple[bool, str]:
        """
        判断是否应该报警
        
        Args:
            app_category: 应用类别 (blacklist/whitelist/unknown)
            behavior_category: 行为类别（如"直接外发"）
            
        Returns:
            (是否报警, 报警级别)
        """
        # 白名单应用不报警
        if app_category == "whitelist":
            return False, "info"
        
        # 黑名单应用且有直接外发行为，触发严重报警
        if app_category == "blacklist" and "外发" in behavior_category:
            return True, "critical"
        
        # 黑名单应用的其他可疑行为，触发警告
        if app_category == "blacklist":
            return True, "warning"
        
        # 未知应用的直接外发行为，仅记录不报警
        if app_category == "unknown" and "外发" in behavior_category:
            return False, "info"
        
        return False, "info"


# 创建全局配置实例
config = UploadDetectionConfig()


# ========== 便捷函数 ==========

def get_sensitive_files():
    """获取敏感文件列表"""
    return config.sensitive_files


def get_blacklist_apps():
    """获取黑名单应用列表"""
    return config.blacklist_apps


def get_whitelist_apps():
    """获取白名单应用列表"""
    return config.whitelist_apps


def is_sensitive_file(file_path: str) -> bool:
    """判断是否为敏感文件"""
    return config.is_sensitive_file(file_path)


def get_app_category(app_name: str) -> str:
    """获取应用类别"""
    return config.get_app_category(app_name)


def should_alert(app_category: str, behavior_category: str) -> tuple[bool, str]:
    """判断是否应该报警"""
    return config.should_alert(app_category, behavior_category)


if __name__ == "__main__":
    """测试配置"""
    print("=" * 80)
    print("上传检测配置测试")
    print("=" * 80)
    
    print(f"\n敏感文件数量: {len(config.sensitive_files)}")
    print(f"黑名单应用数量: {len(config.blacklist_apps)}")
    print(f"白名单应用数量: {len(config.whitelist_apps)}")
    
    # 测试文件检测
    test_files = [
        "/Users/tujiali/Documents/项目1详细规划.docx",
        "/Users/tujiali/Documents/普通文档.txt",
    ]
    
    print("\n测试文件检测:")
    for file_path in test_files:
        is_sensitive = config.is_sensitive_file(file_path)
        print(f"  {file_path}: {'敏感' if is_sensitive else '普通'}")
    
    # 测试应用分类
    test_apps = [
        "微信",
        "企业微信",
        "Chrome浏览器",
    ]
    
    print("\n测试应用分类:")
    for app_name in test_apps:
        category = config.get_app_category(app_name)
        print(f"  {app_name}: {category}")
    
    # 测试报警规则
    test_cases = [
        ("blacklist", "直接外发"),
        ("whitelist", "直接外发"),
        ("unknown", "直接外发"),
    ]
    
    print("\n测试报警规则:")
    for app_cat, behavior_cat in test_cases:
        should_alert_flag, alert_level = config.should_alert(app_cat, behavior_cat)
        print(f"  {app_cat} + {behavior_cat}: 报警={should_alert_flag}, 级别={alert_level}")
