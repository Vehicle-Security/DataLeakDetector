# -*- coding: utf-8 -*-
import sys
from core.platform import get_platform

def setup_console_encoding():
    """设置控制台编码 (仅已在平台层处理，此处保留为空或移除)"""
    pass

def get_drives():
    """获取所有需要监控的根目录/驱动器"""
    return get_platform().get_drives()

def check_admin_rights():
    """检查管理员权限"""
    return get_platform().is_admin()

def get_current_user():
    """获取当前用户名"""
    return get_platform().get_user_name()
