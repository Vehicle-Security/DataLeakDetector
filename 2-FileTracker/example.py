"""
EvidenceTracer 使用示例

展示如何使用 EvidenceTracer 模块追踪敏感资源流转
"""

import json
from graph import run_evidence_tracer


def example_1_simple_file_access():
    """
    示例 1：简单的文件访问场景
    """
    return [
        {
            "operation_id": "op_001",
            "operation_type": "file_access",
            "resource_name": "年度财报.xlsx",
            "app_name": "Microsoft Excel",
            "start_time": "2024-01-05 09:15:20",
            "end_time": "2024-01-05 09:20:35",
            "keyframes": ["/frames/001.jpg"],
            "raw_description": "用户在 Microsoft Excel 中打开了年度财报.xlsx 文件，并查看了其中的敏感财务数据。"
        }
    ]


def example_2_rename_and_upload():
    """
    示例 2：文件重命名后上传（典型的脱敏行为）
    """
    return [
        {
            "operation_id": "op_101",
            "operation_type": "file_access",
            "resource_name": "客户隐私数据.csv",
            "app_name": "Notepad++",
            "start_time": "2024-01-05 10:30:00",
            "end_time": "2024-01-05 10:32:00",
            "keyframes": ["/frames/101.jpg"],
            "raw_description": "用户在 Notepad++ 中打开了客户隐私数据.csv，浏览了客户姓名、电话、地址等敏感信息。"
        },
        {
            "operation_id": "op_102",
            "operation_type": "file_rename",
            "resource_name": "数据备份.csv",
            "app_name": "文件资源管理器",
            "start_time": "2024-01-05 10:33:00",
            "end_time": "2024-01-05 10:33:10",
            "keyframes": ["/frames/102.jpg"],
            "raw_description": "用户在文件资源管理器中将'客户隐私数据.csv'重命名为'数据备份.csv'。"
        },
        {
            "operation_id": "op_103",
            "operation_type": "file_upload",
            "resource_name": "数据备份.csv",
            "app_name": "Chrome - Gmail",
            "start_time": "2024-01-05 10:35:00",
            "end_time": "2024-01-05 10:35:45",
            "keyframes": ["/frames/103.jpg"],
            "raw_description": "用户在 Gmail 中将'数据备份.csv'作为附件发送到个人邮箱。"
        }
    ]


def example_3_compress_encrypt_upload():
    """
    示例 3：压缩加密后外发（高度隐蔽的泄露行为）
    """
    return [
        {
            "operation_id": "op_201",
            "operation_type": "file_access",
            "resource_name": "项目源代码.zip",
            "app_name": "VS Code",
            "start_time": "2024-01-05 14:00:00",
            "end_time": "2024-01-05 14:15:00",
            "keyframes": ["/frames/201.jpg", "/frames/202.jpg"],
            "raw_description": "用户在 VS Code 中打开了多个源代码文件，包含核心算法实现。"
        },
        {
            "operation_id": "op_202",
            "operation_type": "file_compress",
            "resource_name": "backup_20240105.zip",
            "app_name": "WinRAR",
            "start_time": "2024-01-05 14:16:00",
            "end_time": "2024-01-05 14:16:30",
            "keyframes": ["/frames/203.jpg"],
            "raw_description": "用户使用 WinRAR 将源代码文件夹压缩为'backup_20240105.zip'。"
        },
        {
            "operation_id": "op_203",
            "operation_type": "file_encrypt",
            "resource_name": "backup_20240105.zip",
            "app_name": "WinRAR",
            "start_time": "2024-01-05 14:17:00",
            "end_time": "2024-01-05 14:17:15",
            "keyframes": ["/frames/204.jpg"],
            "raw_description": "用户在压缩过程中为'backup_20240105.zip'设置了密码保护。"
        },
        {
            "operation_id": "op_204",
            "operation_type": "file_upload",
            "resource_name": "backup_20240105.zip",
            "app_name": "Chrome - 百度网盘",
            "start_time": "2024-01-05 14:20:00",
            "end_time": "2024-01-05 14:21:30",
            "keyframes": ["/frames/205.jpg", "/frames/206.jpg"],
            "raw_description": "用户在浏览器中登录百度网盘，并将'backup_20240105.zip'上传到个人云存储空间。"
        }
    ]


def example_4_screenshot_and_copy():
    """
    示例 4：截图+文本复制（内容提取型泄露）
    """
    return [
        {
            "operation_id": "op_301",
            "operation_type": "file_access",
            "resource_name": "内部会议记录.docx",
            "app_name": "Microsoft Word",
            "start_time": "2024-01-05 16:00:00",
            "end_time": "2024-01-05 16:05:00",
            "keyframes": ["/frames/301.jpg"],
            "raw_description": "用户在 Word 中打开了内部会议记录，其中包含未公开的战略计划。"
        },
        {
            "operation_id": "op_302",
            "operation_type": "screenshot",
            "resource_name": "screenshot_20240105_160530.png",
            "app_name": "系统截图工具",
            "start_time": "2024-01-05 16:05:30",
            "end_time": "2024-01-05 16:05:35",
            "keyframes": ["/frames/302.jpg"],
            "raw_description": "用户使用系统截图工具对 Word 窗口进行了截图，保存为 screenshot_20240105_160530.png。"
        },
        {
            "operation_id": "op_303",
            "operation_type": "text_copy",
            "resource_name": "文本内容",
            "app_name": "Microsoft Word",
            "start_time": "2024-01-05 16:06:00",
            "end_time": "2024-01-05 16:06:05",
            "keyframes": ["/frames/303.jpg"],
            "raw_description": "用户选中了会议记录中的一段文字，按下 Ctrl+C 进行复制。"
        },
        {
            "operation_id": "op_304",
            "operation_type": "file_upload",
            "resource_name": "screenshot_20240105_160530.png",
            "app_name": "微信",
            "start_time": "2024-01-05 16:07:00",
            "end_time": "2024-01-05 16:07:20",
            "keyframes": ["/frames/304.jpg"],
            "raw_description": "用户在微信中将截图发送给外部联系人。"
        }
    ]


def example_5_format_conversion_chain():
    """
    示例 5：格式转换链（多步转换规避检测）
    """
    return [
        {
            "operation_id": "op_401",
            "operation_type": "file_access",
            "resource_name": "设计图纸.dwg",
            "app_name": "AutoCAD",
            "start_time": "2024-01-05 11:00:00",
            "end_time": "2024-01-05 11:20:00",
            "keyframes": ["/frames/401.jpg"],
            "raw_description": "用户在 AutoCAD 中打开了产品设计图纸，查看了技术细节。"
        },
        {
            "operation_id": "op_402",
            "operation_type": "format_conversion",
            "resource_name": "设计图纸.pdf",
            "app_name": "AutoCAD",
            "start_time": "2024-01-05 11:21:00",
            "end_time": "2024-01-05 11:21:45",
            "keyframes": ["/frames/402.jpg"],
            "raw_description": "用户使用 AutoCAD 的导出功能，将 DWG 文件转换为 PDF 格式，保存为'设计图纸.pdf'。"
        },
        {
            "operation_id": "op_403",
            "operation_type": "file_rename",
            "resource_name": "技术文档.pdf",
            "app_name": "文件资源管理器",
            "start_time": "2024-01-05 11:22:00",
            "end_time": "2024-01-05 11:22:10",
            "keyframes": ["/frames/403.jpg"],
            "raw_description": "用户将'设计图纸.pdf'重命名为'技术文档.pdf'。"
        },
        {
            "operation_id": "op_404",
            "operation_type": "file_upload",
            "resource_name": "技术文档.pdf",
            "app_name": "Chrome - Dropbox",
            "start_time": "2024-01-05 11:25:00",
            "end_time": "2024-01-05 11:25:50",
            "keyframes": ["/frames/404.jpg"],
            "raw_description": "用户在 Dropbox 中上传了'技术文档.pdf'。"
        }
    ]


def run_example(example_name, example_func, max_iterations=10):
    """
    通用示例运行函数
    
    Args:
        example_name: 示例名称
        example_func: 返回 operations 的函数
        max_iterations: 最大迭代次数
    """
    print("\n" + "="*80)
    print(f"示例: {example_name}")
    print("="*80)
    
    operations = example_func()
    result = run_evidence_tracer(operations, max_iterations=max_iterations)
    
    print("\n最终结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print("\n" + "🔍" * 40)
    print("EvidenceTracer 模块使用示例")
    print("🔍" * 40)
    
    # 运行所有示例（可根据需要注释部分示例）
    
    # run_example("简单文件访问", example_1_simple_file_access, max_iterations=10)
    
    # run_example("文件重命名后上传", example_2_rename_and_upload, max_iterations=10)
    
    # run_example("压缩加密后外发", example_3_compress_encrypt_upload, max_iterations=10)
    
    # run_example("截图和文本复制", example_4_screenshot_and_copy, max_iterations=10)
    
    run_example("格式转换链", example_5_format_conversion_chain, max_iterations=10)
    
    print("\n" + "="*80)
    print("✅ 示例运行完成")
    print("="*80)
