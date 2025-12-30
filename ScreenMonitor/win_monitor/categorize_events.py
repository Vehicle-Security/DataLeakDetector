import json

with open('d:/code/win_monitor/output/session_20251223_232636/key_events/key_events_20251223_232636.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

# 分类统计
tmp_files = []
aaa_files = []
shixi_files = []
other_files = []

for event in events:
    fname = event.get('file_name', '')
    fpath = event.get('file_path', '')
    
    if '.tmp' in fname:
        tmp_files.append(event)
    elif 'AAA' in fname or 'AAA' in fpath:
        aaa_files.append(event)
    elif '实习' in str(event):
        shixi_files.append(event)
    else:
        other_files.append(event)

print(f"事件分类统计:")
print(f"  .tmp临时文件: {len(tmp_files)} 个")
print(f"  AAA公司合同: {len(aaa_files)} 个")
print(f"  实习报告相关: {len(shixi_files)} 个")
print(f"  其他文件: {len(other_files)} 个")
print(f"  总计: {len(events)} 个")

if aaa_files:
    print(f"\n{'='*80}\nAAA公司合同文件 (应保留):")
    for e in aaa_files:
        print(f"  - {e['file_name']}")
        print(f"    路径: {e['file_path']}")
        print(f"    应用: {e.get('app_name', 'N/A')}")

if shixi_files:
    print(f"\n{'='*80}\n实习报告文件 (不应捕获! ⚠️):")
    for e in shixi_files:
        print(f"  - {e['file_name']}")
        print(f"    路径: {e['file_path']}")
        print(f"    应用: {e.get('app_name', 'N/A')}")
        print(f"    检测方法: {e.get('detection_method', 'N/A')}")
        window = e.get('window_info', {}).get('window_title', '')
        if window:
            print(f"    窗口标题: {window}")

if tmp_files:
    print(f"\n{'='*80}\n临时文件 (应过滤! ⚠️): 共{len(tmp_files)}个")
    # 只显示前3个
    for e in tmp_files[:3]:
        print(f"  - {e['file_name']}")
        print(f"    应用: {e.get('app_name', 'N/A')}")
    if len(tmp_files) > 3:
        print(f"  ... 还有{len(tmp_files)-3}个临时文件")
