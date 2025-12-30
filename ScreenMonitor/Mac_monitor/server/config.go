// server/config.go
// 数据泄露监控配置 - 黑名单应用和网站列表
package main

// RiskCategory 风险类别
type RiskCategory struct {
	Name     string `json:"name"`
	Type     string `json:"type"` // website, application
	Category string `json:"category"`
}

// 黑名单应用 (macOS 应用名称)
var BlacklistApps = map[string]RiskCategory{
	// 即时通讯类
	"QQ":       {Name: "QQ", Type: "application", Category: "即时通讯"},
	"WeChat":   {Name: "微信", Type: "application", Category: "即时通讯"},
	"微信":      {Name: "微信", Type: "application", Category: "即时通讯"},
	"DingTalk": {Name: "钉钉", Type: "application", Category: "即时通讯"},
	"钉钉":      {Name: "钉钉", Type: "application", Category: "即时通讯"},
	"Feishu":   {Name: "飞书", Type: "application", Category: "协作办公"},
	"飞书":      {Name: "飞书", Type: "application", Category: "协作办公"},
	"Lark":     {Name: "飞书", Type: "application", Category: "协作办公"},

	// 会议类
	"zoom.us":         {Name: "Zoom", Type: "application", Category: "会议"},
	"Zoom":            {Name: "Zoom", Type: "application", Category: "会议"},
	"TencentMeeting":  {Name: "腾讯会议", Type: "application", Category: "会议"},
	"腾讯会议":            {Name: "腾讯会议", Type: "application", Category: "会议"},
	"Meeting":         {Name: "钉钉会议", Type: "application", Category: "会议"},
	"DingTalkMeeting": {Name: "钉钉会议", Type: "application", Category: "会议"},

	// AI 应用类
	"Doubao":        {Name: "豆包", Type: "application", Category: "AI"},
	"豆包":           {Name: "豆包", Type: "application", Category: "AI"},
	"yuanbao":       {Name: "元宝", Type: "application", Category: "AI"},
	"元宝":           {Name: "元宝", Type: "application", Category: "AI"},
	"Cherry Studio": {Name: "Cherry Studio", Type: "application", Category: "AI"},
	"Chatbox":       {Name: "Chatbox", Type: "application", Category: "AI"},

	// 网盘类
	"BaiduNetdisk":     {Name: "百度网盘", Type: "application", Category: "网盘"},
	"百度网盘":             {Name: "百度网盘", Type: "application", Category: "网盘"},
	"Quark":            {Name: "夸克网盘", Type: "application", Category: "网盘"},
	"夸克":               {Name: "夸克网盘", Type: "application", Category: "网盘"},
	"AliyunDrive":      {Name: "阿里云盘", Type: "application", Category: "网盘"},
	"aDrive":           {Name: "阿里云盘", Type: "application", Category: "网盘"},
	"Thunder":          {Name: "迅雷", Type: "application", Category: "下载工具"},
	"迅雷":               {Name: "迅雷", Type: "application", Category: "下载工具"},
	"Jianguoyun":       {Name: "坚果云", Type: "application", Category: "网盘"},
	"坚果云":              {Name: "坚果云", Type: "application", Category: "网盘"},
	"TencentMicroDisk": {Name: "腾讯微云", Type: "application", Category: "网盘"},

	// 办公应用（用于检测敏感文件操作）
	"WPS Office":       {Name: "WPS Office", Type: "application", Category: "办公"},
	"wpsoffice":        {Name: "WPS Office", Type: "application", Category: "办公"},
	"Microsoft Word":   {Name: "Microsoft Word", Type: "application", Category: "办公"},
	"Microsoft Excel":  {Name: "Microsoft Excel", Type: "application", Category: "办公"},
	"Preview":          {Name: "预览", Type: "application", Category: "办公"},
	"TextEdit":         {Name: "文本编辑", Type: "application", Category: "办公"},
}

// 黑名单网站 (域名匹配)
var BlacklistWebsites = map[string]RiskCategory{
	// 网盘类
	"pan.baidu.com":        {Name: "百度网盘", Type: "website", Category: "网盘"},
	"yun.baidu.com":        {Name: "百度网盘", Type: "website", Category: "网盘"},
	"www.aliyundrive.com":  {Name: "阿里云盘", Type: "website", Category: "网盘"},
	"www.alipan.com":       {Name: "阿里云盘", Type: "website", Category: "网盘"},
	"pan.quark.cn":         {Name: "夸克网盘", Type: "website", Category: "网盘"},
	"www.jianguoyun.com":   {Name: "坚果云", Type: "website", Category: "网盘"},
	"www.weiyun.com":       {Name: "腾讯微云", Type: "website", Category: "网盘"},
	"115.com":              {Name: "115网盘", Type: "website", Category: "网盘"},

	// 代码托管类
	"github.com":      {Name: "GitHub", Type: "website", Category: "代码托管"},
	"gist.github.com": {Name: "GitHub Gist", Type: "website", Category: "代码托管"},
	"gitee.com":       {Name: "Gitee", Type: "website", Category: "代码托管"},
	"gitlab.com":      {Name: "GitLab", Type: "website", Category: "代码托管"},

	// 技术社区类
	"www.csdn.net":  {Name: "CSDN", Type: "website", Category: "技术社区"},
	"csdn.net":      {Name: "CSDN", Type: "website", Category: "技术社区"},
	"blog.csdn.net": {Name: "CSDN", Type: "website", Category: "技术社区"},

	// 笔记类
	"note.youdao.com": {Name: "有道云笔记", Type: "website", Category: "笔记"},
	"www.wolai.com":   {Name: "Wolai", Type: "website", Category: "笔记"},
	"www.notion.so":   {Name: "Notion", Type: "website", Category: "笔记"},

	// 邮箱类
	"mail.163.com":    {Name: "网易邮箱", Type: "website", Category: "邮箱"},
	"mail.126.com":    {Name: "网易邮箱", Type: "website", Category: "邮箱"},
	"mail.qq.com":     {Name: "QQ邮箱", Type: "website", Category: "邮箱"},
	"wx.mail.qq.com":  {Name: "QQ邮箱", Type: "website", Category: "邮箱"},
	"outlook.live.com": {Name: "Outlook", Type: "website", Category: "邮箱"},
	"mail.google.com": {Name: "Gmail", Type: "website", Category: "邮箱"},

	// 即时通讯类
	"weixin.qq.com":              {Name: "微信网页版", Type: "website", Category: "即时通讯"},
	"wx.qq.com":                  {Name: "微信网页版", Type: "website", Category: "即时通讯"},
	"filehelper.weixin.qq.com":  {Name: "微信文件传输助手", Type: "website", Category: "即时通讯"},
	"web.telegram.org":          {Name: "Telegram", Type: "website", Category: "即时通讯"},

	// 会议类
	"zoom.us":       {Name: "Zoom", Type: "website", Category: "会议"},
	"app.zoom.us":   {Name: "Zoom", Type: "website", Category: "会议"},
	"meeting.tencent.com": {Name: "腾讯会议", Type: "website", Category: "会议"},

	// AI 类
	"www.doubao.com":     {Name: "豆包", Type: "website", Category: "AI"},
	"doubao.com":         {Name: "豆包", Type: "website", Category: "AI"},
	"www.kimi.com":       {Name: "Kimi", Type: "website", Category: "AI"},
	"kimi.moonshot.cn":   {Name: "Kimi", Type: "website", Category: "AI"},
	"tongyi.aliyun.com":  {Name: "通义千问", Type: "website", Category: "AI"},
	"qianwen.aliyun.com": {Name: "通义千问", Type: "website", Category: "AI"},
	"yiyan.baidu.com":    {Name: "文心一言", Type: "website", Category: "AI"},
	"chat.openai.com":    {Name: "ChatGPT", Type: "website", Category: "AI"},
	"chatgpt.com":        {Name: "ChatGPT", Type: "website", Category: "AI"},
	"www.deepseek.com":   {Name: "DeepSeek", Type: "website", Category: "AI"},
	"chat.deepseek.com":  {Name: "DeepSeek", Type: "website", Category: "AI"},
	"yuanbao.tencent.com": {Name: "元宝", Type: "website", Category: "AI"},
	"www.coze.com":       {Name: "Coze", Type: "website", Category: "AI"},
	"claude.ai":          {Name: "Claude", Type: "website", Category: "AI"},

	// 文件转换类
	"audio2edit.com":       {Name: "Audio2Edit", Type: "website", Category: "文件转换"},
	"www.audio2edit.com":   {Name: "Audio2Edit", Type: "website", Category: "文件转换"},
	"smallpdf.com":         {Name: "SmallPDF", Type: "website", Category: "文件转换"},
	"www.smallpdf.com":     {Name: "SmallPDF", Type: "website", Category: "文件转换"},
	"www.ilovepdf.com":     {Name: "iLovePDF", Type: "website", Category: "文件转换"},
	"www.online-convert.com": {Name: "Online Convert", Type: "website", Category: "文件转换"},
}

// 敏感文件扩展名
var SensitiveFileExtensions = []string{
	".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
	".pdf", ".txt", ".csv", ".json", ".xml",
	".zip", ".rar", ".7z", ".tar", ".gz",
	".mp3", ".mp4", ".wav", ".avi", ".mov",
	".jpg", ".jpeg", ".png", ".gif", ".bmp",
	".pem", ".key", ".crt", ".cer", ".p12",
}

// 敏感文件关键词（文件名包含这些关键词则视为敏感）
var SensitiveFileKeywords = []string{
	"机密", "绝密", "内部", "秘密", "confidential", "secret", "private",
	"合同", "协议", "发票", "工资", "薪酬", "salary", "contract",
	"战略", "规划", "预算", "财务", "报表", "budget", "finance",
	"员工", "人事", "简历", "resume", "hr",
	"密码", "password", "key", "token", "accesskey", "credential",
	"专利", "patent", "算法", "algorithm", "核心", "core",
	"会议纪要", "会议记录", "meeting",
	"客户", "customer", "client", "名单", "list",
}

// 风险操作类型
const (
	RiskTypeFileUpload     = "文件上传"
	RiskTypeFileSend       = "文件发送"
	RiskTypeFileDrag       = "文件拖拽"
	RiskTypeCopyPaste      = "复制粘贴"
	RiskTypeScreenShare    = "屏幕共享"
	RiskTypeAppSwitch      = "应用切换"
	RiskTypeWebsiteVisit   = "访问敏感网站"
	RiskTypeFileOpen       = "打开敏感文件"
)
