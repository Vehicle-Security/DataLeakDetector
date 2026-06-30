// EtwMonitor.cpp - Browser File Access Monitor using ETW Kernel File I/O Events
// Monitors file access events from browser processes

#define _WIN32_WINNT 0x0601
#define INITGUID
#include <windows.h>
#include <evntrace.h>
#include <evntcons.h>
#include <tdh.h>
#include <psapi.h>
#include <aclapi.h>
#include <sddl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cwctype>
#include <cstring>
#include <vector>
#include <unordered_map>

#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "tdh.lib")
#pragma comment(lib, "psapi.lib")

// Session name
#define SESSION_NAME L"BrowserFileMonitor"

// Global handles
TRACEHANDLE g_SessionHandle = 0;
TRACEHANDLE g_ConsumerHandle = 0;
std::ofstream g_LogFile;
volatile bool g_Running = true;
HANDLE g_hStopEvent = NULL;
std::unordered_map<std::wstring, std::wstring> g_FileObjectMap;

// Get current timestamp as string
std::string GetTimestamp() {
    SYSTEMTIME st;
    GetLocalTime(&st);
    std::ostringstream oss;
    oss << std::setfill('0')
        << st.wYear << "-"
        << std::setw(2) << st.wMonth << "-"
        << std::setw(2) << st.wDay << " "
        << std::setw(2) << st.wHour << ":"
        << std::setw(2) << st.wMinute << ":"
        << std::setw(2) << st.wSecond;
    return oss.str();
}

// Convert wide string to UTF-8
std::string WideToUtf8(const std::wstring& wide) {
    if (wide.empty()) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, NULL, 0, NULL, NULL);
    if (len <= 0) return "";
    std::string utf8(len - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, &utf8[0], len, NULL, NULL);
    return utf8;
}

// Escape JSON string
std::string EscapeJson(const std::string& s) {
    std::string result;
    result.reserve(s.size() * 2);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    sprintf_s(buf, "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

// Get process name by PID
std::wstring GetProcessName(DWORD pid) {
    std::wstring name = L"Unknown";
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (hProcess) {
        wchar_t buffer[MAX_PATH] = {0};
        if (GetModuleBaseNameW(hProcess, NULL, buffer, MAX_PATH)) {
            name = buffer;
        }
        CloseHandle(hProcess);
    }
    return name;
}

// Check if process is a browser
bool IsBrowser(const std::wstring& processName) {
    std::wstring lower = processName;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::towlower);
    return lower == L"chrome.exe" || 
           lower == L"msedge.exe" || 
           lower == L"firefox.exe" || 
           lower == L"opera.exe" ||
           lower == L"brave.exe" ||
           lower == L"vivaldi.exe" ||
           lower == L"iexplore.exe";
}

// Convert device path to DOS path (e.g., \Device\HarddiskVolume3\... -> C:\...)
std::wstring DeviceToDosPath(const std::wstring& devicePath) {
    if (devicePath.find(L"\\Device\\") != 0) return devicePath;
    
    wchar_t drives[512];
    if (GetLogicalDriveStringsW(512, drives) == 0) return devicePath;
    
    wchar_t* drive = drives;
    while (*drive) {
        wchar_t driveLetter[3] = { drive[0], L':', L'\0' };
        wchar_t deviceName[MAX_PATH];
        
        if (QueryDosDeviceW(driveLetter, deviceName, MAX_PATH)) {
            size_t deviceLen = wcslen(deviceName);
            if (_wcsnicmp(devicePath.c_str(), deviceName, deviceLen) == 0 &&
                (devicePath[deviceLen] == L'\\' || devicePath[deviceLen] == L'\0')) {
                return driveLetter + devicePath.substr(deviceLen);
            }
        }
        drive += wcslen(drive) + 1;
    }
    return devicePath;
}

// Write event to JSON log
std::wstring ToLower(const std::wstring& value) {
    std::wstring lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::towlower);
    return lower;
}

std::wstring GetInfoString(PTRACE_EVENT_INFO pInfo, ULONG offset) {
    if (!pInfo || offset == 0) return L"";
    return std::wstring((LPCWSTR)((PBYTE)pInfo + offset));
}

std::string ClassifyOperation(PEVENT_RECORD pEvent, PTRACE_EVENT_INFO pInfo) {
    UCHAR opcode = pEvent->EventHeader.EventDescriptor.Opcode;
    std::wstring taskName = ToLower(GetInfoString(pInfo, pInfo->TaskNameOffset));
    std::wstring opcodeName = ToLower(GetInfoString(pInfo, pInfo->OpcodeNameOffset));
    std::wstring combined = taskName + L" " + opcodeName;

    if (combined.find(L"close") != std::wstring::npos ||
        combined.find(L"cleanup") != std::wstring::npos ||
        opcode == 65) {
        return "closed";
    }
    if (combined.find(L"read") != std::wstring::npos || opcode == 67) {
        return "read";
    }
    if (combined.find(L"create") != std::wstring::npos ||
        combined.find(L"open") != std::wstring::npos ||
        opcode == 12 || opcode == 64 || opcode == 32) {
        return "opened";
    }
    return "";
}

void LogEvent(const std::wstring& process, DWORD pid, const std::wstring& path, const std::string& operation) {
    if (!g_LogFile.is_open()) return;
    
    // Convert device path to DOS path
    std::wstring dosPath = DeviceToDosPath(path);
    
    std::string processUtf8 = EscapeJson(WideToUtf8(process));
    std::string pathUtf8 = EscapeJson(WideToUtf8(dosPath));
    std::string timestamp = GetTimestamp();
    
    g_LogFile << "{\"timestamp\": \"" << timestamp 
              << "\",\"process\": \"" << processUtf8 
              << "\",\"pid\": " << pid 
              << ",\"path\": \"" << pathUtf8
              << "\",\"operation\": \"" << operation << "\"}\n";
    g_LogFile.flush();
}

// Get property value as string from ETW event
std::wstring GetStringProperty(PEVENT_RECORD pEvent, PTRACE_EVENT_INFO pInfo, LPCWSTR propertyName) {
    for (ULONG i = 0; i < pInfo->PropertyCount; i++) {
        LPCWSTR propName = (LPCWSTR)((PBYTE)pInfo + pInfo->EventPropertyInfoArray[i].NameOffset);
        if (_wcsicmp(propName, propertyName) == 0) {
            PROPERTY_DATA_DESCRIPTOR dataDesc;
            ZeroMemory(&dataDesc, sizeof(dataDesc));
            dataDesc.PropertyName = (ULONGLONG)propName;
            dataDesc.ArrayIndex = ULONG_MAX;
            
            ULONG size = 0;
            if (TdhGetPropertySize(pEvent, 0, NULL, 1, &dataDesc, &size) != ERROR_SUCCESS) return L"";
            if (size == 0) return L"";
            
            std::vector<BYTE> buffer(size);
            if (TdhGetProperty(pEvent, 0, NULL, 1, &dataDesc, size, buffer.data()) != ERROR_SUCCESS) return L"";
            
            return std::wstring((LPCWSTR)buffer.data());
        }
    }
    return L"";
}

std::wstring GetPointerPropertyKey(PEVENT_RECORD pEvent, PTRACE_EVENT_INFO pInfo, LPCWSTR propertyName) {
    for (ULONG i = 0; i < pInfo->PropertyCount; i++) {
        LPCWSTR propName = (LPCWSTR)((PBYTE)pInfo + pInfo->EventPropertyInfoArray[i].NameOffset);
        if (_wcsicmp(propName, propertyName) == 0) {
            PROPERTY_DATA_DESCRIPTOR dataDesc;
            ZeroMemory(&dataDesc, sizeof(dataDesc));
            dataDesc.PropertyName = (ULONGLONG)propName;
            dataDesc.ArrayIndex = ULONG_MAX;

            ULONG size = 0;
            if (TdhGetPropertySize(pEvent, 0, NULL, 1, &dataDesc, &size) != ERROR_SUCCESS) return L"";
            if (size == 0 || size > sizeof(ULONGLONG)) return L"";

            BYTE buffer[sizeof(ULONGLONG)] = {0};
            if (TdhGetProperty(pEvent, 0, NULL, 1, &dataDesc, size, buffer) != ERROR_SUCCESS) return L"";

            ULONGLONG value = 0;
            memcpy(&value, buffer, size);
            std::wstringstream ss;
            ss << std::hex << value;
            return ss.str();
        }
    }
    return L"";
}

// Event callback function
VOID WINAPI EventRecordCallback(PEVENT_RECORD pEvent) {
    if (!g_Running) return;
    if (!pEvent || !pEvent->UserData) return;
    
    // Get process info
    DWORD pid = pEvent->EventHeader.ProcessId;
    std::wstring processName = GetProcessName(pid);
    
    // Only track browser processes
    if (!IsBrowser(processName)) return;
    
    // Get event information
    DWORD bufferSize = 0;
    if (TdhGetEventInformation(pEvent, 0, NULL, NULL, &bufferSize) != ERROR_INSUFFICIENT_BUFFER) return;
    
    std::vector<BYTE> buffer(bufferSize);
    PTRACE_EVENT_INFO pInfo = reinterpret_cast<PTRACE_EVENT_INFO>(buffer.data());
    
    if (TdhGetEventInformation(pEvent, 0, NULL, pInfo, &bufferSize) != ERROR_SUCCESS) return;
    
    std::string operation = ClassifyOperation(pEvent, pInfo);
    if (operation.empty()) return;

    // Try different property names for file path
    std::wstring fileName = GetStringProperty(pEvent, pInfo, L"FileName");
    if (fileName.empty()) fileName = GetStringProperty(pEvent, pInfo, L"OpenPath");
    if (fileName.empty()) fileName = GetStringProperty(pEvent, pInfo, L"FilePath");
    std::wstring fileObjectKey = GetPointerPropertyKey(pEvent, pInfo, L"FileObject");

    if (!fileObjectKey.empty() && !fileName.empty()) {
        g_FileObjectMap[fileObjectKey] = fileName;
    }

    if (fileName.empty() && operation == "closed" && !fileObjectKey.empty()) {
        auto it = g_FileObjectMap.find(fileObjectKey);
        if (it != g_FileObjectMap.end()) {
            fileName = it->second;
        }
    }
    
    // Skip if no file name
    if (fileName.empty()) return;
    
    // Log the event
    LogEvent(processName, pid, fileName, operation);

    if (operation == "closed" && !fileObjectKey.empty()) {
        g_FileObjectMap.erase(fileObjectKey);
    }
    
    // Print to console (for debug)
    // std::wcout << L"[" << processName << L":" << pid << L"] " << fileName << std::endl;
}

// Ctrl+C handler
BOOL WINAPI ConsoleHandler(DWORD ctrlType) {
    if (ctrlType == CTRL_C_EVENT || ctrlType == CTRL_BREAK_EVENT || ctrlType == CTRL_CLOSE_EVENT) {
        std::wcout << L"\nReceived stop signal via ConsoleHandler..." << std::endl;
        if (g_hStopEvent) {
            SetEvent(g_hStopEvent);
        }
        return TRUE;
    }
    return FALSE;
}

// Helper to create World Access Security Attributes
PSECURITY_ATTRIBUTES CreateWorldAccessSA() {
    PSECURITY_ATTRIBUTES pSA = (PSECURITY_ATTRIBUTES)LocalAlloc(LPTR, sizeof(SECURITY_ATTRIBUTES));
    pSA->nLength = sizeof(SECURITY_ATTRIBUTES);
    pSA->bInheritHandle = FALSE;
    
    // SDDL String: D:(A;;GA;;;WD) -> DACL: Allow Generic All to World (Everyone)
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            L"D:(A;;GA;;;WD)", 
            SDDL_REVISION_1, 
            &(pSA->lpSecurityDescriptor), 
            NULL)) {
        std::wcerr << L"Failed to create security descriptor: " << GetLastError() << std::endl;
        LocalFree(pSA);
        return NULL;
    }
        
    return pSA;
}

// Thread function for ProcessTrace
DWORD WINAPI TraceThreadProc(LPVOID lpParam) {
    if (g_ConsumerHandle != INVALID_PROCESSTRACE_HANDLE && g_ConsumerHandle != 0) {
       std::wcout << L"Processing Trace in background thread..." << std::endl;
       ULONG status = ProcessTrace(&g_ConsumerHandle, 1, 0, 0);
       if (status != ERROR_SUCCESS && status != ERROR_CANCELLED) {
           std::wcerr << L"ProcessTrace failed with " << status << std::endl;
       }
       std::wcout << L"Trace thread exiting." << std::endl;
    }
    return 0;
}

int wmain(int argc, wchar_t* argv[]) {
    // Enable UTF-8 output
    SetConsoleOutputCP(CP_UTF8);

    std::wcout << L"=== ETW Browser File Monitor ===" << std::endl;
    std::wcout << L"Requires Administrator privileges" << std::endl << std::endl;
    
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
    
    // Create Named Event for stopping
    PSECURITY_ATTRIBUTES pSA = CreateWorldAccessSA();
    g_hStopEvent = CreateEventW(pSA, TRUE, FALSE, L"Global\\EtwMonitorStopEvent");
    if (g_hStopEvent == NULL) {
        std::wcerr << L"Failed to create stop event: " << GetLastError() << std::endl;
        // Proceed without event? No, we need it for robust shutdown.
        // But maybe fall back to basic mode if fails? 
        // For now, just continue, fallback to Ctrl+C.
    } else {
        std::wcout << L"Listening for stop signal on Global\\EtwMonitorStopEvent" << std::endl;
    }

    // Determine output directory
    std::wstring outputDir = L"logs";
    if (argc > 1) {
        outputDir = argv[1];
        std::wcout << L"Output directory: " << outputDir << std::endl;
    }
    
    // Create output directory if it doesn't exist
    CreateDirectoryW(outputDir.c_str(), NULL);
    
    // Generate log filename with timestamp
    SYSTEMTIME st;
    GetLocalTime(&st);
    wchar_t logPath[MAX_PATH];
    swprintf_s(logPath, L"%s\\etw_session_%04d%02d%02d_%02d%02d%02d.json",
               outputDir.c_str(), st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    
    g_LogFile.open(logPath, std::ios::out | std::ios::trunc);
    if (!g_LogFile.is_open()) {
        std::wcerr << L"ERROR: Cannot create log file: " << logPath << std::endl;
        return 1;
    }
    std::wcout << L"Log: " << logPath << std::endl;
    
    g_LogFile << "{\"event\": \"monitor_started\", \"timestamp\": \"" << GetTimestamp() << "\"}\n";
    g_LogFile.flush();
    
    // Calculate buffer size for trace properties
    ULONG bufferSize = sizeof(EVENT_TRACE_PROPERTIES) + (wcslen(SESSION_NAME) + 1) * sizeof(wchar_t) + 256;
    std::vector<BYTE> propBuffer(bufferSize, 0);
    PEVENT_TRACE_PROPERTIES pProps = reinterpret_cast<PEVENT_TRACE_PROPERTIES>(propBuffer.data());
    
    // Stop any existing session first
    pProps->Wnode.BufferSize = bufferSize;
    ControlTraceW(0, SESSION_NAME, pProps, EVENT_TRACE_CONTROL_STOP);
    
    // Setup trace properties for Kernel Logger
    ZeroMemory(propBuffer.data(), bufferSize);
    pProps->Wnode.BufferSize = bufferSize;
    pProps->Wnode.Flags = WNODE_FLAG_TRACED_GUID;
    pProps->Wnode.ClientContext = 1;  // QPC timestamp
    pProps->Wnode.Guid = SystemTraceControlGuid;
    // Use bufffer mode to reduce overhead, but flush often?
    // Using EVENT_TRACE_REAL_TIME_MODE means we get events directly.
    pProps->LogFileMode = EVENT_TRACE_REAL_TIME_MODE; 
    pProps->EnableFlags = EVENT_TRACE_FLAG_FILE_IO | EVENT_TRACE_FLAG_FILE_IO_INIT;
    pProps->LoggerNameOffset = sizeof(EVENT_TRACE_PROPERTIES);
    
    // Start as NT Kernel Logger for file events
    ULONG status = StartTraceW(&g_SessionHandle, KERNEL_LOGGER_NAMEW, pProps);
    if (status == ERROR_ALREADY_EXISTS) {
        ControlTraceW(0, KERNEL_LOGGER_NAMEW, pProps, EVENT_TRACE_CONTROL_STOP);
        Sleep(500);
        ZeroMemory(propBuffer.data(), bufferSize);
        pProps->Wnode.BufferSize = bufferSize;
        pProps->Wnode.Flags = WNODE_FLAG_TRACED_GUID;
        pProps->Wnode.ClientContext = 1;
        pProps->Wnode.Guid = SystemTraceControlGuid;
        pProps->LogFileMode = EVENT_TRACE_REAL_TIME_MODE;
        pProps->EnableFlags = EVENT_TRACE_FLAG_FILE_IO | EVENT_TRACE_FLAG_FILE_IO_INIT;
        pProps->LoggerNameOffset = sizeof(EVENT_TRACE_PROPERTIES);
        status = StartTraceW(&g_SessionHandle, KERNEL_LOGGER_NAMEW, pProps);
    }
    
    if (status != ERROR_SUCCESS) {
        std::wcerr << L"ERROR: StartTrace failed with " << status << std::endl;
        if (status == ERROR_ACCESS_DENIED) {
            std::wcerr << L"Please run as Administrator!" << std::endl;
        }
        g_LogFile.close();
        if (g_hStopEvent) CloseHandle(g_hStopEvent);
        if (pSA && pSA->lpSecurityDescriptor) LocalFree(pSA->lpSecurityDescriptor);
        if (pSA) LocalFree(pSA);
        return 1;
    }
    std::wcout << L"Kernel trace started." << std::endl;
    
    // Open trace consumer
    EVENT_TRACE_LOGFILEW logFile = {0};
    logFile.LoggerName = const_cast<LPWSTR>(KERNEL_LOGGER_NAMEW);
    logFile.ProcessTraceMode = PROCESS_TRACE_MODE_REAL_TIME | PROCESS_TRACE_MODE_EVENT_RECORD;
    logFile.EventRecordCallback = EventRecordCallback;
    
    g_ConsumerHandle = OpenTraceW(&logFile);
    if (g_ConsumerHandle == INVALID_PROCESSTRACE_HANDLE) {
        std::wcerr << L"ERROR: OpenTrace failed with " << GetLastError() << std::endl;
        ControlTraceW(g_SessionHandle, KERNEL_LOGGER_NAMEW, pProps, EVENT_TRACE_CONTROL_STOP);
        g_LogFile.close();
        if (g_hStopEvent) CloseHandle(g_hStopEvent);
        return 1;
    }
    std::wcout << L"Trace consumer opened." << std::endl;
    
    std::wcout << L"\nMonitoring browser file access...\n";
    std::wcout << L"Press Ctrl+C to stop\n";
    std::wcout << L"=================================" << std::endl;
    
    // Start Trace Thread
    HANDLE hThread = CreateThread(NULL, 0, TraceThreadProc, NULL, 0, NULL);
    if (hThread == NULL) {
         std::wcerr << L"ERROR: Failed to create thread: " << GetLastError() << std::endl;
         CloseTrace(g_ConsumerHandle);
         // Cleanup...
         return 1;
    }

    // Wait for Stop Event OR Trace Thread Exit
    // If thread exits (e.g. error), we should stop.
    // If stop event signaled, we should stop.
    HANDLE waitHandles[2];
    int handleCount = 0;
    if (g_hStopEvent) waitHandles[handleCount++] = g_hStopEvent;
    waitHandles[handleCount++] = hThread;
    
    WaitForMultipleObjects(handleCount, waitHandles, FALSE, INFINITE);
    
    std::wcout << L"\nStop signal received. shutting down..." << std::endl;
    g_Running = false;
    
    // Cleanup
    // CloseTrace causes ProcessTrace to return (in the thread)
    CloseTrace(g_ConsumerHandle);
    
    // Stop the session
    ControlTraceW(g_SessionHandle, KERNEL_LOGGER_NAMEW, pProps, EVENT_TRACE_CONTROL_STOP);
    
    // Wait for thread to exit
    WaitForSingleObject(hThread, 5000);
    CloseHandle(hThread);
    
    g_LogFile << "{\"event\": \"monitor_stopped\", \"timestamp\": \"" << GetTimestamp() << "\"}\n";
    g_LogFile.close();
    
    if (g_hStopEvent) CloseHandle(g_hStopEvent);
    if (pSA && pSA->lpSecurityDescriptor) LocalFree(pSA->lpSecurityDescriptor);
    if (pSA) LocalFree(pSA);
    
    std::wcout << L"Monitor stopped gracefully." << std::endl;
    return 0;
}
