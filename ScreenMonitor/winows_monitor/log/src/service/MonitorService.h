#ifndef MONITOR_SERVICE_H
#define MONITOR_SERVICE_H

#include <windows.h>
#include <fltUser.h>
#include <stdio.h>
#include <string>
#include <vector>
#include "../common/SharedDefs.h"

// Service Configuration
#define SERVICE_NAME L"FileMonitorService"
#define SERVICE_DISPLAY_NAME L"File Monitor Service"
#define LOG_FILE_PATH L"C:\\Logs\\FileMonitor.log"

// Browser Process Names (lowercase for comparison)
static const std::vector<std::wstring> BROWSER_PROCESSES = {
    L"chrome.exe",
    L"firefox.exe",
    L"msedge.exe",
    L"iexplore.exe",
    L"opera.exe",
    L"brave.exe",
    L"safari.exe"
};

// MonitorService Class
class MonitorService {
public:
    MonitorService();
    ~MonitorService();

    // Service control
    BOOL Install();
    BOOL Uninstall();
    BOOL Start();
    BOOL Stop();
    
    // Main service routine
    static void WINAPI ServiceMain(DWORD argc, LPWSTR* argv);
    static void WINAPI ServiceCtrlHandler(DWORD ctrlCode);
    
    // Worker thread
    static DWORD WINAPI WorkerThread(LPVOID param);

    // Public members for Test Mode
    HANDLE m_hPort;
    BOOL m_bRunning;
    BOOL ConnectToDriver();
    void ProcessLogRecord(const LOG_RECORD* pRecord);

private:
    static MonitorService* s_Instance;
    static SERVICE_STATUS s_ServiceStatus;
    static SERVICE_STATUS_HANDLE s_ServiceStatusHandle;
    
    HANDLE m_hThread;
    
    // Helper functions
    void DisconnectFromDriver();
    void WriteLog(const std::wstring& message);
    BOOL IsBrowserProcess(const WCHAR* processPath);
    std::wstring GetOperationString(OPERATION_TYPE op);
    std::wstring FormatTimestamp(LARGE_INTEGER timestamp);
};

#endif // MONITOR_SERVICE_H
