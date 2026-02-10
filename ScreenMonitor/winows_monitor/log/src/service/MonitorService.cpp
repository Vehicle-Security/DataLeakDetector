#include "MonitorService.h"
#include <sddl.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

// Static members
MonitorService* MonitorService::s_Instance = nullptr;
SERVICE_STATUS MonitorService::s_ServiceStatus = { 0 };
SERVICE_STATUS_HANDLE MonitorService::s_ServiceStatusHandle = nullptr;

//
// Constructor
//
MonitorService::MonitorService()
    : m_hPort(INVALID_HANDLE_VALUE)
    , m_hThread(NULL)
    , m_bRunning(FALSE)
{
    s_Instance = this;
}

//
// Destructor
//
MonitorService::~MonitorService()
{
    Stop();
    s_Instance = nullptr;
}

//
// Install Service
//
BOOL MonitorService::Install()
{
    SC_HANDLE hSCManager = OpenSCManager(NULL, NULL, SC_MANAGER_CREATE_SERVICE);
    if (!hSCManager) {
        wprintf(L"OpenSCManager failed: %d\n", GetLastError());
        return FALSE;
    }

    WCHAR szPath[MAX_PATH];
    GetModuleFileName(NULL, szPath, MAX_PATH);

    SC_HANDLE hService = CreateService(
        hSCManager,
        SERVICE_NAME,
        SERVICE_DISPLAY_NAME,
        SERVICE_ALL_ACCESS,
        SERVICE_WIN32_OWN_PROCESS,
        SERVICE_DEMAND_START,
        SERVICE_ERROR_NORMAL,
        szPath,
        NULL, NULL, NULL, NULL, NULL
    );

    if (!hService) {
        DWORD err = GetLastError();
        if (err != ERROR_SERVICE_EXISTS) {
            wprintf(L"CreateService failed: %d\n", err);
            CloseServiceHandle(hSCManager);
            return FALSE;
        }
    }
    else {
        wprintf(L"Service installed successfully\n");
        CloseServiceHandle(hService);
    }

    CloseServiceHandle(hSCManager);
    return TRUE;
}

//
// Uninstall Service
//
BOOL MonitorService::Uninstall()
{
    SC_HANDLE hSCManager = OpenSCManager(NULL, NULL, SC_MANAGER_CONNECT);
    if (!hSCManager) {
        wprintf(L"OpenSCManager failed: %d\n", GetLastError());
        return FALSE;
    }

    SC_HANDLE hService = OpenService(hSCManager, SERVICE_NAME, SERVICE_STOP | DELETE);
    if (!hService) {
        wprintf(L"OpenService failed: %d\n", GetLastError());
        CloseServiceHandle(hSCManager);
        return FALSE;
    }

    SERVICE_STATUS status;
    ControlService(hService, SERVICE_CONTROL_STOP, &status);

    if (!DeleteService(hService)) {
        wprintf(L"DeleteService failed: %d\n", GetLastError());
        CloseServiceHandle(hService);
        CloseServiceHandle(hSCManager);
        return FALSE;
    }

    wprintf(L"Service uninstalled successfully\n");
    CloseServiceHandle(hService);
    CloseServiceHandle(hSCManager);
    return TRUE;
}

//
// Start Service Logic
//
BOOL MonitorService::Start()
{
    // Logic moved to ServiceMain/WorkerThread
    return TRUE;
}

//
// Stop Service Logic
//
BOOL MonitorService::Stop()
{
    m_bRunning = FALSE;
    if (m_hThread) {
        WaitForSingleObject(m_hThread, 5000);
        CloseHandle(m_hThread);
        m_hThread = NULL;
    }
    return TRUE;
}

//
// Service Main
//
void WINAPI MonitorService::ServiceMain(DWORD argc, LPWSTR* argv)
{
    s_ServiceStatusHandle = RegisterServiceCtrlHandler(SERVICE_NAME, ServiceCtrlHandler);
    if (!s_ServiceStatusHandle) {
        return;
    }

    ZeroMemory(&s_ServiceStatus, sizeof(s_ServiceStatus));
    s_ServiceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    s_ServiceStatus.dwCurrentState = SERVICE_START_PENDING;
    SetServiceStatus(s_ServiceStatusHandle, &s_ServiceStatus);

    // Start the worker thread
    if (s_Instance) {
        s_Instance->m_hThread = CreateThread(NULL, 0, WorkerThread, s_Instance, 0, NULL);
        if (s_Instance->m_hThread) {
            s_ServiceStatus.dwCurrentState = SERVICE_RUNNING;
            s_ServiceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP;
        }
        else {
            s_ServiceStatus.dwCurrentState = SERVICE_STOPPED;
        }
    }

    SetServiceStatus(s_ServiceStatusHandle, &s_ServiceStatus);
}

//
// Service Control Handler
//
void WINAPI MonitorService::ServiceCtrlHandler(DWORD ctrlCode)
{
    if (ctrlCode == SERVICE_CONTROL_STOP) {
        s_ServiceStatus.dwCurrentState = SERVICE_STOP_PENDING;
        SetServiceStatus(s_ServiceStatusHandle, &s_ServiceStatus);

        if (s_Instance) {
            s_Instance->m_bRunning = FALSE;
            if (s_Instance->m_hThread) {
                WaitForSingleObject(s_Instance->m_hThread, 5000);
                CloseHandle(s_Instance->m_hThread);
                s_Instance->m_hThread = NULL;
            }
        }

        s_ServiceStatus.dwCurrentState = SERVICE_STOPPED;
        s_ServiceStatus.dwControlsAccepted = 0;
        SetServiceStatus(s_ServiceStatusHandle, &s_ServiceStatus);
    }
}

//
// Worker Thread
//
DWORD WINAPI MonitorService::WorkerThread(LPVOID param)
{
    MonitorService* pService = static_cast<MonitorService*>(param);
    if (!pService) {
        return 1;
    }

    pService->m_bRunning = TRUE;

    if (!pService->ConnectToDriver()) {
        pService->WriteLog(L"[ERROR] Failed to connect to driver");
        return 1;
    }

    pService->WriteLog(L"[INFO] Monitoring started");

    // Message loop
    while (pService->m_bRunning) {
        LOG_RECORD logRecord;
        DWORD bytesReturned = 0;

        HRESULT hr = FilterGetMessage(
            pService->m_hPort,
            (PFILTER_MESSAGE_HEADER)&logRecord,
            sizeof(LOG_RECORD),
            NULL
        );

        if (SUCCEEDED(hr)) {
            pService->ProcessLogRecord(&logRecord);
        }
        else if (hr == HRESULT_FROM_WIN32(ERROR_INVALID_HANDLE)) {
            pService->WriteLog(L"[ERROR] Driver connection lost");
            break;
        }
    }

    pService->DisconnectFromDriver();
    pService->WriteLog(L"[INFO] Monitoring stopped");
    return 0;
}

//
// Connect to Driver
//
BOOL MonitorService::ConnectToDriver()
{
    HRESULT hr = FilterConnectCommunicationPort(
        FSFILTER_PORT_NAME,
        0,
        NULL,
        0,
        NULL,
        &m_hPort
    );

    return SUCCEEDED(hr);
}

//
// Disconnect from Driver
//
void MonitorService::DisconnectFromDriver()
{
    if (m_hPort != INVALID_HANDLE_VALUE) {
        CloseHandle(m_hPort);
        m_hPort = INVALID_HANDLE_VALUE;
    }
}

//
// Process Log Record
//
void MonitorService::ProcessLogRecord(const LOG_RECORD* pRecord)
{
    if (!pRecord) {
        return;
    }

    // Format: [Timestamp] ProcessPath -> FilePath (Operation)
    std::wstringstream ss;
    ss << L"[" << FormatTimestamp(pRecord->Timestamp) << L"] ";
    ss << pRecord->ProcessName << L" -> ";
    ss << pRecord->FilePath;
    ss << L" (" << GetOperationString(pRecord->Operation) << L")";

    // Check if it's a browser
    if (IsBrowserProcess(pRecord->ProcessName)) {
        ss << L" [BROWSER]";
    }

    WriteLog(ss.str());
}

//
// Write Log to File
//
void MonitorService::WriteLog(const std::wstring& message)
{
    // Ensure log directory exists
    CreateDirectory(L"C:\\Logs", NULL);

    FILE* fp = nullptr;
    _wfopen_s(&fp, LOG_FILE_PATH, L"a+, ccs=UTF-8");
    if (fp) {
        fwprintf(fp, L"%s\n", message.c_str());
        fclose(fp);
    }
}

//
// Check if Browser Process
//
BOOL MonitorService::IsBrowserProcess(const WCHAR* processPath)
{
    if (!processPath || wcslen(processPath) == 0) {
        return FALSE;
    }

    std::wstring path(processPath);
    std::transform(path.begin(), path.end(), path.begin(), ::towlower);

    for (const auto& browser : BROWSER_PROCESSES) {
        if (path.find(browser) != std::wstring::npos) {
            return TRUE;
        }
    }

    return FALSE;
}

//
// Get Operation String
//
std::wstring MonitorService::GetOperationString(OPERATION_TYPE op)
{
    switch (op) {
    case OP_CREATE: return L"CREATE";
    case OP_READ: return L"READ";
    case OP_WRITE: return L"WRITE";
    case OP_DELETE: return L"DELETE";
    case OP_RENAME: return L"RENAME";
    case OP_SETINFO: return L"SETINFO";
    case OP_CLEANUP: return L"CLEANUP";
    default: return L"UNKNOWN";
    }
}

//
// Format Timestamp
//
std::wstring MonitorService::FormatTimestamp(LARGE_INTEGER timestamp)
{
    FILETIME ft;
    SYSTEMTIME st;

    ft.dwLowDateTime = timestamp.LowPart;
    ft.dwHighDateTime = timestamp.HighPart;

    FileTimeToSystemTime(&ft, &st);

    wchar_t buffer[64];
    swprintf_s(buffer, L"%04d-%02d-%02d %02d:%02d:%02d.%03d",
        st.wYear, st.wMonth, st.wDay,
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);

    return std::wstring(buffer);
}

//
// Main Entry Point
//
int wmain(int argc, wchar_t* argv[])
{
    MonitorService service;

    if (argc > 1) {
        if (_wcsicmp(argv[1], L"/install") == 0) {
            return service.Install() ? 0 : 1;
        }
        else if (_wcsicmp(argv[1], L"/uninstall") == 0) {
            return service.Uninstall() ? 0 : 1;
        }
        else if (_wcsicmp(argv[1], L"/test") == 0) {
            // Test mode - run as console app
            wprintf(L"Starting in test mode...\n");
            service.m_bRunning = TRUE;
            if (service.ConnectToDriver()) {
                wprintf(L"Connected to driver. Monitoring...\n");
                
                while (service.m_bRunning) {
                    LOG_RECORD logRecord;
                    HRESULT hr = FilterGetMessage(
                        service.m_hPort,
                        (PFILTER_MESSAGE_HEADER)&logRecord,
                        sizeof(LOG_RECORD),
                        NULL
                    );

                    if (SUCCEEDED(hr)) {
                        service.ProcessLogRecord(&logRecord);
                    }
                    else if (hr == HRESULT_FROM_WIN32(ERROR_INVALID_HANDLE)) {
                        wprintf(L"Driver connection lost\n");
                        break;
                    }
                }
            }
            else {
                wprintf(L"Failed to connect to driver\n");
            }
            return 0;
        }
        else {
            wprintf(L"Usage:\n");
            wprintf(L"  %s /install   - Install service\n", argv[0]);
            wprintf(L"  %s /uninstall - Uninstall service\n", argv[0]);
            wprintf(L"  %s /test      - Run in test mode (console)\n", argv[0]);
            return 1;
        }
    }

    // Run as service
    SERVICE_TABLE_ENTRY serviceTable[] = {
        { (LPWSTR)SERVICE_NAME, MonitorService::ServiceMain },
        { NULL, NULL }
    };

    if (!StartServiceCtrlDispatcher(serviceTable)) {
        wprintf(L"StartServiceCtrlDispatcher failed: %d\n", GetLastError());
        wprintf(L"Use /install, /uninstall, or /test parameters\n");
        return 1;
    }

    return 0;
}
