#include "FsFilter.h"

// Global data
FSFILTER_DATA g_FilterData = { 0 };

// Operation registration
CONST FLT_OPERATION_REGISTRATION Callbacks[] = {
    {
        IRP_MJ_CREATE,
        0,
        FsFilterPreCreate,
        FsFilterPostCreate
    },
    {
        IRP_MJ_WRITE,
        0,
        FsFilterPreWrite,
        NULL
    },
    {
        IRP_MJ_SET_INFORMATION,
        0,
        FsFilterPreSetInfo,
        NULL
    },
    { IRP_MJ_OPERATION_END }
};

// Registration structure
CONST FLT_REGISTRATION FilterRegistration = {
    sizeof(FLT_REGISTRATION),           // Size
    FLT_REGISTRATION_VERSION,           // Version
    0,                                  // Flags
    NULL,                               // Context
    Callbacks,                          // Operation callbacks
    FsFilterUnload,                     // Unload
    FsFilterInstanceSetup,              // InstanceSetup
    NULL,                               // InstanceQueryTeardown
    NULL,                               // InstanceTeardownStart
    NULL,                               // InstanceTeardownComplete
    NULL, NULL                          // NameProvider callbacks
};

//
// Driver Entry Point
//
NTSTATUS DriverEntry(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PUNICODE_STRING RegistryPath
)
{
    NTSTATUS status;
    PSECURITY_DESCRIPTOR sd = NULL;
    OBJECT_ATTRIBUTES oa;
    UNICODE_STRING portName;

    UNREFERENCED_PARAMETER(RegistryPath);

    DbgPrint("[FsFilter] DriverEntry called\n");

    // Initialize global data
    RtlZeroMemory(&g_FilterData, sizeof(FSFILTER_DATA));
    KeInitializeSpinLock(&g_FilterData.PortLock);
    g_FilterData.MonitoringEnabled = TRUE;

    // Register the minifilter
    status = FltRegisterFilter(DriverObject, &FilterRegistration, &g_FilterData.Filter);
    if (!NT_SUCCESS(status)) {
        DbgPrint("[FsFilter] FltRegisterFilter failed: 0x%X\n", status);
        return status;
    }

    // Create communication port
    RtlInitUnicodeString(&portName, FSFILTER_PORT_NAME);

    status = FltBuildDefaultSecurityDescriptor(&sd, FLT_PORT_ALL_ACCESS);
    if (!NT_SUCCESS(status)) {
        DbgPrint("[FsFilter] FltBuildDefaultSecurityDescriptor failed: 0x%X\n", status);
        FltUnregisterFilter(g_FilterData.Filter);
        return status;
    }

    InitializeObjectAttributes(&oa, &portName, OBJ_KERNEL_HANDLE | OBJ_CASE_INSENSITIVE, NULL, sd);

    status = FltCreateCommunicationPort(
        g_FilterData.Filter,
        &g_FilterData.ServerPort,
        &oa,
        NULL,
        FsFilterPortConnect,
        FsFilterPortDisconnect,
        FsFilterPortMessage,
        1
    );

    FltFreeSecurityDescriptor(sd);

    if (!NT_SUCCESS(status)) {
        DbgPrint("[FsFilter] FltCreateCommunicationPort failed: 0x%X\n", status);
        FltUnregisterFilter(g_FilterData.Filter);
        return status;
    }

    // Start filtering
    status = FltStartFiltering(g_FilterData.Filter);
    if (!NT_SUCCESS(status)) {
        DbgPrint("[FsFilter] FltStartFiltering failed: 0x%X\n", status);
        FltCloseCommunicationPort(g_FilterData.ServerPort);
        FltUnregisterFilter(g_FilterData.Filter);
        return status;
    }

    DbgPrint("[FsFilter] Driver loaded successfully\n");
    return STATUS_SUCCESS;
}

//
// Unload routine
//
NTSTATUS FsFilterUnload(
    _In_ FLT_FILTER_UNLOAD_FLAGS Flags
)
{
    UNREFERENCED_PARAMETER(Flags);

    DbgPrint("[FsFilter] Unloading driver\n");

    if (g_FilterData.ServerPort) {
        FltCloseCommunicationPort(g_FilterData.ServerPort);
    }

    if (g_FilterData.Filter) {
        FltUnregisterFilter(g_FilterData.Filter);
    }

    return STATUS_SUCCESS;
}

//
// Instance Setup
//
NTSTATUS FsFilterInstanceSetup(
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _In_ FLT_INSTANCE_SETUP_FLAGS Flags,
    _In_ DEVICE_TYPE VolumeDeviceType,
    _In_ FLT_FILESYSTEM_TYPE VolumeFilesystemType
)
{
    UNREFERENCED_PARAMETER(FltObjects);
    UNREFERENCED_PARAMETER(Flags);
    UNREFERENCED_PARAMETER(VolumeDeviceType);

    // Only attach to NTFS volumes
    if (VolumeFilesystemType == FLT_FSTYPE_NTFS) {
        DbgPrint("[FsFilter] Attaching to NTFS volume\n");
        return STATUS_SUCCESS;
    }

    return STATUS_FLT_DO_NOT_ATTACH;
}

//
// PreCreate Callback
//
FLT_PREOP_CALLBACK_STATUS FsFilterPreCreate(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Flt_CompletionContext_Outptr_ PVOID* CompletionContext
)
{
    UNREFERENCED_PARAMETER(FltObjects);
    UNREFERENCED_PARAMETER(CompletionContext);

    if (!g_FilterData.MonitoringEnabled) {
        return FLT_PREOP_SUCCESS_NO_CALLBACK;
    }

    // Let PostCreate handle the logging
    return FLT_PREOP_SUCCESS_WITH_CALLBACK;
}

//
// PostCreate Callback
//
FLT_POSTOP_CALLBACK_STATUS FsFilterPostCreate(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _In_opt_ PVOID CompletionContext,
    _In_ FLT_POST_OPERATION_FLAGS Flags
)
{
    LOG_RECORD logRecord = { 0 };
    PFLT_FILE_NAME_INFORMATION nameInfo = NULL;
    NTSTATUS status;

    UNREFERENCED_PARAMETER(CompletionContext);
    UNREFERENCED_PARAMETER(Flags);

    if (!g_FilterData.MonitoringEnabled || !NT_SUCCESS(Data->IoStatus.Status)) {
        return FLT_POSTOP_FINISHED_PROCESSING;
    }

    // Get file name
    status = FltGetFileNameInformation(Data, FLT_FILE_NAME_NORMALIZED | FLT_FILE_NAME_QUERY_DEFAULT, &nameInfo);
    if (NT_SUCCESS(status)) {
        FltParseFileNameInformation(nameInfo);

        // Fill log record
        KeQuerySystemTime(&logRecord.Timestamp);
        logRecord.Operation = OP_CREATE;
        logRecord.ProcessId = (ULONG)PsGetCurrentProcessId();
        logRecord.Status = Data->IoStatus.Status;

        // Copy file path
        if (nameInfo->Name.Length < (MAX_PATH_LEN * sizeof(WCHAR))) {
            RtlCopyMemory(logRecord.FilePath, nameInfo->Name.Buffer, nameInfo->Name.Length);
            logRecord.FilePath[nameInfo->Name.Length / sizeof(WCHAR)] = L'\0';
        }

        // Get process name
        GetProcessImageName((HANDLE)logRecord.ProcessId, logRecord.ProcessName);

        // Send to user-mode
        SendLogToUserMode(&logRecord);

        FltReleaseFileNameInformation(nameInfo);
    }

    return FLT_POSTOP_FINISHED_PROCESSING;
}

//
// PreWrite Callback
//
FLT_PREOP_CALLBACK_STATUS FsFilterPreWrite(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Flt_CompletionContext_Outptr_ PVOID* CompletionContext
)
{
    LOG_RECORD logRecord = { 0 };
    PFLT_FILE_NAME_INFORMATION nameInfo = NULL;
    NTSTATUS status;

    UNREFERENCED_PARAMETER(FltObjects);
    UNREFERENCED_PARAMETER(CompletionContext);

    if (!g_FilterData.MonitoringEnabled) {
        return FLT_PREOP_SUCCESS_NO_CALLBACK;
    }

    // Get file name
    status = FltGetFileNameInformation(Data, FLT_FILE_NAME_NORMALIZED | FLT_FILE_NAME_QUERY_DEFAULT, &nameInfo);
    if (NT_SUCCESS(status)) {
        FltParseFileNameInformation(nameInfo);

        KeQuerySystemTime(&logRecord.Timestamp);
        logRecord.Operation = OP_WRITE;
        logRecord.ProcessId = (ULONG)PsGetCurrentProcessId();
        logRecord.Status = STATUS_SUCCESS;

        if (nameInfo->Name.Length < (MAX_PATH_LEN * sizeof(WCHAR))) {
            RtlCopyMemory(logRecord.FilePath, nameInfo->Name.Buffer, nameInfo->Name.Length);
            logRecord.FilePath[nameInfo->Name.Length / sizeof(WCHAR)] = L'\0';
        }

        GetProcessImageName((HANDLE)logRecord.ProcessId, logRecord.ProcessName);
        SendLogToUserMode(&logRecord);

        FltReleaseFileNameInformation(nameInfo);
    }

    return FLT_PREOP_SUCCESS_NO_CALLBACK;
}

//
// PreSetInfo Callback (for Delete/Rename)
//
FLT_PREOP_CALLBACK_STATUS FsFilterPreSetInfo(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Flt_CompletionContext_Outptr_ PVOID* CompletionContext
)
{
    LOG_RECORD logRecord = { 0 };
    PFLT_FILE_NAME_INFORMATION nameInfo = NULL;
    NTSTATUS status;
    FILE_INFORMATION_CLASS fileInfoClass;

    UNREFERENCED_PARAMETER(FltObjects);
    UNREFERENCED_PARAMETER(CompletionContext);

    if (!g_FilterData.MonitoringEnabled) {
        return FLT_PREOP_SUCCESS_NO_CALLBACK;
    }

    fileInfoClass = Data->Iopb->Parameters.SetFileInformation.FileInformationClass;

    // Check for Delete or Rename
    if (fileInfoClass == FileDispositionInformation || fileInfoClass == FileRenameInformation) {
        status = FltGetFileNameInformation(Data, FLT_FILE_NAME_NORMALIZED | FLT_FILE_NAME_QUERY_DEFAULT, &nameInfo);
        if (NT_SUCCESS(status)) {
            FltParseFileNameInformation(nameInfo);

            KeQuerySystemTime(&logRecord.Timestamp);
            logRecord.Operation = (fileInfoClass == FileDispositionInformation) ? OP_DELETE : OP_RENAME;
            logRecord.ProcessId = (ULONG)PsGetCurrentProcessId();
            logRecord.Status = STATUS_SUCCESS;

            if (nameInfo->Name.Length < (MAX_PATH_LEN * sizeof(WCHAR))) {
                RtlCopyMemory(logRecord.FilePath, nameInfo->Name.Buffer, nameInfo->Name.Length);
                logRecord.FilePath[nameInfo->Name.Length / sizeof(WCHAR)] = L'\0';
            }

            GetProcessImageName((HANDLE)logRecord.ProcessId, logRecord.ProcessName);
            SendLogToUserMode(&logRecord);

            FltReleaseFileNameInformation(nameInfo);
        }
    }

    return FLT_PREOP_SUCCESS_NO_CALLBACK;
}

//
// Port Connect
//
NTSTATUS FsFilterPortConnect(
    _In_ PFLT_PORT ClientPort,
    _In_opt_ PVOID ServerPortCookie,
    _In_reads_bytes_opt_(SizeOfContext) PVOID ConnectionContext,
    _In_ ULONG SizeOfContext,
    _Outptr_result_maybenull_ PVOID* ConnectionPortCookie
)
{
    KIRQL oldIrql;

    UNREFERENCED_PARAMETER(ServerPortCookie);
    UNREFERENCED_PARAMETER(ConnectionContext);
    UNREFERENCED_PARAMETER(SizeOfContext);
    UNREFERENCED_PARAMETER(ConnectionPortCookie);

    DbgPrint("[FsFilter] Client connected\n");

    KeAcquireSpinLock(&g_FilterData.PortLock, &oldIrql);
    g_FilterData.ClientPort = ClientPort;
    KeReleaseSpinLock(&g_FilterData.PortLock, oldIrql);

    return STATUS_SUCCESS;
}

//
// Port Disconnect
//
VOID FsFilterPortDisconnect(
    _In_opt_ PVOID ConnectionCookie
)
{
    KIRQL oldIrql;

    UNREFERENCED_PARAMETER(ConnectionCookie);

    DbgPrint("[FsFilter] Client disconnected\n");

    KeAcquireSpinLock(&g_FilterData.PortLock, &oldIrql);
    if (g_FilterData.ClientPort) {
        FltCloseClientPort(g_FilterData.Filter, &g_FilterData.ClientPort);
        g_FilterData.ClientPort = NULL;
    }
    KeReleaseSpinLock(&g_FilterData.PortLock, oldIrql);
}

//
// Port Message Handler
//
NTSTATUS FsFilterPortMessage(
    _In_opt_ PVOID PortCookie,
    _In_reads_bytes_opt_(InputBufferLength) PVOID InputBuffer,
    _In_ ULONG InputBufferLength,
    _Out_writes_bytes_to_opt_(OutputBufferLength, *ReturnOutputBufferLength) PVOID OutputBuffer,
    _In_ ULONG OutputBufferLength,
    _Out_ PULONG ReturnOutputBufferLength
)
{
    PCOMMAND_MESSAGE cmd;

    UNREFERENCED_PARAMETER(PortCookie);
    UNREFERENCED_PARAMETER(OutputBuffer);
    UNREFERENCED_PARAMETER(OutputBufferLength);

    *ReturnOutputBufferLength = 0;

    if (InputBufferLength < sizeof(COMMAND_MESSAGE)) {
        return STATUS_INVALID_PARAMETER;
    }

    cmd = (PCOMMAND_MESSAGE)InputBuffer;

    switch (cmd->Command) {
    case CMD_START_MONITOR:
        g_FilterData.MonitoringEnabled = TRUE;
        DbgPrint("[FsFilter] Monitoring started\n");
        break;
    case CMD_STOP_MONITOR:
        g_FilterData.MonitoringEnabled = FALSE;
        DbgPrint("[FsFilter] Monitoring stopped\n");
        break;
    default:
        DbgPrint("[FsFilter] Unknown command: %u\n", cmd->Command);
        return STATUS_INVALID_PARAMETER;
    }

    return STATUS_SUCCESS;
}

//
// Send log to user-mode
//
VOID SendLogToUserMode(
    _In_ PLOG_RECORD LogRecord
)
{
    NTSTATUS status;
    KIRQL oldIrql;
    PFLT_PORT clientPort;

    KeAcquireSpinLock(&g_FilterData.PortLock, &oldIrql);
    clientPort = g_FilterData.ClientPort;
    KeReleaseSpinLock(&g_FilterData.PortLock, oldIrql);

    if (!clientPort) {
        return; // No client connected
    }

    status = FltSendMessage(
        g_FilterData.Filter,
        &clientPort,
        LogRecord,
        sizeof(LOG_RECORD),
        NULL,
        NULL,
        NULL
    );

    if (!NT_SUCCESS(status)) {
        DbgPrint("[FsFilter] FltSendMessage failed: 0x%X\n", status);
    }
}

//
// Get process image name
//
NTSTATUS GetProcessImageName(
    _In_ HANDLE ProcessId,
    _Out_writes_(MAX_PROCESS_NAME_LEN) PWCHAR ProcessName
)
{
    NTSTATUS status;
    PEPROCESS process = NULL;
    PUNICODE_STRING processImageName;

    RtlZeroMemory(ProcessName, MAX_PROCESS_NAME_LEN * sizeof(WCHAR));

    status = PsLookupProcessByProcessId(ProcessId, &process);
    if (!NT_SUCCESS(status)) {
        return status;
    }

    status = SeLocateProcessImageName(process, &processImageName);
    if (NT_SUCCESS(status) && processImageName) {
        if (processImageName->Length < (MAX_PROCESS_NAME_LEN * sizeof(WCHAR))) {
            RtlCopyMemory(ProcessName, processImageName->Buffer, processImageName->Length);
            ProcessName[processImageName->Length / sizeof(WCHAR)] = L'\0';
        }
        ExFreePool(processImageName);
    }

    ObDereferenceObject(process);
    return status;
}
