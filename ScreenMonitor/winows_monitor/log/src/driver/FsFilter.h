#ifndef FS_FILTER_H
#define FS_FILTER_H

#define _KERNEL_MODE
#include <fltKernel.h>
#include <ntstrsafe.h>
#include <ntddk.h>
#include "../common/SharedDefs.h"

// Pool tags
#define FSFILTER_TAG 'tsFM'  // 'MFst' in little-endian

// Global Data
typedef struct _FSFILTER_DATA {
    PFLT_FILTER Filter;                      // Filter handle
    PFLT_PORT ServerPort;                    // Server communication port
    PFLT_PORT ClientPort;                    // Client port
    BOOLEAN MonitoringEnabled;               // Monitoring state
    KSPIN_LOCK PortLock;                     // Protect port access
} FSFILTER_DATA, *PFSFILTER_DATA;

// Global instance
extern FSFILTER_DATA g_FilterData;

// Driver Functions
NTSTATUS DriverEntry(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PUNICODE_STRING RegistryPath
);

NTSTATUS FsFilterUnload(
    _In_ FLT_FILTER_UNLOAD_FLAGS Flags
);

// Instance Callbacks
NTSTATUS FsFilterInstanceSetup(
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _In_ FLT_INSTANCE_SETUP_FLAGS Flags,
    _In_ DEVICE_TYPE VolumeDeviceType,
    _In_ FLT_FILESYSTEM_TYPE VolumeFilesystemType
);

// Pre/Post Operation Callbacks
FLT_PREOP_CALLBACK_STATUS FsFilterPreCreate(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Flt_CompletionContext_Outptr_ PVOID* CompletionContext
);

FLT_POSTOP_CALLBACK_STATUS FsFilterPostCreate(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _In_opt_ PVOID CompletionContext,
    _In_ FLT_POST_OPERATION_FLAGS Flags
);

FLT_PREOP_CALLBACK_STATUS FsFilterPreWrite(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Flt_CompletionContext_Outptr_ PVOID* CompletionContext
);

FLT_PREOP_CALLBACK_STATUS FsFilterPreSetInfo(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Flt_CompletionContext_Outptr_ PVOID* CompletionContext
);

// Communication Port Callbacks
NTSTATUS FsFilterPortConnect(
    _In_ PFLT_PORT ClientPort,
    _In_opt_ PVOID ServerPortCookie,
    _In_reads_bytes_opt_(SizeOfContext) PVOID ConnectionContext,
    _In_ ULONG SizeOfContext,
    _Outptr_result_maybenull_ PVOID* ConnectionPortCookie
);

VOID FsFilterPortDisconnect(
    _In_opt_ PVOID ConnectionCookie
);

NTSTATUS FsFilterPortMessage(
    _In_opt_ PVOID PortCookie,
    _In_reads_bytes_opt_(InputBufferLength) PVOID InputBuffer,
    _In_ ULONG InputBufferLength,
    _Out_writes_bytes_to_opt_(OutputBufferLength, *ReturnOutputBufferLength) PVOID OutputBuffer,
    _In_ ULONG OutputBufferLength,
    _Out_ PULONG ReturnOutputBufferLength
);

// Helper Functions
VOID SendLogToUserMode(
    _In_ PLOG_RECORD LogRecord
);

NTSTATUS GetProcessImageName(
    _In_ HANDLE ProcessId,
    _Out_writes_(MAX_PROCESS_NAME_LEN) PWCHAR ProcessName
);

#endif // FS_FILTER_H
