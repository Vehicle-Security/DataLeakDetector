#ifndef SHARED_DEFS_H
#define SHARED_DEFS_H

#ifdef _KERNEL_MODE
#include <fltKernel.h>
#else
#include <windows.h>
#endif

// Communication Port Name
#define FSFILTER_PORT_NAME L"\\FileMonitorPort"

// Maximum lengths
#define MAX_PATH_LEN 512
#define MAX_PROCESS_NAME_LEN 260

// Operation Types
typedef enum _OPERATION_TYPE {
    OP_CREATE = 1,
    OP_READ = 2,
    OP_WRITE = 3,
    OP_DELETE = 4,
    OP_RENAME = 5,
    OP_SETINFO = 6,
    OP_CLEANUP = 7
} OPERATION_TYPE;

// Log Record Structure (sent from Kernel to User-mode)
typedef struct _LOG_RECORD {
    LARGE_INTEGER Timestamp;           // System time
    OPERATION_TYPE Operation;          // Operation type
    ULONG ProcessId;                   // Process ID
    WCHAR ProcessName[MAX_PROCESS_NAME_LEN]; // Process name with path
    WCHAR FilePath[MAX_PATH_LEN];      // Full file path
    NTSTATUS Status;                   // Operation status
    ULONG64 FileSize;                  // File size (if applicable)
} LOG_RECORD, *PLOG_RECORD;

// Command Message Structure (sent from User-mode to Kernel)
typedef struct _COMMAND_MESSAGE {
    ULONG Command;                     // Command code
    ULONG Data;                        // Command-specific data
} COMMAND_MESSAGE, *PCOMMAND_MESSAGE;

// Commands
#define CMD_START_MONITOR 1
#define CMD_STOP_MONITOR 2
#define CMD_ADD_WHITELIST 3
#define CMD_REM_WHITELIST 4

#endif // SHARED_DEFS_H
