"""Queue delivery receipts without persisting CLI output or credentials."""
import subprocess
from datetime import datetime, timezone


def enqueue(thread, message):
    receipt = {'attempted_at': datetime.now(timezone.utc).isoformat(), 'queued': False}
    try:
        result = subprocess.run(['codex', 'queue', '--thread', thread, '--message', message],
                                capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return receipt | {'error': 'timeout_delivery_unknown'}
    except OSError as error:
        return receipt | {'error': type(error).__name__}
    diagnostic = (result.stdout + result.stderr).decode(errors='replace')
    error = None if result.returncode == 0 else 'queue_command_failed'
    if result.returncode and 'queue storage failed' in diagnostic and '(code: 522)' in diagnostic:
        error = 'queue_storage_sqlite_522'
    return receipt | {'queued': result.returncode == 0, 'returncode': result.returncode, 'error': error}
