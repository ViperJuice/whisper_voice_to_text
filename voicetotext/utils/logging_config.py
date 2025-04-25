import logging
import sys
import builtins
from types import ModuleType
import os


def init_logging(debug: bool = False, redirect_print: bool = True) -> None:
    """Initialise application‑wide logging.

    Parameters
    ----------
    debug : bool
        When *True* the root logger is set to DEBUG, otherwise INFO.
    redirect_print : bool
        When *True* the built‑in *print* function is monkey‑patched to
        forward its messages to *logging.info*.  This gives us backwards
        compatibility with legacy `print()` calls while we migrate the
        codebase to proper logging statements.
    """
    # Prevent multiple initializations
    if hasattr(init_logging, '_initialized'):
        return
    init_logging._initialized = True

    level = logging.DEBUG if debug else logging.INFO

    # Create root logger manually so we can attach separate handlers
    logging.root.handlers.clear()
    logging.root.setLevel(logging.DEBUG)  # always collect everything at root

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    # Console handler – INFO unless debug requested
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logging.root.addHandler(console_handler)

    # File handler – always DEBUG, rotated at 5 MB with 3 backups
    try:
        from logging.handlers import RotatingFileHandler
        log_dir = os.path.join(os.path.expanduser("~"), ".voice_to_text")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "vtt.log")
        
        # Ensure the log file is writable
        if os.path.exists(log_path):
            try:
                with open(log_path, 'a'):
                    pass
            except (IOError, PermissionError):
                # If we can't write to the file, skip file logging
                console_handler.error("Cannot write to log file: %s", log_path)
                return
                
        file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logging.root.addHandler(file_handler)
    except Exception as e:
        console_handler.error("Failed to set up file logging: %s", e)

    # Optionally patch built‑in print so old calls still reach the log.
    if redirect_print:
        original_print = builtins.print

        def _print_to_log(*args, **kwargs):  # type: ignore[var-annotated]
            message = " ".join(str(a) for a in args)
            logging.getLogger("print").debug(message)

        _print_to_log.__doc__ = original_print.__doc__  # preserve docstring
        builtins.print = _print_to_log  # type: ignore[assignment] 