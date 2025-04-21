import logging
import sys
import builtins
from types import ModuleType


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
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any existing root configuration
    )

    # Optionally patch built‑in print so old calls still reach the log.
    if redirect_print:
        original_print = builtins.print

        def _print_to_log(*args, **kwargs):  # type: ignore[var-annotated]
            message = " ".join(str(a) for a in args)
            logging.getLogger("print").info(message)

        _print_to_log.__doc__ = original_print.__doc__  # preserve docstring
        builtins.print = _print_to_log  # type: ignore[assignment] 