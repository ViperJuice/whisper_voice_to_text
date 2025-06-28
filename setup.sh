#!/usr/bin/env bash

# Universal installer for the Voice to Text project
# Detects the operating system and calls the appropriate install script.

set -e

case "$(uname)" in
    Linux*)
        SCRIPT="install-linux.sh";;
    Darwin*)
        SCRIPT="install-mac.sh";;
    CYGWIN*|MINGW*|MSYS*)
        SCRIPT="install-windows.bat";;
    *)
        echo "Unsupported OS: $(uname)" >&2
        exit 1;;
esac

if [ ! -f "$SCRIPT" ]; then
    echo "Installation script $SCRIPT not found in project root." >&2
    exit 1
fi

if [[ "$SCRIPT" == *.bat ]]; then
    cmd.exe /c "$SCRIPT"
else
    chmod +x "$SCRIPT"
    if [ "$SCRIPT" = "install-linux.sh" ] && [ "$EUID" -ne 0 ]; then
        sudo ./$SCRIPT
    else
        ./$SCRIPT
    fi
fi
