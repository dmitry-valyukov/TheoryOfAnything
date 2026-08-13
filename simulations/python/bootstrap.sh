#!/usr/bin/env bash
# Разовая настройка окружения на чистой Linux/macOS-машине:  ./bootstrap.sh
# Предустановленный Python НЕ нужен: uv скачает версию из .python-version сам.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv не найден, ставлю..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv: $(uv --version)"

if [ -f uv.lock ]; then
    echo "Восстанавливаю окружение строго по uv.lock..."
    uv sync --locked
else
    echo "uv.lock отсутствует — создаю его (это делается один раз)..."
    uv lock
    uv sync
    echo "Не забудьте закоммитить uv.lock!"
fi

echo
echo "Проверка:"
uv run pytest

echo
echo "Запуск симуляций:  uv run python Photon.py"
