# Разовая настройка окружения на чистой Windows-машине.
#   powershell -ExecutionPolicy Bypass -File bootstrap.ps1
#
# Предустановленный Python НЕ нужен: uv скачает версию из .python-version сам.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv не найден, ставлю..." -ForegroundColor Yellow
    # официальный установщик Astral; альтернатива: winget install astral-sh.uv
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "uv: $(uv --version)" -ForegroundColor Green

if (Test-Path "uv.lock") {
    Write-Host "Восстанавливаю окружение строго по uv.lock..." -ForegroundColor Cyan
    uv sync --locked
} else {
    Write-Host "uv.lock отсутствует — создаю его (это делается один раз)..." -ForegroundColor Cyan
    uv lock
    uv sync
    Write-Host "Не забудьте закоммитить uv.lock!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Готово. Проверка:" -ForegroundColor Green
uv run pytest

Write-Host ""
Write-Host "Запуск симуляций:" -ForegroundColor Green
Write-Host "  uv run python Photon.py"
Write-Host "  uv run python a1.py"
