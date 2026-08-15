@echo off
rem Файл в UTF-8, а cmd.exe по умолчанию в CP866 -> без этой строки русские
rem сообщения ниже выведутся кракозябрами.
chcp 65001 >nul

rem ---------------------------------------------------------------------------
rem Восстановление окружения, когда проект и кеш uv лежат на РАЗНЫХ дисках.
rem
rem uv по умолчанию раскладывает пакеты в .venv жёсткими ссылками из кеша
rem (%LOCALAPPDATA%\uv\cache, то есть C:). Жёсткая ссылка не пересекает границу
rem тома, поэтому для проекта на M: выбран режим symlink.
rem
rem Зависимости здесь НЕ пересчитываются: --locked ставит ровно то, что записано
rem в uv.lock, и отказывается работать, если лок разошёлся с pyproject.toml.
rem Это и есть смысл файла — воспроизвести окружение один-в-один.
rem
rem МЕНЯТЬ ЗАВИСИМОСТИ ЭТИМ ФАЙЛОМ НЕ НУЖНО:
rem     uv add scipy        - добавить (сам правит pyproject.toml, uv.lock и .venv)
rem     uv remove sympy     - убрать
rem     uv lock --upgrade   - поднять версии в пределах ограничений
rem Отдельная команда `uv lock` почти никогда не нужна: обычные `uv sync` и
rem `uv run` сами пересчитывают лок, если pyproject.toml изменился.
rem
rem Чтобы режим линковки действовал и для этих команд тоже, задайте его
rem один раз для пользователя, а не только внутри этого окна:
rem     setx UV_LINK_MODE symlink
rem либо (надёжнее) перенесите кеш на тот же диск, вернув быстрые жёсткие ссылки:
rem     setx UV_CACHE_DIR "M:\.uv-cache"
rem ---------------------------------------------------------------------------

cd /d "%~dp0"

set UV_LINK_MODE=symlink
uv sync --locked
if not errorlevel 1 goto ok

echo.
echo Первая попытка не удалась.
echo Если в ошибке упомянуты привилегии (os error 1314) - на Windows symlink
echo требует Developer Mode или запуска от администратора. Пробую режим copy:
echo.
uv sync --locked --link-mode=copy
if errorlevel 1 goto fail

:ok
echo.
echo Окружение готово. Проверка:
uv run pytest
goto end

:fail
echo.
echo Не удалось восстановить окружение.
echo Если ошибка про рассинхронизацию лока - выполните `uv lock` и закоммитьте uv.lock.

:end
echo.
rem pause
