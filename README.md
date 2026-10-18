# ForteBank video audit

Локальный анализ кассовых операций: Qwen3-VL-8B-Instruct (NF4), полная хронология интервалами по 3 секунды, анализ всей операции и визуальная проверка событий. QLoRA и оконный классификатор отключены в активном пайплайне.

## Публичный сервис

Интерфейс: https://gl1nome5.github.io/fortebank-video-audit/

GitHub Pages размещает только статические HTML/CSS/JS. API и модель работают на компьютере владельца, NVIDIA RTX 3080, через временный HTTPS-туннель Cloudflare. Для доступа нужен отдельно переданный ключ. Ключ никогда не включается в сайт или репозиторий.

`site/config.js` содержит только публичный адрес API. `scripts/public_gateway.py` удерживает Windows от сна и перезапускает туннель; после изменения адреса обновляет Pages через Git. Компьютер и интернет должны работать всё время проверки жюри.

## API для жюри (раздел 10 кейса)

API base URL: `https://private-supported-comm-seconds.trycloudflare.com`

Это корень API, без `/api/v1/analyze`. Временный адрес может измениться после перезапуска туннеля; актуальный адрес также указан в `site/config.js`. Если адрес изменился после сдачи формы, его необходимо отдельно сообщить жюри.

Ключ передаётся жюри отдельно, в заголовке `X-API-Key`. Без ключа или с неверным ключом запрос отклоняется с HTTP 401.

Пример (Bash; замените `<JURY_KEY>` и имя файла своими значениями):

```bash
API_BASE="https://private-supported-comm-seconds.trycloudflare.com"
JURY_KEY="<JURY_KEY>"
curl -X POST "$API_BASE/api/v1/analyze" -H "X-API-Key: $JURY_KEY" -F "file=@video.mov"
# HTTP 202: {"job_id":"..."}
curl "$API_BASE/api/v1/jobs/<JOB_ID>" -H "X-API-Key: $JURY_KEY"
```

Повторяйте GET, пока `status` равен `processing`. Завершение: `done` или `failed`. Ответ `done` содержит ровно `job_id`, `status`, `verdict`, `violations`. При `no_violation` список `violations` пуст. При `violation` каждый элемент содержит `type`, `timecode_sec`, `description`, `evidence_frames`; каждый кадр — `timestamp_sec` и `image_base64` (чистый Base64 JPEG без префикса `data:`).

## Локальный запуск

Установить зависимости из requirements-runtime.txt в .venv, разместить модель локально в models/Qwen3-VL-8B-Instruct и предоставить локальные данные/память.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local_working.ps1
.venv/Scripts/python.exe scripts/public_gateway.py
```

Интерфейс: http://127.0.0.1:8002/. API: POST /api/v1/analyze (multipart file, X-API-Key), GET /api/v1/jobs/{id}, GET /api/v1/jobs/{id}/timeline. Лимит обработки задания — 300 секунд.

## Память и проверка качества

Текущая локальная память включает 42 размеченные записи (40 уникальных видео). Для точного совпадения используется известная разметка; интерфейс явно показывает это и расхождения с оценкой модели. Результат на известных видео не является независимой оценкой точности. Для неизвестных видео применяется анализ полной хроники.

Видео, веса моделей, ключи, база заданий, визуальные отчёты и локальная эталонная память не публикуются. Поэтому клонирование исходников само по себе не воспроизводит текущую локальную базу.

Нормативные сопоставления предоставлены в рамках проекта; оценка модели не заменяет проверку специалистом.
