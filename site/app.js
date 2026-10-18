(function () {
  'use strict';

  const apiFetch = (path, options) => fetch((window.FORTE_API_BASE || '') + path, options);

  const SESSION_KEY_STORAGE = 'fortebank_audit_api_key';
  const DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024; // 256 MiB
  const DEFAULT_POLL_INTERVAL_MS = 1500;
  const MAX_CONSECUTIVE_POLL_ERRORS = 5;
  const MAX_POLL_DURATION_MS = 5 * 60 * 1000 + 30000; // Server deadline plus polling grace
  const MAX_EVIDENCE_FRAMES_PER_EVENT = 3;

  const STAGE_LABELS_RU = {
    chronological_log: 'Просмотр всей операции по порядку',
    holistic_reasoning: 'Проверка полной истории операции',
    queued: 'В очереди на обработку',
    starting: 'Подготовка анализа',
    warming_up: 'Подготовка модели',
    scene_calibration: 'Определение кассовой зоны',
    uploading: 'Загрузка видеофайла на сервер',
    extracting_frames: 'Извлечение видеокадров',
    preprocessing: 'Предварительная обработка видеопотока',
    analyzing: 'Анализ кассовых операций',
    detecting: 'Поиск потенциальных признаков нарушений',
    verifying: 'Формирование отчета и стоп-кадров',
    finalizing: 'Завершение проверки'
  };

  const VIOLATION_TYPE_LABELS_RU = {
    unauthorized_person: 'Доступ постороннего лица в кассовую зону',
    unattended_cash: 'Оставление наличных денег без присмотра',
    open_safe_unattended: 'Открытое хранилище / сейф без контроля кассира',
    cash_recount_violation: 'Нарушение порядка пересчета наличных денег',
    document_security_violation: 'Нарушение сохранности кассовых документов / ценностей',
    camera_obstruction: 'Перекрытие обзора камеры видеонаблюдения',
    procedure_deviation: 'Отклонение от регламента кассовой операции'
  };

  const state = {
    uiState: 'idle', // idle | invalid | uploading | processing | network_retry | timeout | failed | done_no_violation | done_violation
    maxFileBytes: DEFAULT_MAX_FILE_BYTES,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
    selectedFile: null,
    localBlobUrl: null,
    remoteBlobUrl: null,
    videoPlayable: true,
    activeJobId: null,
    pollTimerId: null,
    pollAbortController: null,
    pollGeneration: 0,
    pollStartedAt: 0,
    consecutivePollErrors: 0,
    modalContext: null,
    healthTimerId: null,
    lastFocusedElement: null
  };

  const dom = {
    serviceHealthBadge: document.getElementById('service-health-badge'),
    serviceHealthText: document.getElementById('service-health-text'),
    configMetaFooter: document.getElementById('config-meta-footer'),

    apiKeyInput: document.getElementById('api-key-input'),
    toggleKeyVisibilityBtn: document.getElementById('toggle-key-visibility-btn'),
    clearKeyBtn: document.getElementById('clear-key-btn'),
    keyStorageBadge: document.getElementById('key-storage-badge'),

    fileLimitBadge: document.getElementById('file-limit-badge'),
    dropZone: document.getElementById('drop-zone'),
    videoFileInput: document.getElementById('video-file-input'),
    selectedFilePanel: document.getElementById('selected-file-panel'),
    selectedFileName: document.getElementById('selected-file-name'),
    selectedFileMeta: document.getElementById('selected-file-meta'),
    removeFileBtn: document.getElementById('remove-file-btn'),

    videoPreviewContainer: document.getElementById('video-preview-container'),
    videoSourceIndicator: document.getElementById('video-source-indicator'),
    videoPlayer: document.getElementById('audit-video-player'),
    videoCodecFallback: document.getElementById('video-codec-fallback'),
    videoSeekFeedback: document.getElementById('video-seek-feedback'),

    submitAuditBtn: document.getElementById('submit-audit-btn'),

    statePill: document.getElementById('state-pill'),
    jobIdDisplay: document.getElementById('job-id-display'),
    jobIdValue: document.getElementById('job-id-value'),

    stateAlertBox: document.getElementById('state-alert-box'),
    stateAlertTitle: document.getElementById('state-alert-title'),
    stateAlertMessage: document.getElementById('state-alert-message'),
    stateAlertActions: document.getElementById('state-alert-actions'),
    retryPollBtn: document.getElementById('retry-poll-btn'),
    resetJobBtn: document.getElementById('reset-job-btn'),

    progressPanel: document.getElementById('progress-panel'),
    progressStageLabel: document.getElementById('progress-stage-label'),
    progressPercentLabel: document.getElementById('progress-percent-label'),
    progressBarTrack: document.getElementById('progress-bar-track'),
    progressBarFill: document.getElementById('progress-bar-fill'),
    progressSubtext: document.getElementById('progress-subtext'),

    idlePlaceholder: document.getElementById('idle-placeholder'),
    verdictSummaryPanel: document.getElementById('verdict-summary-panel'),
    verdictBadge: document.getElementById('verdict-badge'),
    eventsCountBadge: document.getElementById('events-count-badge'),
    verdictExplanation: document.getElementById('verdict-explanation'),
    startNewAuditBtn: document.getElementById('start-new-audit-btn'),

    violationsSection: document.getElementById('violations-section'),
    violationsListCounter: document.getElementById('violations-list-counter'),
    violationsList: document.getElementById('violations-list'),

    evidenceModal: document.getElementById('evidence-modal'),
    modalTitle: document.getElementById('modal-title'),
    modalSubtitle: document.getElementById('modal-subtitle'),
    modalCloseBtn: document.getElementById('modal-close-btn'),
    modalImage: document.getElementById('modal-image'),
    modalCaption: document.getElementById('modal-caption'),
    modalDescription: document.getElementById('modal-description'),
    modalPrevBtn: document.getElementById('modal-prev-btn'),
    modalNextBtn: document.getElementById('modal-next-btn'),
    modalFrameCounter: document.getElementById('modal-frame-counter'),
    modalSeekBtn: document.getElementById('modal-seek-btn')
  };

  /* Utility Functions */

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) {
      return '0 Б';
    }
    if (bytes < 1024) {
      return bytes + ' Б';
    }
    const kb = bytes / 1024;
    if (kb < 1024) {
      return kb.toFixed(1).replace('.', ',') + ' КБ';
    }
    const mb = kb / 1024;
    return mb.toFixed(2).replace('.', ',') + ' МБ';
  }

  function formatTimecode(secondsInput) {
    const num = Number(secondsInput);
    if (!Number.isFinite(num) || num < 0) {
      return '00:00';
    }
    const totalSeconds = Math.floor(num);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    const mm = String(minutes).padStart(2, '0');
    const ss = String(seconds).padStart(2, '0');
    if (hours > 0) {
      const hh = String(hours).padStart(2, '0');
      return hh + ':' + mm + ':' + ss;
    }
    return mm + ':' + ss;
  }

  function isJobBusy() {
    return state.uiState === 'uploading' || state.uiState === 'processing';
  }

  function hasActiveLockedJob() {
    return Boolean(state.activeJobId) && (
      state.uiState === 'uploading' ||
      state.uiState === 'processing' ||
      state.uiState === 'network_retry' ||
      state.uiState === 'timeout'
    );
  }

  /* SessionStorage API Key Handling */

  function getStoredApiKey() {
    try {
      return window.sessionStorage.getItem(SESSION_KEY_STORAGE) || '';
    } catch (_err) {
      return '';
    }
  }

  function setStoredApiKey(value) {
    const trimmed = (value || '').trim();
    try {
      if (trimmed) {
        window.sessionStorage.setItem(SESSION_KEY_STORAGE, trimmed);
      } else {
        window.sessionStorage.removeItem(SESSION_KEY_STORAGE);
      }
    } catch (_err) {
      // Ignore storage quota/security errors in restricted mode
    }
    updateKeyBadge(Boolean(trimmed));
  }

  function getCurrentApiKey() {
    return (dom.apiKeyInput.value || '').trim();
  }

  function updateKeyBadge(hasKey) {
    if (hasKey) {
      dom.keyStorageBadge.textContent = 'Сохранен в сессии';
      dom.keyStorageBadge.className = 'badge badge--ok';
    } else {
      dom.keyStorageBadge.textContent = 'Ключ не задан';
      dom.keyStorageBadge.className = 'badge badge--muted';
    }
  }

  /* Safe Text & Normative URL Rendering */

  function sanitizeHttpUrl(candidate) {
    try {
      const parsed = new URL(candidate);
      if (parsed.protocol === 'https:' || parsed.protocol === 'http:') {
        return parsed.href;
      }
    } catch (_err) {
      return null;
    }
    return null;
  }

  function renderUntrustedTextWithSafeLinks(container, rawText) {
    container.textContent = '';
    const text = typeof rawText === 'string' ? rawText : '';
    if (!text) {
      container.textContent = 'Описание не предоставлено.';
      return;
    }

    const urlRegex = /https?:\/\/[^\s<>")\]]+/gi;
    let lastIndex = 0;
    let match;

    while ((match = urlRegex.exec(text)) !== null) {
      const matchStart = match.index;
      let rawUrl = match[0];
      let trailingPunct = '';

      while (/[.,;:!?]$/.test(rawUrl)) {
        trailingPunct = rawUrl.slice(-1) + trailingPunct;
        rawUrl = rawUrl.slice(0, -1);
      }

      if (matchStart > lastIndex) {
        container.appendChild(document.createTextNode(text.slice(lastIndex, matchStart)));
      }

      const safeHref = sanitizeHttpUrl(rawUrl);
      if (safeHref) {
        const anchor = document.createElement('a');
        anchor.href = safeHref;
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.className = 'normative-link';
        anchor.textContent = rawUrl;
        container.appendChild(anchor);
      } else {
        container.appendChild(document.createTextNode(rawUrl));
      }

      if (trailingPunct) {
        container.appendChild(document.createTextNode(trailingPunct));
      }

      lastIndex = urlRegex.lastIndex;
    }

    if (lastIndex < text.length) {
      container.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
  }

  /* Safe Base64 Evidence Image Validation */

  function normalizeEvidenceImageSrc(imageBase64Input) {
    if (typeof imageBase64Input !== 'string') {
      return null;
    }
    const trimmed = imageBase64Input.trim();
    if (!trimmed) {
      return null;
    }

    if (trimmed.startsWith('data:')) {
      const match = /^data:image\/(jpeg|jpg|png|webp);base64,([A-Za-z0-9+/=\s]+)$/i.exec(trimmed);
      if (!match) {
        return null;
      }
      const subtype = match[1].toLowerCase() === 'jpg' ? 'jpeg' : match[1].toLowerCase();
      const cleanBase64 = match[2].replace(/\s+/g, '');
      if (cleanBase64.length < 16) {
        return null;
      }
      return 'data:image/' + subtype + ';base64,' + cleanBase64;
    }

    // Reject any other scheme (javascript:, http:, file:, etc.)
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)) {
      return null;
    }

    if (!/^[A-Za-z0-9+/=\s]+$/.test(trimmed)) {
      return null;
    }

    const cleanRaw = trimmed.replace(/\s+/g, '');
    if (cleanRaw.length < 16) {
      return null;
    }

    let mime = 'image/jpeg';
    if (cleanRaw.startsWith('iVBORw0KGgo')) {
      mime = 'image/png';
    } else if (cleanRaw.startsWith('UklGR')) {
      mime = 'image/webp';
    } else if (cleanRaw.startsWith('/9j/')) {
      mime = 'image/jpeg';
    }

    return 'data:' + mime + ';base64,' + cleanRaw;
  }

  /* Blob URL Lifecycle */

  function revokeLocalBlobUrl() {
    if (state.localBlobUrl) {
      URL.revokeObjectURL(state.localBlobUrl);
      state.localBlobUrl = null;
    }
  }

  function revokeRemoteBlobUrl() {
    if (state.remoteBlobUrl) {
      URL.revokeObjectURL(state.remoteBlobUrl);
      state.remoteBlobUrl = null;
    }
  }

  function revokeAllBlobUrls() {
    revokeLocalBlobUrl();
    revokeRemoteBlobUrl();
  }

  /* File Validation & Preview */

  function validateMovFile(file) {
    if (!file) {
      return 'Файл не выбран. Пожалуйста, выберите видеозапись в формате .mov.';
    }
    const fileName = (file.name || '').trim();
    const lowerName = fileName.toLowerCase();
    if (!lowerName.endsWith('.mov')) {
      return 'Недопустимый формат файла («' + fileName + '»). Разрешена загрузка только видеофайлов с расширением .mov.';
    }
    if (file.type && file.type !== 'video/quicktime' && !file.type.startsWith('video/')) {
      return 'Недопустимый тип содержимого файла. Ожидается видеофайл контейнера QuickTime (.mov).';
    }
    if (file.size <= 0) {
      return 'Выбранный файл пуст (0 Б). Выберите корректную видеозапись .mov.';
    }
    if (file.size > state.maxFileBytes) {
      const maxMb = Math.round(state.maxFileBytes / (1024 * 1024));
      return 'Размер файла (' + formatBytes(file.size) + ') превышает максимально допустимый предел ' + maxMb + ' МБ.';
    }
    return null;
  }

  function handleFileSelection(file) {
    if (hasActiveLockedJob()) {
      return;
    }

    const validationError = validateMovFile(file);
    if (validationError) {
      clearSelectedFile(false);
      setUiState('invalid', {
        title: 'Ошибка проверки видеофайла',
        message: validationError
      });
      updateControlsInteractivity();
      return;
    }

    revokeAllBlobUrls();
    state.selectedFile = file;
    state.videoPlayable = true;
    state.localBlobUrl = URL.createObjectURL(file);

    dom.selectedFileName.textContent = file.name;
    dom.selectedFileMeta.textContent = 'Размер: ' + formatBytes(file.size) + ' • Формат QuickTime (.mov)';
    dom.selectedFilePanel.hidden = false;

    dom.videoCodecFallback.hidden = true;
    dom.videoSeekFeedback.hidden = true;
    dom.videoSourceIndicator.textContent = 'Локальный файл';
    dom.videoPlayer.src = state.localBlobUrl;
    dom.videoPreviewContainer.hidden = false;

    if (state.uiState === 'invalid') {
      setUiState('idle');
    }
    updateControlsInteractivity();
  }

  function clearSelectedFile(resetToIdle) {
    revokeAllBlobUrls();
    state.selectedFile = null;
    state.videoPlayable = true;
    dom.videoFileInput.value = '';
    dom.selectedFilePanel.hidden = true;
    dom.selectedFileName.textContent = '';
    dom.selectedFileMeta.textContent = '';

    dom.videoPlayer.pause();
    dom.videoPlayer.removeAttribute('src');
    dom.videoPlayer.load();
    dom.videoPreviewContainer.hidden = true;
    dom.videoCodecFallback.hidden = true;
    dom.videoSeekFeedback.hidden = true;

    if (resetToIdle) {
      setUiState('idle');
    }
    updateControlsInteractivity();
  }

  function seekVideoToSeconds(seconds) {
    const targetSec = Number(seconds);
    if (!Number.isFinite(targetSec) || targetSec < 0) {
      return;
    }

    const formatted = formatTimecode(targetSec);
    dom.videoSeekFeedback.hidden = false;

    if (dom.videoPreviewContainer.hidden || !dom.videoPlayer.src) {
      dom.videoSeekFeedback.textContent = 'Выбран таймкод ' + formatted + '. Видеоплеер недоступен.';
      return;
    }

    dom.videoPreviewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    if (!state.videoPlayable || dom.videoPlayer.error) {
      dom.videoSeekFeedback.textContent =
        'Отмечен таймкод ' + formatted + ' (' + Math.floor(targetSec) + ' сек.). Браузер не поддерживает прямое воспроизведение кодека этого файла .mov.';
      return;
    }

    try {
      dom.videoPlayer.currentTime = targetSec;
      dom.videoSeekFeedback.textContent = 'Переход к таймкоду ' + formatted + ' выполнен.';
    } catch (_err) {
      dom.videoSeekFeedback.textContent = 'Запрошен переход к таймкоду ' + formatted + '.';
    }
  }

  /* UI State Machine */

  function updateControlsInteractivity() {
    const locked = hasActiveLockedJob();
    const hasKey = Boolean(getCurrentApiKey());
    const hasFile = Boolean(state.selectedFile);

    dom.submitAuditBtn.disabled = locked || !hasKey || !hasFile;
    dom.removeFileBtn.disabled = locked;
    dom.videoFileInput.disabled = locked;
    dom.apiKeyInput.disabled = isJobBusy();
    dom.clearKeyBtn.disabled = isJobBusy();

    if (locked) {
      dom.dropZone.classList.add('is-disabled');
      dom.dropZone.setAttribute('aria-disabled', 'true');
    } else {
      dom.dropZone.classList.remove('is-disabled');
      dom.dropZone.setAttribute('aria-disabled', 'false');
    }
  }

  function setProgressView(stageCode, progressValue, subtext) {
    const readableStage = (stageCode && STAGE_LABELS_RU[stageCode])
      ? STAGE_LABELS_RU[stageCode]
      : (typeof stageCode === 'string' && stageCode.trim() ? stageCode.trim() : 'Обработка видеозаписи...');

    dom.progressStageLabel.textContent = readableStage;

    if (typeof progressValue === 'number' && Number.isFinite(progressValue)) {
      const clamped = Math.max(0, Math.min(1, progressValue));
      const percent = Math.round(clamped * 100);
      dom.progressPercentLabel.textContent = percent + '%';
      dom.progressBarTrack.setAttribute('aria-valuenow', String(percent));
      dom.progressBarFill.classList.remove('progress-fill--indeterminate');
      dom.progressBarFill.style.width = percent + '%';
    } else {
      dom.progressPercentLabel.textContent = 'В работе';
      dom.progressBarTrack.removeAttribute('aria-valuenow');
      dom.progressBarFill.style.width = '';
      dom.progressBarFill.classList.add('progress-fill--indeterminate');
    }

    if (subtext) {
      dom.progressSubtext.textContent = subtext;
    }
  }

  function setUiState(newState, payload) {
    const info = payload || {};
    state.uiState = newState;

    // Reset dynamic panels
    dom.stateAlertBox.hidden = true;
    dom.stateAlertBox.className = 'alert-box';
    dom.stateAlertActions.hidden = true;
    dom.progressPanel.hidden = true;
    dom.idlePlaceholder.hidden = true;
    dom.verdictSummaryPanel.hidden = true;

    if (state.activeJobId) {
      dom.jobIdDisplay.hidden = false;
      dom.jobIdValue.textContent = state.activeJobId;
    } else {
      dom.jobIdDisplay.hidden = true;
      dom.jobIdValue.textContent = '';
    }

    switch (newState) {
      case 'idle': {
        dom.statePill.textContent = 'Ожидание';
        dom.statePill.className = 'state-pill state-pill--idle';
        dom.idlePlaceholder.hidden = false;
        dom.violationsSection.hidden = true;
        break;
      }

      case 'invalid': {
        dom.statePill.textContent = 'Ошибка данных';
        dom.statePill.className = 'state-pill state-pill--danger';
        dom.stateAlertBox.hidden = false;
        dom.stateAlertTitle.textContent = info.title || 'Проверьте введенные данные';
        dom.stateAlertMessage.textContent = info.message || 'Некорректный файл или ключ доступа API.';
        dom.idlePlaceholder.hidden = false;
        dom.violationsSection.hidden = true;
        break;
      }

      case 'uploading': {
        dom.statePill.textContent = 'Загрузка файла';
        dom.statePill.className = 'state-pill state-pill--active';
        dom.progressPanel.hidden = false;
        dom.violationsSection.hidden = true;
        setProgressView(
          'uploading',
          null,
          'Отправка видеофайла .mov на сервер для постановки задачи в очередь...'
        );
        break;
      }

      case 'processing': {
        dom.statePill.textContent = 'Выполняется анализ';
        dom.statePill.className = 'state-pill state-pill--active';
        dom.progressPanel.hidden = false;
        dom.violationsSection.hidden = true;
        setProgressView(
          info.stage,
          info.progress,
          'Задача обрабатывается. Опрос статуса выполняется автоматически каждые 1,5 сек.'
        );
        break;
      }

      case 'network_retry': {
        dom.statePill.textContent = 'Сбой связи';
        dom.statePill.className = 'state-pill state-pill--warning';
        dom.stateAlertBox.hidden = false;
        dom.stateAlertBox.classList.add('alert-box--warning');
        dom.stateAlertTitle.textContent = info.title || 'Временная ошибка сети при опросе статуса';
        dom.stateAlertMessage.textContent =
          info.message || 'Не удалось получить ответ от сервера. Вы можете повторить запрос статуса текущей задачи без повторной отправки видео.';
        dom.stateAlertActions.hidden = false;
        dom.progressPanel.hidden = false;
        break;
      }

      case 'timeout': {
        dom.statePill.textContent = 'Тайм-аут ожидания';
        dom.statePill.className = 'state-pill state-pill--warning';
        dom.stateAlertBox.hidden = false;
        dom.stateAlertBox.classList.add('alert-box--warning');
        dom.stateAlertTitle.textContent = info.title || 'Превышено время ожидания ответа по задаче';
        dom.stateAlertMessage.textContent =
          info.message || 'Автоматический опрос приостановлен. Нажмите «Повторить запрос статуса», чтобы проверить готовность текущей задачи без повторной загрузки файла.';
        dom.stateAlertActions.hidden = false;
        break;
      }

      case 'failed': {
        // Explicitly never show "no violations" for failed jobs
        dom.statePill.textContent = 'Ошибка анализа';
        dom.statePill.className = 'state-pill state-pill--danger';
        dom.stateAlertBox.hidden = false;
        dom.stateAlertTitle.textContent = info.title || 'Анализ видеозаписи завершился с ошибкой';
        dom.stateAlertMessage.textContent =
          info.message || 'Сервер сообщил о невозможности завершить обработку видеофайла. Заключение об отсутствии нарушений не сформировано.';
        dom.stateAlertActions.hidden = false;
        dom.retryPollBtn.hidden = !info.allowRetryPoll;
        dom.verdictSummaryPanel.hidden = true;
        dom.violationsSection.hidden = true;
        break;
      }

      case 'done_no_violation': {
        dom.statePill.textContent = 'Завершено';
        dom.statePill.className = 'state-pill state-pill--success';
        dom.verdictSummaryPanel.hidden = false;
        dom.verdictBadge.textContent = 'Признаков нарушений не выявлено';
        dom.verdictBadge.className = 'verdict-badge verdict-badge--clean';
        dom.eventsCountBadge.textContent = String(info.eventsCount || 0);
        dom.verdictExplanation.textContent =
          'Автоматизированная проверка завершена успешно. Эпизодов с потенциальными признаками нарушений на предоставленной видеозаписи не зафиксировано.';
        if (info.eventsCount > 0) {
          dom.violationsSection.hidden = false;
        } else {
          dom.violationsSection.hidden = true;
        }
        break;
      }

      case 'done_violation': {
        dom.statePill.textContent = 'Требует проверки';
        dom.statePill.className = 'state-pill state-pill--warning';
        dom.verdictSummaryPanel.hidden = false;
        dom.verdictBadge.textContent = 'Выявлены потенциальные нарушения';
        dom.verdictBadge.className = 'verdict-badge verdict-badge--violation';
        dom.eventsCountBadge.textContent = String(info.eventsCount || 0);
        dom.verdictExplanation.textContent =
          'Зафиксированы визуальные эпизоды, требующие обязательного рассмотрения специалистом кассового контроля. Изучите таймкоды и стоп-кадры ниже.';
        dom.violationsSection.hidden = false;
        break;
      }

      default:
        break;
    }

    if (newState !== 'failed') {
      dom.retryPollBtn.hidden = false;
    }
    updateControlsInteractivity();
  }

  /* Rendering Violations & Evidence */

  function renderViolationsList(violationsArray) {
    dom.violationsList.textContent = '';
    const list = Array.isArray(violationsArray) ? violationsArray : [];
    dom.violationsListCounter.textContent = String(list.length);

    list.forEach(function (item, eventIndex) {
      const rawType = typeof item.type === 'string' && item.type.trim() ? item.type.trim() : 'unclassified_event';
      const readableTitle = VIOLATION_TYPE_LABELS_RU[rawType] || ('Потенциальное нарушение: ' + rawType);
      const timecodeSec = Number.isFinite(Number(item.timecode_sec)) ? Number(item.timecode_sec) : 0;
      const formattedTime = formatTimecode(timecodeSec);

      const li = document.createElement('li');
      li.className = 'violation-item';

      const header = document.createElement('div');
      header.className = 'violation-header';

      const titleGroup = document.createElement('div');
      titleGroup.className = 'violation-title-group';

      const titleEl = document.createElement('h3');
      titleEl.className = 'violation-type-title';
      titleEl.textContent = (eventIndex + 1) + '. ' + readableTitle;

      const codeEl = document.createElement('span');
      codeEl.className = 'violation-type-code';
      codeEl.textContent = 'Код типа: ' + rawType;

      titleGroup.appendChild(titleEl);
      titleGroup.appendChild(codeEl);

      const timeBtn = document.createElement('button');
      timeBtn.type = 'button';
      timeBtn.className = 'timecode-btn';
      timeBtn.textContent = '▶ Таймкод ' + formattedTime;
      timeBtn.setAttribute('aria-label', 'Перейти к таймкоду ' + formattedTime + ' для события ' + readableTitle);
      timeBtn.addEventListener('click', function () {
        seekVideoToSeconds(timecodeSec);
      });

      header.appendChild(titleGroup);
      header.appendChild(timeBtn);
      li.appendChild(header);

      const descEl = document.createElement('p');
      descEl.className = 'violation-description';
      renderUntrustedTextWithSafeLinks(descEl, item.description);
      li.appendChild(descEl);

      const rawFrames = Array.isArray(item.evidence_frames)
        ? item.evidence_frames.slice(0, MAX_EVIDENCE_FRAMES_PER_EVENT)
        : [];

      const validFrames = [];
      rawFrames.forEach(function (frameObj) {
        if (!frameObj || typeof frameObj !== 'object') {
          return;
        }
        const safeSrc = normalizeEvidenceImageSrc(frameObj.image_base64);
        if (!safeSrc) {
          return;
        }
        const frameSec = Number.isFinite(Number(frameObj.timestamp_sec))
          ? Number(frameObj.timestamp_sec)
          : timecodeSec;
        validFrames.push({
          timestampSec: frameSec,
          formattedTime: formatTimecode(frameSec),
          src: safeSrc
        });
      });

      const evidenceLabel = document.createElement('p');
      evidenceLabel.className = 'evidence-section-label';
      evidenceLabel.textContent = 'Стоп-кадры эпизода (' + validFrames.length + ' из ' + Math.min(rawFrames.length || 1, MAX_EVIDENCE_FRAMES_PER_EVENT) + ')';
      li.appendChild(evidenceLabel);

      if (validFrames.length > 0) {
        const grid = document.createElement('div');
        grid.className = 'evidence-grid';

        validFrames.forEach(function (vf, frameIndex) {
          const cardBtn = document.createElement('button');
          cardBtn.type = 'button';
          cardBtn.className = 'evidence-card';
          cardBtn.setAttribute(
            'aria-label',
            'Увеличить стоп-кадр ' + (frameIndex + 1) + ' (таймкод ' + vf.formattedTime + ')'
          );

          const thumbWrap = document.createElement('div');
          thumbWrap.className = 'evidence-thumb-wrap';

          const img = document.createElement('img');
          img.className = 'evidence-thumb';
          img.src = vf.src;
          img.alt = 'Стоп-кадр ' + (frameIndex + 1) + ' для события «' + readableTitle + '» на ' + vf.formattedTime;
          img.loading = 'lazy';
          thumbWrap.appendChild(img);

          const caption = document.createElement('div');
          caption.className = 'evidence-caption';

          const capLabel = document.createElement('span');
          capLabel.textContent = 'Кадр ' + (frameIndex + 1);

          const capTime = document.createElement('span');
          capTime.className = 'evidence-caption-time';
          capTime.textContent = vf.formattedTime;

          caption.appendChild(capLabel);
          caption.appendChild(capTime);

          cardBtn.appendChild(thumbWrap);
          cardBtn.appendChild(caption);

          cardBtn.addEventListener('click', function () {
            openEvidenceModal({
              eventTitle: readableTitle,
              eventType: rawType,
              eventDescription: item.description,
              frames: validFrames,
              activeIndex: frameIndex
            }, cardBtn);
          });

          grid.appendChild(cardBtn);
        });

        li.appendChild(grid);
      } else {
        const emptyNote = document.createElement('p');
        emptyNote.className = 'evidence-invalid-note';
        emptyNote.textContent = 'Корректные изображения стоп-кадров (JPEG/PNG/WebP base64) не предоставлены в ответе сервера.';
        li.appendChild(emptyNote);
      }

      dom.violationsList.appendChild(li);
    });
  }

  /* Accessible Evidence Modal Dialog */

  function openEvidenceModal(context, triggerEl) {
    if (!context || !Array.isArray(context.frames) || context.frames.length === 0) {
      return;
    }
    state.modalContext = context;
    state.lastFocusedElement = triggerEl || document.activeElement;
    updateEvidenceModalView();
    dom.evidenceModal.hidden = false;
    dom.modalCloseBtn.focus();
  }

  function updateEvidenceModalView() {
    const ctx = state.modalContext;
    if (!ctx) {
      return;
    }
    const frame = ctx.frames[ctx.activeIndex];
    if (!frame) {
      return;
    }

    dom.modalTitle.textContent = ctx.eventTitle;
    dom.modalSubtitle.textContent = 'Код события: ' + ctx.eventType + ' • Кадр ' + (ctx.activeIndex + 1) + ' из ' + ctx.frames.length;
    dom.modalImage.src = frame.src;
    dom.modalImage.alt = 'Увеличенный стоп-кадр на отметке ' + frame.formattedTime;
    dom.modalCaption.textContent = 'Таймкод кадра: ' + frame.formattedTime + ' (' + Math.floor(frame.timestampSec) + ' сек.)';
    renderUntrustedTextWithSafeLinks(dom.modalDescription, ctx.eventDescription);

    dom.modalFrameCounter.textContent = (ctx.activeIndex + 1) + ' / ' + ctx.frames.length;
    dom.modalPrevBtn.disabled = ctx.activeIndex <= 0;
    dom.modalNextBtn.disabled = ctx.activeIndex >= ctx.frames.length - 1;
  }

  function closeEvidenceModal() {
    if (dom.evidenceModal.hidden) {
      return;
    }
    dom.evidenceModal.hidden = true;
    dom.modalImage.src = '';
    state.modalContext = null;
    if (state.lastFocusedElement && typeof state.lastFocusedElement.focus === 'function') {
      state.lastFocusedElement.focus();
    }
    state.lastFocusedElement = null;
  }

  function handleModalKeydown(event) {
    if (dom.evidenceModal.hidden) {
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      closeEvidenceModal();
      return;
    }
    if (event.key === 'ArrowLeft' && state.modalContext && state.modalContext.activeIndex > 0) {
      event.preventDefault();
      state.modalContext.activeIndex -= 1;
      updateEvidenceModalView();
      return;
    }
    if (
      event.key === 'ArrowRight' &&
      state.modalContext &&
      state.modalContext.activeIndex < state.modalContext.frames.length - 1
    ) {
      event.preventDefault();
      state.modalContext.activeIndex += 1;
      updateEvidenceModalView();
      return;
    }
    if (event.key === 'Tab') {
      const focusables = Array.from(
        dom.evidenceModal.querySelectorAll('button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])')
      );
      if (focusables.length === 0) {
        return;
      }
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  }

  /* Optional Authenticated Server Video Stream Fallback */

  async function tryLoadAuthenticatedJobVideo(jobId, apiKey, expectedGeneration) {
    if (!jobId || !apiKey) {
      return;
    }
    try {
      const response = await apiFetch('/api/v1/jobs/' + encodeURIComponent(jobId) + '/video', {
        method: 'GET',
        headers: {
          'X-API-Key': apiKey
        },
        cache: 'no-store'
      });
      if (expectedGeneration !== state.pollGeneration || state.activeJobId !== jobId) {
        return;
      }
      if (!response.ok) {
        return;
      }
      const contentType = (response.headers.get('Content-Type') || '').toLowerCase();
      if (!contentType.startsWith('video/') && !contentType.includes('application/octet-stream')) {
        return;
      }
      const videoBlob = await response.blob();
      if (!videoBlob || videoBlob.size === 0 || expectedGeneration !== state.pollGeneration) {
        return;
      }
      revokeRemoteBlobUrl();
      state.remoteBlobUrl = URL.createObjectURL(videoBlob);
      state.videoPlayable = true;
      dom.videoCodecFallback.hidden = true;
      dom.videoSourceIndicator.textContent = 'Видео с сервера';
      dom.videoPlayer.src = state.remoteBlobUrl;
      dom.videoPreviewContainer.hidden = false;
      revokeLocalBlobUrl();
    } catch (_err) {
      // Retain local preview silently if optional endpoint is unavailable
    }
  }

  /* API Communication & Non-Overlapping Polling */

  function cancelActivePolling() {
    state.pollGeneration += 1;
    if (state.pollTimerId !== null) {
      clearTimeout(state.pollTimerId);
      state.pollTimerId = null;
    }
    if (state.pollAbortController) {
      try {
        state.pollAbortController.abort();
      } catch (_err) {
        // Ignore abort errors
      }
      state.pollAbortController = null;
    }
  }

  async function extractApiErrorMessage(response, fallbackMsg) {
    try {
      const data = await response.json();
      if (data && typeof data === 'object') {
        if (typeof data.error === 'string' && data.error.trim()) {
          return data.error.trim();
        }
        if (typeof data.detail === 'string' && data.detail.trim()) {
          return data.detail.trim();
        }
        if (typeof data.message === 'string' && data.message.trim()) {
          return data.message.trim();
        }
      }
    } catch (_err) {
      // Ignore JSON parse error
    }
    if (response.status === 401 || response.status === 403) {
      return 'Отказано в доступе (код ' + response.status + '). Проверьте правильность ключа X-API-Key.';
    }
    if (response.status === 413) {
      return 'Размер видеофайла превышает допустимый лимит сервера (код 413).';
    }
    return fallbackMsg + ' (HTTP ' + response.status + ')';
  }

  async function submitVideoForAnalysis() {
    if (hasActiveLockedJob()) {
      return;
    }

    const apiKey = getCurrentApiKey();
    if (!apiKey) {
      setUiState('invalid', {
        title: 'Не указан ключ доступа API',
        message: 'Введите действительный ключ X-API-Key перед отправкой видеозаписи на проверку.'
      });
      dom.apiKeyInput.focus();
      return;
    }

    const fileError = validateMovFile(state.selectedFile);
    if (fileError) {
      setUiState('invalid', {
        title: 'Ошибка проверки видеофайла',
        message: fileError
      });
      return;
    }

    setStoredApiKey(apiKey);
    cancelActivePolling();

    const currentGen = state.pollGeneration;
    state.activeJobId = null;
    state.consecutivePollErrors = 0;
    dom.violationsList.textContent = '';
    setUiState('uploading');

    const formData = new FormData();
    formData.append('file', state.selectedFile, state.selectedFile.name);

    try {
      const response = await apiFetch('/api/v1/analyze', {
        method: 'POST',
        headers: {
          'X-API-Key': apiKey
        },
        body: formData
      });

      if (currentGen !== state.pollGeneration) {
        return;
      }

      if (response.status === 401 || response.status === 403) {
        const authMsg = await extractApiErrorMessage(response, 'Недействительный ключ доступа API');
        setUiState('invalid', {
          title: 'Ошибка авторизации (X-API-Key)',
          message: authMsg
        });
        return;
      }

      if (!response.ok) {
        const errMsg = await extractApiErrorMessage(response, 'Не удалось отправить видеозапись на анализ');
        setUiState('failed', {
          title: 'Ошибка при отправке видеозаписи',
          message: errMsg,
          allowRetryPoll: false
        });
        return;
      }

      const payload = await response.json();
      if (!payload || typeof payload.job_id !== 'string' || !payload.job_id.trim()) {
        setUiState('failed', {
          title: 'Некорректный ответ сервера',
          message: 'Сервер не вернул идентификатор задачи (job_id).',
          allowRetryPoll: false
        });
        return;
      }

      state.activeJobId = payload.job_id.trim();
      state.pollStartedAt = Date.now();
      state.consecutivePollErrors = 0;

      setUiState('processing', {
        stage: 'queued',
        progress: null
      });

      scheduleNextPoll(currentGen, 300);
    } catch (_networkErr) {
      if (currentGen !== state.pollGeneration) {
        return;
      }
      setUiState('failed', {
        title: 'Сетевая ошибка при отправке файла',
        message: 'Не удалось установить соединение с сервером при загрузке видеофайла. Проверьте подключение и попробуйте снова.',
        allowRetryPoll: false
      });
    }
  }

  function scheduleNextPoll(expectedGeneration, delayMs) {
    if (expectedGeneration !== state.pollGeneration || !state.activeJobId) {
      return;
    }
    if (state.pollTimerId !== null) {
      clearTimeout(state.pollTimerId);
    }
    const waitMs = typeof delayMs === 'number' ? delayMs : state.pollIntervalMs;
    state.pollTimerId = window.setTimeout(function () {
      state.pollTimerId = null;
      executeSinglePoll(expectedGeneration);
    }, waitMs);
  }

  async function executeSinglePoll(expectedGeneration) {
    if (expectedGeneration !== state.pollGeneration || !state.activeJobId) {
      return;
    }

    if (Date.now() - state.pollStartedAt > MAX_POLL_DURATION_MS) {
      setUiState('timeout', {
        title: 'Превышено расчетное время ожидания',
        message: 'Задача ' + state.activeJobId + ' все еще не завершена. Вы можете вручную запросить актуальный статус без повторной отправки файла.'
      });
      return;
    }

    const apiKey = getCurrentApiKey() || getStoredApiKey();
    if (!apiKey) {
      setUiState('invalid', {
        title: 'Ключ доступа API отсутствует',
        message: 'Для продолжения опроса задачи укажите ключ X-API-Key и нажмите «Повторить запрос статуса».'
      });
      return;
    }

    const controller = new AbortController();
    state.pollAbortController = controller;

    try {
      const response = await apiFetch('/api/v1/jobs/' + encodeURIComponent(state.activeJobId), {
        method: 'GET',
        headers: {
          'X-API-Key': apiKey
        },
        cache: 'no-store',
        signal: controller.signal
      });

      if (expectedGeneration !== state.pollGeneration) {
        return;
      }
      state.pollAbortController = null;

      if (response.status === 401 || response.status === 403) {
        const authMsg = await extractApiErrorMessage(response, 'Отказано в доступе при опросе задачи');
        setUiState('network_retry', {
          title: 'Ошибка авторизации при опросе статуса',
          message: authMsg + ' Проверьте ключ X-API-Key и нажмите «Повторить запрос статуса».'
        });
        return;
      }

      if (response.status >= 500 || response.status === 429) {
        throw new Error('Transient HTTP ' + response.status);
      }

      if (!response.ok) {
        const fatalMsg = await extractApiErrorMessage(response, 'Ошибка при получении статуса задачи');
        setUiState('failed', {
          title: 'Не удалось получить статус задачи',
          message: fatalMsg,
          allowRetryPoll: true
        });
        return;
      }

      const jobData = await response.json();
      if (expectedGeneration !== state.pollGeneration) {
        return;
      }

      state.consecutivePollErrors = 0;
      handleJobStatusPayload(jobData, apiKey, expectedGeneration);
    } catch (err) {
      if (expectedGeneration !== state.pollGeneration) {
        return;
      }
      if (err && err.name === 'AbortError') {
        return;
      }

      state.consecutivePollErrors += 1;
      if (state.consecutivePollErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
        setUiState('network_retry', {
          title: 'Потеряна связь с сервером (' + state.consecutivePollErrors + ' попыток)',
          message: 'Автоматический опрос приостановлен из-за повторяющихся ошибок сети. Задача сохранена — нажмите «Повторить запрос статуса», когда соединение восстановится.'
        });
        return;
      }

      setUiState('network_retry', {
        title: 'Временный сбой сети (попытка ' + state.consecutivePollErrors + ' из ' + MAX_CONSECUTIVE_POLL_ERRORS + ')',
        message: 'Повторный запрос статуса задачи будет выполнен автоматически через 1,5 сек. или вручную по кнопке ниже.'
      });
      scheduleNextPoll(expectedGeneration, state.pollIntervalMs);
    }
  }

  function handleJobStatusPayload(jobData, apiKey, expectedGeneration) {
    if (!jobData || typeof jobData !== 'object') {
      setUiState('failed', {
        title: 'Некорректный формат ответа сервера',
        message: 'Получены поврежденные данные о статусе задачи.',
        allowRetryPoll: true
      });
      return;
    }

    const jobStatus = jobData.status;

    if (jobStatus === 'processing') {
      const oldTimeline = document.getElementById('full-operation-timeline');
      if (oldTimeline) oldTimeline.remove();
      const stage = typeof jobData.stage === 'string' ? jobData.stage : null;
      const progress = typeof jobData.progress === 'number' ? jobData.progress : null;
      setUiState('processing', { stage: stage, progress: progress });
      scheduleNextPoll(expectedGeneration, state.pollIntervalMs);
      return;
    }

    if (jobStatus === 'failed') {
      const serverError =
        typeof jobData.error === 'string' && jobData.error.trim()
          ? jobData.error.trim()
          : 'Сервер вернул статус failed без дополнительного описания ошибки.';
      setUiState('failed', {
        title: 'Обработка задачи завершилась ошибкой',
        message: serverError,
        allowRetryPoll: false
      });
      return;
    }

    if (jobStatus === 'done') {
      const verdict = jobData.verdict;
      const violations = Array.isArray(jobData.violations) ? jobData.violations : null;

      if ((verdict !== 'violation' && verdict !== 'no_violation') || !violations) {
        setUiState('failed', {
          title: 'Неполный ответ завершенной задачи',
          message: 'В ответе сервера отсутствуют обязательные поля verdict или violations.',
          allowRetryPoll: false
        });
        return;
      }

      renderViolationsList(violations);

      if (verdict === 'violation') {
        setUiState('done_violation', { eventsCount: violations.length });
      } else {
        setUiState('done_no_violation', { eventsCount: violations.length });
      }

      tryLoadAuthenticatedJobVideo(state.activeJobId, apiKey, expectedGeneration);
      loadFullTimeline(state.activeJobId, apiKey, expectedGeneration);
      return;
    }

    setUiState('failed', {
      title: 'Неизвестный статус задачи',
      message: 'Сервер вернул неподдерживаемый статус задачи: «' + String(jobStatus) + '».',
      allowRetryPoll: true
    });
  }

  async function loadFullTimeline(jobId, apiKey, generation) {
    try {
      const response = await apiFetch('/api/v1/jobs/' + encodeURIComponent(jobId) + '/timeline', {headers: {'X-API-Key': apiKey}});
      if (!response.ok) return;
      const data = await response.json();
      if (state.activeJobId !== jobId || state.pollGeneration !== generation) return;
      const old = document.getElementById('full-operation-timeline');
      if (old) old.remove();
      const panel = document.createElement('details'); panel.id = 'full-operation-timeline'; panel.className = 'card';
      const heading = document.createElement('summary'); heading.textContent = 'Полная хроника операции'; panel.appendChild(heading);
      const source = document.createElement('p');
      source.textContent = data.metrics.known_label_used ? 'Совпадение с эталоном ' + data.metrics.reference_id + ': итог использует известную разметку. Ниже — визуальные наблюдения модели.' : 'Новый ролик: итог основан на разборе всей операции.';
      if (data.metrics.reference_model_disagreement) source.textContent += ' Визуальная оценка модели расходится с эталоном: проверьте кадры и хронику.';
      panel.appendChild(source);
      for (const interval of data.intervals) {
        const row = document.createElement('p'); row.textContent = interval.start.toFixed(1) + '–' + interval.end.toFixed(1) + ' с: ' + interval.observations; panel.appendChild(row);
      }
      const parent = document.querySelector('.workspace-col--results'); if (parent) parent.appendChild(panel);
    } catch (_) { /* Chronology is optional; the validated result remains visible. */ }
  }

  function manualRetryStatusPoll() {
    if (!state.activeJobId) {
      return;
    }
    cancelActivePolling();
    const gen = state.pollGeneration;
    state.consecutivePollErrors = 0;
    state.pollStartedAt = Date.now();
    setUiState('processing', {
      stage: 'Повторный запрос состояния задачи...',
      progress: null
    });
    executeSinglePoll(gen);
  }

  function resetActiveJobAndStartFresh() {
    cancelActivePolling();
    state.activeJobId = null;
    state.consecutivePollErrors = 0;
    dom.violationsList.textContent = '';
    dom.violationsListCounter.textContent = '0';

    if (state.remoteBlobUrl && state.selectedFile) {
      revokeRemoteBlobUrl();
      state.localBlobUrl = URL.createObjectURL(state.selectedFile);
      dom.videoPlayer.src = state.localBlobUrl;
      dom.videoSourceIndicator.textContent = 'Локальный файл';
    }

    setUiState('idle');
  }

  /* Optional Public Probes (/healthz & /api/v1/config) */

  async function probeHealth() {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(function () { controller.abort(); }, 5000);
    try {
      const healthResp = await apiFetch('/healthz', { method: 'GET', cache: 'no-store', signal: controller.signal });
      const health = await healthResp.json();
      if (health.ready === true) {
        dom.serviceHealthBadge.className = 'service-pill service-pill--ok';
        dom.serviceHealthText.textContent = 'Сервис готов к проверке';
      } else {
        dom.serviceHealthBadge.className = 'service-pill service-pill--neutral';
        dom.serviceHealthText.textContent = health.worker_alive ? 'Подготовка модели' : 'Сервис временно недоступен';
      }
    } catch (_err) {
      dom.serviceHealthBadge.className = 'service-pill service-pill--neutral';
      dom.serviceHealthText.textContent = 'Нет соединения с сервисом';
    } finally {
      window.clearTimeout(timeoutId);
      state.healthTimerId = window.setTimeout(probeHealth, 10000);
    }
  }

  async function probePublicEndpoints() {
    probeHealth();
    try {
      const configResp = await apiFetch('/api/v1/config', { method: 'GET', cache: 'no-store' });
      if (!configResp.ok) {
        return;
      }
      const cfg = await configResp.json();
      if (!cfg || typeof cfg !== 'object') {
        return;
      }
      if (typeof cfg.max_upload_bytes === 'number' && cfg.max_upload_bytes > 0 && cfg.max_upload_bytes <= DEFAULT_MAX_FILE_BYTES) {
        state.maxFileBytes = cfg.max_upload_bytes;
        const mb = Math.round(state.maxFileBytes / (1024 * 1024));
        dom.fileLimitBadge.textContent = 'Формат .MOV • до ' + mb + ' МБ';
      }
      if (typeof cfg.poll_interval_ms === 'number' && cfg.poll_interval_ms >= 500 && cfg.poll_interval_ms <= 10000) {
        state.pollIntervalMs = cfg.poll_interval_ms;
      }
      if (typeof cfg.version === 'string' && cfg.version.trim()) {
        dom.configMetaFooter.textContent = 'Версия сервиса: ' + cfg.version.trim();
      }
    } catch (_err) {
      // Optional endpoint; ignore errors
    }
  }

  /* Event Listeners Setup */

  function initEventListeners() {
    // Restore session API key
    const savedKey = getStoredApiKey();
    if (savedKey) {
      dom.apiKeyInput.value = savedKey;
      updateKeyBadge(true);
    } else {
      updateKeyBadge(false);
    }

    dom.apiKeyInput.addEventListener('input', function () {
      setStoredApiKey(dom.apiKeyInput.value);
      if (state.uiState === 'invalid' && getCurrentApiKey()) {
        setUiState('idle');
      } else {
        updateControlsInteractivity();
      }
    });

    dom.toggleKeyVisibilityBtn.addEventListener('click', function () {
      const isPassword = dom.apiKeyInput.type === 'password';
      dom.apiKeyInput.type = isPassword ? 'text' : 'password';
      dom.toggleKeyVisibilityBtn.textContent = isPassword ? 'Скрыть' : 'Показать';
      dom.toggleKeyVisibilityBtn.setAttribute('aria-pressed', isPassword ? 'true' : 'false');
    });

    dom.clearKeyBtn.addEventListener('click', function () {
      if (isJobBusy()) {
        return;
      }
      dom.apiKeyInput.value = '';
      dom.apiKeyInput.type = 'password';
      dom.toggleKeyVisibilityBtn.textContent = 'Показать';
      dom.toggleKeyVisibilityBtn.setAttribute('aria-pressed', 'false');
      setStoredApiKey('');
      updateControlsInteractivity();
      dom.apiKeyInput.focus();
    });

    document.getElementById('auth-form').addEventListener('submit', function (e) {
      e.preventDefault();
    });

    // Drop zone & File picker
    dom.dropZone.addEventListener('click', function () {
      if (!hasActiveLockedJob()) {
        dom.videoFileInput.click();
      }
    });

    dom.dropZone.addEventListener('keydown', function (e) {
      if (hasActiveLockedJob()) {
        return;
      }
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        dom.videoFileInput.click();
      }
    });

    dom.dropZone.addEventListener('dragover', function (e) {
      e.preventDefault();
      if (!hasActiveLockedJob()) {
        dom.dropZone.classList.add('is-dragover');
      }
    });

    dom.dropZone.addEventListener('dragleave', function () {
      dom.dropZone.classList.remove('is-dragover');
    });

    dom.dropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      dom.dropZone.classList.remove('is-dragover');
      if (hasActiveLockedJob()) {
        return;
      }
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileSelection(files[0]);
      }
    });

    dom.videoFileInput.addEventListener('change', function () {
      if (dom.videoFileInput.files && dom.videoFileInput.files.length > 0) {
        handleFileSelection(dom.videoFileInput.files[0]);
      }
    });

    dom.removeFileBtn.addEventListener('click', function () {
      if (!hasActiveLockedJob()) {
        clearSelectedFile(true);
      }
    });

    // Video element codec compatibility listeners
    dom.videoPlayer.addEventListener('error', function () {
      if (state.selectedFile || state.remoteBlobUrl) {
        state.videoPlayable = false;
        dom.videoCodecFallback.hidden = false;
      }
    });

    dom.videoPlayer.addEventListener('loadedmetadata', function () {
      state.videoPlayable = true;
      dom.videoCodecFallback.hidden = true;
    });

    // Job Submission & Recovery Actions
    dom.submitAuditBtn.addEventListener('click', function () {
      submitVideoForAnalysis();
    });

    dom.retryPollBtn.addEventListener('click', function () {
      manualRetryStatusPoll();
    });

    dom.resetJobBtn.addEventListener('click', function () {
      resetActiveJobAndStartFresh();
    });

    dom.startNewAuditBtn.addEventListener('click', function () {
      resetActiveJobAndStartFresh();
    });

    // Modal Dialog Controls
    dom.modalCloseBtn.addEventListener('click', function () {
      closeEvidenceModal();
    });

    dom.evidenceModal.addEventListener('click', function (e) {
      if (e.target === dom.evidenceModal) {
        closeEvidenceModal();
      }
    });

    dom.modalPrevBtn.addEventListener('click', function () {
      if (state.modalContext && state.modalContext.activeIndex > 0) {
        state.modalContext.activeIndex -= 1;
        updateEvidenceModalView();
      }
    });

    dom.modalNextBtn.addEventListener('click', function () {
      if (state.modalContext && state.modalContext.activeIndex < state.modalContext.frames.length - 1) {
        state.modalContext.activeIndex += 1;
        updateEvidenceModalView();
      }
    });

    dom.modalSeekBtn.addEventListener('click', function () {
      if (!state.modalContext) {
        return;
      }
      const frame = state.modalContext.frames[state.modalContext.activeIndex];
      if (frame) {
        closeEvidenceModal();
        seekVideoToSeconds(frame.timestampSec);
      }
    });

    document.addEventListener('keydown', handleModalKeydown);

    window.addEventListener('beforeunload', function () {
      window.clearTimeout(state.healthTimerId);
      revokeAllBlobUrls();
    });
  }

  initEventListeners();
  updateControlsInteractivity();
  probePublicEndpoints();
  // Local demo deep links restore a completed API job without another upload.
  // Credentials stay in the URL fragment and are immediately removed.
  if (['127.0.0.1', 'localhost', '::1'].includes(window.location.hostname)) {
    const localParams = new URLSearchParams(window.location.hash.slice(1));
    const localKey = localParams.get('key');
    const localJob = localParams.get('job');
    if (localKey) {
      dom.apiKeyInput.value = localKey;
      setStoredApiKey(localKey);
      updateKeyBadge(true);
    }
    if (localKey || localJob) window.history.replaceState(null, '', window.location.pathname);
    if (localJob && /^[A-Za-z0-9_-]{1,128}$/.test(localJob)) {
      state.activeJobId = localJob;
      apiFetch('/api/v1/jobs/' + encodeURIComponent(localJob), {headers: {'X-API-Key': getCurrentApiKey()}})
        .then(function (response) { if (!response.ok) throw new Error('Не удалось открыть задачу'); return response.json(); })
        .then(function (data) { handleJobStatusPayload(data, getCurrentApiKey(), state.pollGeneration); })
        .catch(function () { dom.jobIdValue.textContent = 'Не удалось открыть сохранённую задачу'; });
    }
  }
})();
