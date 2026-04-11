# Запуск Sandbox-агента на бесплатной модели Nemotron через OpenRouter

## Зачем

OpenRouter предоставляет бесплатный доступ к модели NVIDIA Nemotron. Sandbox-агент использует OpenAI SDK, который совместим с OpenRouter — достаточно поменять URL и ключ.

## Предварительные требования

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов
- Аккаунт на [openrouter.ai](https://openrouter.ai)

## 1. Получить API-ключ OpenRouter

1. Зарегистрируйтесь на [openrouter.ai](https://openrouter.ai)
2. Перейдите в [Keys](https://openrouter.ai/keys)
3. Создайте новый ключ — он начинается с `sk-or-v1-...`

## 2. Установить зависимости

```bash
cd sandbox-py
uv sync
```

## 3. Настроить переменные окружения

OpenAI SDK читает `OPENAI_BASE_URL` и `OPENAI_API_KEY` автоматически — менять код не нужно.

```bash
export OPENAI_API_KEY="sk-or-v1-ваш-ключ-openrouter"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

## 4. Поменять модель в `main.py`

Замените строку с `MODEL_ID`:

```python
# было:
MODEL_ID = "gpt-4.1-2025-04-14"

# стало (бесплатная Nemotron):
MODEL_ID = "nvidia/llama-3.1-nemotron-70b-instruct:free"
```

Актуальный список бесплатных моделей: [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0) — ищите по "nemotron".

## 5. Запустить

```bash
# все задачи
uv run python main.py

# одна задача
uv run python main.py t01
```

## Быстрый запуск одной командой

```bash
OPENAI_API_KEY="sk-or-v1-..." \
OPENAI_BASE_URL="https://openrouter.ai/api/v1" \
uv run python main.py t01
```

(не забудьте предварительно поменять `MODEL_ID` в `main.py`)

## Возможные проблемы

### Structured outputs не поддерживаются

Агент использует `client.beta.chat.completions.parse()` с Pydantic-схемой (`response_format=NextStep`). Это structured outputs — не все модели на OpenRouter их поддерживают.

**Симптом**: ошибка вида `400 Bad Request`, `response_format is not supported`, или парсинг возвращает `None`.

**Решение**: заменить вызов в `agent.py` на обычный JSON-mode:

```python
# было (строка ~139):
resp = client.beta.chat.completions.parse(
    model=model,
    response_format=NextStep,
    messages=log,
    max_completion_tokens=16384,
)
job = resp.choices[0].message.parsed

# стало:
resp = client.chat.completions.create(
    model=model,
    response_format={"type": "json_object"},
    messages=log,
    max_completion_tokens=16384,
)
raw = json.loads(resp.choices[0].message.content)
job = NextStep.model_validate(raw)
```

При этом добавьте в `system_prompt` в `agent.py` инструкцию отвечать строго в JSON:

```
Always respond with valid JSON matching this schema:
{
  "current_state": "string",
  "plan_remaining_steps_brief": ["string"],
  "task_completed": bool,
  "function": { "tool": "tool_name", ...tool_params }
}
```

### Rate limits на бесплатных моделях

Бесплатные модели на OpenRouter имеют лимиты (обычно ~20 запросов/мин). Если получаете `429 Too Many Requests` — подождите минуту или добавьте задержку между шагами.

### Модель отвечает не по схеме

Nemotron 70B слабее GPT-4.1, поэтому может генерировать невалидный JSON или игнорировать часть схемы. Ожидайте более низкие баллы на бенчмарке.

## Альтернативные бесплатные модели на OpenRouter

| Модель | MODEL_ID |
|--------|----------|
| Nemotron 70B | `nvidia/llama-3.1-nemotron-70b-instruct:free` |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct:free` |
| Qwen3 235B (MoE) | `qwen/qwen3-235b-a22b:free` |
| DeepSeek V3 | `deepseek/deepseek-chat-v3-0324:free` |

Проверяйте актуальность на [openrouter.ai/models](https://openrouter.ai/models?max_price=0).
