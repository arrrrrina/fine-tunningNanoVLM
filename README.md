# fine-tunningNanoVLM

Код для подготовки датасетов и запуска экспериментов по fine-tuning NanoVLM в среде `MiniGrid-Empty-Random-6x6-v0`.

Агент получает RGB-наблюдение среды и должен выбрать следующее действие:

* `0` — `left`
* `1` — `right`
* `2` — `forward`

## Структура репозитория

```text
fine-tunningNanoVLM/
├── README.md
├── dataset_creation/
│   ├── create_dataset_with_actions.py
│   ├── convert_dataset_with_actions.py
│   ├── create_dataset_with_text.py
│   └── convert_dataset_with_text.py
└── notebooks_with_processing/
    ├── GRPO.ipynb
    └── nanoVLM.ipynb
```

## Установка зависимостей

```bash
pip install gymnasium minigrid datasets pillow matplotlib transformers accelerate safetensors
```

Для запуска ноутбуков с обучением также нужен код NanoVLM:

```bash
git clone https://github.com/huggingface/nanoVLM.git
```

## Файлы

### `dataset_creation/create_dataset_with_actions.py`

Создаёт датасет экспертных траекторий для формата **action-only**.

Скрипт запускает `MiniGrid-Empty-Random-6x6-v0`, использует эвристического эксперта и сохраняет пары:

```text
image -> action
```

Результат:

```text
dataset_high_resolution/
├── images/
└── dataset.json
```

Запуск:

```bash
python dataset_creation/create_dataset_with_actions.py
```

---

### `dataset_creation/convert_dataset_with_actions.py`

Конвертирует `dataset_high_resolution/dataset.json` в формат HuggingFace `Dataset`.

Действия переводятся в числовой формат:

```text
left    -> 0
right   -> 1
forward -> 2
```

Результат:

```text
my_expert_dataset_action_ids
├── dataset_info.json
└── dataset.arrow
```

Запуск:

```bash
python dataset_creation/convert_dataset_with_actions.py
```

---

### `dataset_creation/create_dataset_with_text.py`

Создаёт датасет экспертных траекторий для формата **text + action**.

В этом формате ответ модели содержит описание состояния, короткий план и действие:

```text
State: ...
Plan: ...
Action: 0/1/2
```

Результат:

```text
dataset_text_action/
├── images/
└── dataset.json
```

Запуск:

```bash
python dataset_creation/create_dataset_with_text.py
```

---

### `dataset_creation/convert_dataset_with_text.py`

Конвертирует `dataset_text_action/dataset.json` в формат HuggingFace `Dataset`.

Результат:

```text
my_expert_dataset_text_action
├── dataset_info.json
└── dataset.arrow
```

Запуск:

```bash
python dataset_creation/convert_dataset_with_text.py
```

---

### `notebooks_with_processing/nanoVLM.ipynb`

Ноутбук для запуска **SFT-бэйзлайна** NanoVLM.

В нём выполняются основные шаги supervised fine-tuning:

* загрузка NanoVLM;
* загрузка подготовленного action-only датасета;
* настройка processor/tokenizer;
* обучение модели предсказывать следующее действие по изображению;
* сохранение чекпоинтов;
* оценка модели в `MiniGrid-Empty-Random-6x6-v0`;
* построение графиков success rate по чекпоинтам.

Перед запуском нужно создать action-only датасет:

```bash
python dataset_creation/create_dataset_with_actions.py
python dataset_creation/convert_dataset_with_actions.py
```

---

### `notebooks_with_processing/GRPO.ipynb`

Ноутбук для запуска **GRPO-обучения**.

В нём реализован RL-цикл, где модель взаимодействует со средой `MiniGrid-Empty-Random-6x6-v0` и дообучается по reward из среды.

Основные шаги:

* загрузка SFT-чекпоинта;
* генерация действия через `model.generate()`;
* rollout эпизодов в MiniGrid;
* расчёт return и advantage;
* GRPO-обновление модели;
* evaluation после updates;
* построение графиков success rate / average return.

Перед запуском нужен SFT-чекпоинт, полученный в `notebooks_with_processing/nanoVLM.ipynb`.

## Полный пайплайн запуска

### 1. Создание action-only датасета

```bash
python dataset_creation/create_dataset_with_actions.py
python dataset_creation/convert_dataset_with_actions.py
```

### 2. Запуск SFT-бэйзлайна

Открыть ноутбук:

```text
notebooks_with_processing/nanoVLM.ipynb
```

и выполнить ячейки по порядку.

### 3. Запуск GRPO

Открыть ноутбук:

```text
notebooks_with_processing/GRPO.ipynb
```

и выполнить ячейки по порядку, указав путь к SFT-чекпоинту.

### 4. Создание text + action датасета

```bash
python dataset_creation/create_dataset_with_text.py
python dataset_creation/convert_dataset_with_text.py
```

## Форматы датасетов

### Action-only

Prompt:

```text
Look at the image. Action IDs: 0 = left, 1 = right, 2 = forward. Choose the best action to reach the goal. Answer with only one number.
```

Ответ модели:

```text
0
```

или

```text
1
```

или

```text
2
```

### Text + action

Prompt просит модель описать состояние, предложить план и выбрать действие.

Ответ модели:

```text
State: ...
Plan: ...
Action: 0/1/2
```

## Expert policy

Для сбора данных используется эвристический эксперт. Он находит позицию агента и цели, затем выбирает действие, которое приближает агента к цели сначала по оси `x`, затем по оси `y`.

Так как `EmptyEnv` не содержит препятствий, такой эксперт даёт короткие и корректные траектории.
