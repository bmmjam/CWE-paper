# Пайплайн экспериментов: When Does Test-Time Search Help LLMs Classify Vulnerabilities?

Документ фиксирует протокол экспериментов для **диагностической** статьи о том,
**при каких условиях** структурированный поиск во время инференса (MCTS и его
более дешёвые альтернативы) улучшает классификацию уязвимостей кода по таксономии
CWE — и при каких **не** улучшает.

Это сознательно **не** advocacy-статья («наш MCTS лучше»). Центральный вклад —
**контролируемое сравнение семейств test-time-стратегий при равном compute-бюджете**
с изоляцией двух осей, которые в литературе почти всегда смешаны: сила policy и
качество reward-сигнала. Статья публикуема при любом исходе, включая нулевой или
отрицательный прирост от поиска — именно это и является результатом.

**Целевая конференция**: AAAI main track (альтернативы — ICLR, EMNLP-Findings).
**Бюджетный режим**: pilot на $100 OpenRouter (cheap-tier модели, см. §3.3 и §5.3).
**Зафиксированная политика**: §12 (lock file) — не менять без явного решения.

> **Отличие от прежних внутренних экспериментов.** Все данные, сплиты и метрики
> здесь — **публичные и воспроизводимые**. Внутренний контур, корпоративные модели
> и закрытые датасеты в этой статье не используются и не цитируются. Прежние
> внутренние находки служат только источником гипотез (H2/H3), но не доказательной
> базой.

---

## 1. Постановка задачи и гипотезы

### 1.1 Исследовательские вопросы

**RQ1** *(основной — compute-matched)*: Превосходит ли MCTS более дешёвые
test-time-стратегии (greedy, self-consistency, best-of-N, beam) на задаче
иерархической CWE-классификации, **при равном суммарном бюджете токенов**, а не при
равном числе итераций? Где проходит граница, за которой поиск перестаёт окупаться?

**RQ2** *(policy)*: Как выигрыш от поиска зависит от силы базовой policy (per-level
accuracy модели)? Существует ли порог силы модели, ниже которого поиск **усиливает**
каскадную ошибку вместо того, чтобы её исправлять?

**RQ3** *(reward)*: Что является «бутылочным горлышком» — policy или reward-сигнал?
Сравниваем reward от self-evaluation policy-модели против reward от внешнего
LLM-judge. (PRM в этой статье **не** обучаем, см. §3.5.)

**RQ4** *(search space)*: Где поиск полезнее — над **таксономией CWE** (узлы дерева =
CWE-классы) или над **трасой рассуждения** (ReAct-style шаги: gather evidence →
hypothesize → verify, лист отображается в CWE)? Различается ли ответ для слабых и
сильных policy?

### 1.2 Гипотезы (формально)

Пусть `B` — суммарный бюджет токенов (input+output) на один пример; `Acc_X(B)` —
leaf-accuracy метода `X` при бюджете `B`; `α(π)` — per-level (точнее, L0/pillar)
accuracy жадного прохода policy `π`.

- **H1** *(compute-matched parity)*: При равном `B` разрыв
  `Acc_MCTS(B) − Acc_SelfConsistency(B) ≤ ε` (ε мал, проверяем `ε < 0.02` с 95% CI)
  для слабых policy. Иначе говоря, **при честном контроле бюджета** MCTS не
  доминирует над простым сэмплированием. Проверяется по всей кривой `B`.
- **H2** *(policy threshold)*: Маржинальный выигрыш поиска
  `Δ(π) := Acc_search(π) − Acc_greedy(π)` монотонно растёт по `α(π)`; существует
  порог `α*` такой, что при `α(π) < α*` имеем `Δ(π) ≤ 0` (поиск амплифицирует
  каскадную ошибку). Оцениваем `α*` по 3–4 моделям разной силы.
- **H3** *(reward bottleneck)*: Замена self-eval reward на внешний judge поднимает
  leaf-accuracy менее чем на `δ` при том, что pillar-accuracy уже высока →
  бутылочное горлышко в policy, а не в reward. Формально: `Δ_judge ≪ Δ_policy`,
  где `Δ_policy` — прирост при переходе к более сильной модели.
- **H4** *(search space)*: При слабой policy reasoning-trajectory search даёт
  больший `Δ`, чем taxonomy search, потому что taxonomy search компаундирует
  раннюю ошибку на верхних уровнях дерева. При сильной policy разрыв сокращается.

Каждая гипотеза проверяется bootstrap-доверительным интервалом 95% (10K resamples)
и paired McNemar test с поправкой Bonferroni.

### 1.3 Чем это отличается от наивных MCTS-статей

Большинство работ «MCTS улучшает LLM-reasoning» сравнивают MCTS с `N` итерациями
против greedy с 1 вызовом — то есть дают MCTS в десятки раз больше compute. Наш
протокол закрывает эту дыру: **все методы выравниваются по бюджету токенов** (§5.2),
и headline-рисунок — это `accuracy(B)` для всех методов на одной оси. Это и есть
методологический вклад, независимо от знака результата.

---

## 2. Данные

### 2.1 Источники (только публичные, с CWE-метками)

| Датасет | Содержание | Размер | CWE-разнообразие | Использование |
|---|---|---|---|---|
| PrimeVul | C/C++, ручная верификация, CWE-метки | ~7K функций | ~140 CWE | основной test + train |
| DiverseVul | C/C++, multi-source, CWE-метки | ~330K | ~150 CWE | train + расширение хвоста |
| BigVul | C/C++, mining-based, CWE-метки | ~190K | ~91 CWE | train + holdout |
| CVEFixes | multi-language, CVE→CWE→commit | ~12K CVE | ~180 CWE | cross-language robustness |
| CWE corpus (MITRE) | text descriptions, relationships, examples | ~900 узлов | full | дерево + judge-контекст |

Задача — **hierarchical multi-class CWE-classification**: по фрагменту кода
предсказать путь `(pillar → … → leaf)` в дереве CWE. Это сохраняет преемственность
с прежней линией работ (иерархия естественна для MCTS), но на полностью публичных
данных.

### 2.2 Таксономия (фиксированное дерево)

- Дерево строится из MITRE CWE relationships (`ChildOf`/`ParentOf`), приводится к
  **направленному дереву** (cross-edges разрешаются выбором primary-parent).
- Фиксируем подмножество: pillars + 2–3 уровня глубины, **дедупликация путей**
  (дубли путей в прошлом разрушали accuracy — критично).
- Публикуем `data/cwe_tree.json` со SHA-256 — это часть артефакта воспроизводимости.
- Granularity-ablation (§6, A6): pillars-only vs full-depth — отдельная ось.

### 2.3 Сплиты

Основной режим — **стандартная стратифицированная** train/calibration/test-разбивка
(в отличие от temporal-фокуса прежних работ; temporal здесь — лишь secondary
robustness-проверка, §2.5).

1. `Train` — для (а) few-shot-пулов и (б) калибровки reward-порогов. Энкодерные
   baseline'ы (если включаем) обучаются только здесь.
2. `Calibration` — стратифицированные 10% от Train для conformal/threshold-калибровки.
3. `Test` — стратифицированный по CWE, сбалансированный по pillar'ам, целевой размер
   ~2000 примеров (dev-сет 50–100 для итерации промптов).

### 2.4 Очистка и контроль leakage

- Дедупликация по точному и near-duplicate (MinHash, Jaccard ≥ 0.85) хешу функции
  между **всеми** источниками и сплитами — иначе train/test leakage.
- Split по `repo_id`, не только по функции (репозиторий не пересекает границу).
- Удаление функций < 5 строк и > 1000 строк; нормализация CWE-алиасов и устаревших
  ID через MITRE relationships.
- **Pretraining-leakage** (критично для LLM): отдельный анализ корреляции accuracy с
  датой публикации CVE относительно cutoff каждой модели. Репортим subset, где CVE
  опубликован **после** cutoff модели, как «честный» срез.
- Сохраняем мета: `language`, `commit_date`, `cve_id`, `cve_pub_date`, `repo`,
  `cwe_path`.

### 2.5 Temporal robustness (secondary)

Не главный угол, но добавляем контрольный temporal-сплит (cutoff по `cve_pub_date`),
чтобы показать, что выводы про test-time search не артефакт случайного сплита.

### 2.6 Статистика, которую нужно отрепортить

Размеры сплитов; распределение по CWE (head/mid/tail); распределение по языкам и
году; средняя длина функции (LOC); число CWE с N≥10 примерами в test (минимум для
per-class статистики); доля примеров, прошедших pretraining-cutoff-фильтр.

---

## 3. Методы (ось сравнения)

Все методы используют **одну и ту же** policy-модель `π` и **один и тот же**
фиксированный prompt-формат шага, чтобы различие было только в стратегии поиска.

### 3.1 Семейство test-time-стратегий

| ID | Метод | Knob (выравнивается под бюджет `B`) | Что изолирует |
|---|---|---|---|
| M0a | **Flat** (один вызов, все листья сразу) | — | no-search, no-hierarchy |
| M0b | **Greedy-hierarchical** (argmax на каждом уровне) | — | no-search baseline |
| M1 | **Self-Consistency** (sample `N` путей, majority vote) | `N` | sampling без reward |
| M2 | **Best-of-N** (sample `N` путей, выбор по reward) | `N` | sampling + reward |
| M3 | **Beam search** над таксономией (ширина `w` на уровень) | `w` | детерминир. structured search |
| M4 | **MCTS-taxonomy** (UCT над деревом CWE) | `#iterations` | MCTS над таксономией (RQ4-a) |
| M5 | **MCTS-reasoning** (UCT над ReAct-трасой) | `#iterations` | MCTS над рассуждением (RQ4-b) |

M0–M3 — это «лестница» сложности поиска, относительно которой измеряется
маржинальный вклад MCTS. M4 vs M5 — ядро RQ4.

### 3.2 MCTS-taxonomy (M4)

```
Вход: код c, дерево CWE T, policy π, reward r(·), бюджет I итераций
Состояние s = частичный путь (v_0..v_l)
Действия A(s) = children(v_l, T)
1. Selection: спуск по UCT(s,a) = Q(s,a) + c_exp·sqrt(ln N(s)/N(s,a))
2. Expansion: π предлагает k кандидатов-детей с rationale
3. Simulation/Eval: r(s') = reward частичного/полного пути (self-eval или judge)
4. Backprop: обновить Q, N вверх по дереву
Возврат: путь с макс. посещаемостью корневых детей (robust child)
```

### 3.3 MCTS-reasoning (M5)

То же UCT-ядро, но узлы дерева поиска = **шаги анализа кода**, а не CWE-узлы:
`gather_evidence` (что за код, какие операции с памятью/индексами/lifetime) →
`hypothesize` (кандидат-слабость) → `verify` (проверка контр-примером/трассой) →
`commit` (отображение в CWE-лист). Лист трасы детерминированно мапится в путь CWE.
Это «detection in code»-стиль рассуждения, обёрнутый в ту же CWE-классификацию.

### 3.4 Reward-сигнал (ось RQ3)

| Reward | Источник | Стоимость |
|---|---|---|
| `self-eval` | та же policy `π` оценивает свой (частичный) путь, 0–1 | дёшево, тот же tier |
| `external-judge` | отдельная (возможно более сильная) модель оценивает путь + код + CWE-описание | дороже |

**PRM не обучаем** в этой статье (нет GPU-фазы; см. §3.5). Сравнение self-eval vs
judge напрямую отвечает на RQ3: если judge не помогает при высокой pillar-accuracy
— горлышко в policy.

### 3.5 Что сознательно вне scope

- **Обучение PRM** (требует GPU, ~$300) — вынесено в Future Work. Reward — только
  zero-shot self-eval и judge.
- **Fine-tuning policy** — не делаем; все модели zero/few-shot, чтобы изолировать
  именно test-time-эффект.
- **Temporal generalization как главный угол** — это другая статья.

### 3.6 Базелайны вне семейства поиска (для контекста)

Минимальный набор, чтобы привязать числа к литературе (не главный фокус):

| Категория | Метод | Зачем |
|---|---|---|
| Retrieval-free LLM | flat zero-shot (M0a) | нижняя граница |
| Sampling SOTA | self-consistency (M1) | стандарт test-time-scaling |
| Encoder FT (опц.) | UniXcoder + linear head | привязка к «классике», если бюджет/время позволят |

### 3.7 Модели (policy) — ось RQ2 + tier-стратегия

Сила policy `α(π)` измеряется как pillar/L0-accuracy жадного прохода (M0b). Берём
модели, заведомо покрывающие диапазон силы, минимум 3 для оценки порога `α*`.

#### Cheap-tier (всё помещается в $100 OpenRouter pilot)

| Модель | $/1M in | $/1M out | Ожидаемая сила | Роль |
|---|---|---|---|---|
| GPT-4.1-nano | 0.10 | 0.40 | слабая | нижняя точка оси `α` |
| Gemini 2.5 Flash-Lite | 0.10 | 0.40 | слабая–средняя | альтернативная cheap |
| DeepSeek V3.1 | 0.15 | 0.75 | средняя | open-weight через API |
| Claude Haiku 4.5 | 1.00 | 5.00 | средняя–сильная | верхняя точка для оси `α` в pilot |

#### Mid/Premium (после pilot, при подтверждённом сигнале)

GPT-4.1-mini, Gemini 2.5 Flash, DeepSeek-R1 (mid); Claude Sonnet 4.6, GPT-4o/5,
Gemini 2.5 Pro (premium — нужны, чтобы получить точку с высокой `α(π)` и проверить
H2/H3 на сильном конце). Минимум 2 open-weight модели (DeepSeek + Qwen2.5-Coder-32B
или Llama-3.3-70B) для воспроизводимости AAAI.

#### Политика подключения моделей

1. **Pilot**: cheap-tier — валидировать сигнал на оси поиска и слабом конце `α`.
2. **Ablation**: cheap-tier × 5 seeds + 1–2 mid-tier.
3. **Headline**: добавить premium-точку, чтобы закрыть сильный конец оси `α(π)`.
   Без неё H2 (порог) недоказуема.
4. **Reproducibility**: open-weight (Qwen + Llama) для финальных таблиц.

Закрытое правило: новые closed-weight модели не вводим в середине фазы — snapshot
моделей фиксируется на старте каждой фазы.

### 3.8 Judge-модель

Для `external-judge` reward берём модель **сильнее** текущей policy (например, judge
= Haiku/Sonnet при policy = nano). В отчёте фиксируем пару (policy, judge), чтобы
сравнение self-eval vs judge было честным. Judge-промпт — contrastive (показывает
альтернативные CWE), как и policy-промпт.

---

## 4. Метрики

### 4.1 Качество классификации

- **Leaf exact-match accuracy** (главная).
- **Path Accuracy@k** (правильный путь в top-k).
- **Pillar / L0 accuracy** — критична для каскадного анализа (RQ2).
- **Hierarchical-F1** (усреднение по уровням, Kosmopoulos et al. 2015).
- **Macro-F1 / Weighted-F1** на листе (учёт дисбаланса классов).
- **Wu–Palmer similarity** между предсказанным и истинным путём.

### 4.2 Диагностика поиска (ядро статьи)

- **Gain-vs-budget кривая** `Acc_X(B)` для каждого метода `X` — headline-рисунок.
- **Marginal search gain** `Δ(π) = Acc_search − Acc_greedy` как функция `α(π)`.
- **Path-prefix error histogram**: на каком уровне впервые ошибается метод — прямой
  тест каскадной амплификации (RQ2).
- **Search efficiency**: accuracy на единицу токенов; «iterations-to-plateau».
- **Reward fidelity**: корреляция reward-сигнала с корректностью пути (AUROC reward
  как предиктора правильности) — отдельно для self-eval и judge (RQ3).

### 4.3 Калибровка

- **ECE** (15 bins), **Brier score** на confidence/reward.
- **Coverage / Set Size** при conformal prediction (если включаем).

### 4.4 Стоимость

- **Tokens/sample** (input+output) — основная ось бюджета.
- **API $/sample**.
- **Wall-clock latency P50/P95**.

---

## 5. Протокол

### 5.1 Гиперпараметры

```yaml
llm:
  temperature_greedy: 0.0
  temperature_sample: [0.6, 0.8, 1.0]   # для sampling-методов (M1, M2, MCTS rollout)
  top_p: 1.0
  max_tokens: 1024
search:
  self_consistency_N: [1, 2, 4, 8, 16]
  best_of_N: [1, 2, 4, 8, 16]
  beam_width: [1, 2, 4, 8]
  mcts_iterations: [4, 8, 16, 32, 64]
  mcts_c_exp: [0.5, 1.0, 1.41]
  expand_k: 3
reward:
  source: [self_eval, external_judge]
  judge_model: <stronger-than-policy>
tree:
  max_depth: [pillar_only, 3]            # granularity ablation A6
seed: [13, 42, 1337, 2025, 31415]        # 5 seeds
```

### 5.2 Compute-matched протокол (крае­угольный камень)

Для честного RQ1:

1. Измеряем фактический `tokens/sample` каждого метода для каждого значения его knob
   на dev-сете (он почти линеен по `N`/`w`/итерациям).
2. Определяем сетку бюджетов `B ∈ {B_1 < … < B_m}` (например, кратные стоимости
   одного greedy-прохода: 1×, 2×, 4×, 8×, 16×, 32×).
3. Для каждого `B` подбираем knob каждого метода так, чтобы ожидаемые токены ≈ `B`.
4. Сравниваем методы **только в точках равного `B`**. Headline-рисунок — `Acc_X(B)`.
5. Отдельно фиксируем latency, т.к. MCTS плохо параллелится, а self-consistency —
   тривиально (важный практический вывод).

### 5.3 Compute-бюджет (оценки на ценах apr 2026, OpenRouter)

Стоимость **доминируется методами с поиском**. Оценка на пример (policy=nano,
~2K input на вызов): greedy ≈ 1 вызов; SC/BoN с N=8 ≈ 8 вызовов; MCTS с 32 итер.
≈ 32–48 вызовов (selection дёшев, expansion+eval — основной расход).

**Pilot-бюджет ($100 OpenRouter)**

| Этап | Конфигурация | Цена |
|---|---|---|
| Dev / промпт-итерации | nano + ds-v3.1, ~5K calls | $5 |
| Gain-vs-budget кривые: 7 методов × 4 cheap-модели × ~500 samples × budget-grid | основная сетка RQ1+RQ2 | $55 |
| Reward-ablation (self-eval vs judge): 2 метода (M2, M4) × 4 модели × 500 samples | RQ3 | $15 |
| Search-space (M4 vs M5): 4 модели × 500 samples × budget-grid | RQ4 | $15 |
| Буфер | | $10 |
| **Pilot итого** | | **~$100** |

> Для pilot test-сет урезаем до ~500 стратифицированных примеров; полный 2K-прогон —
> в standard-фазе. Кэш делает повторные прогоны бесплатными.

**Standard ($300–600)**: полный 2K test, 5 seeds, mid-tier модели, granularity-ablation.
**Full ($1–2K)**: + premium-точка (сильный конец `α(π)`) для headline-таблицы и H2/H3.

#### Жёсткие правила экономии

1. Дисковый кэш по `hash(prompt + model_id)` обязателен — повторные прогоны бесплатны.
2. Dev — только nano и flash-lite; премиум — после стабилизации промпта.
3. Prompt caching на статическом префиксе (CWE-описания кандидатов) — −60% input.
4. Маленький dev-сет (50–100) для итерации, 500 (pilot) / 2000 (full) для прогона.
5. Все вызовы логируются в JSONL — постпроцессинг не требует пере-прогона.

### 5.4 Seeds и статистика

Каждая конфигурация — 5 seeds (sampling-методы стохастичны, MCTS тем более).
Сообщаем mean ± std + bootstrap-CI 95% (10K resamples). Главные сравнения — paired
McNemar на одних и тех же примерах, Bonferroni-поправка.

---

## 6. Ablation studies

| Ablation | Цель | Что меняется |
|---|---|---|
| A1. Budget sweep | Кривая `Acc(B)` (RQ1) | knob каждого метода под сетку `B` |
| A2. `c_exp` sweep | Чувствительность MCTS к exploration | `c_exp ∈ {0.5,1.0,1.41}` |
| A3. Reward source | Policy vs reward bottleneck (RQ3) | self-eval vs external-judge |
| A4. Search space | Taxonomy vs reasoning (RQ4) | M4 vs M5 при равном `B` |
| A5. Policy strength | Порог `α*` (RQ2) | 4 модели разной силы |
| A6. Tree granularity | Глубина vs ширина ошибки | pillar-only vs full-depth |
| A7. Sampling temperature | Diversity sampling-методов | T ∈ {0.6,0.8,1.0} |
| A8. Aggregation rule | majority vs reward-weighted vs robust-child | для M1/M2/M4 |
| A9. `expand_k` | Branch factor расширения MCTS | k ∈ {2,3,5} |

Каждое значение — ≥3 seeds (cheap), ключевые — 5 seeds. Автоматизация через
Hydra-конфиги + локальный планировщик прогонов.

---

## 7. Статистический анализ

- **Главное сравнение**: paired McNemar между MCTS и каждым методом семейства **в
  точках равного `B`**. Bonferroni по числу методов × бюджетов.
- **Effect size**: Cohen's h для разности пропорций.
- **Порог `α*` (H2)**: регрессия `Δ(π)` на `α(π)` по моделям; репортим точку
  пересечения нуля с CI (bootstrap по моделям/seeds).
- **Reward fidelity (H3)**: AUROC reward как предиктора корректности; сравнение
  self-eval vs judge.
- **Cascade analysis**: path-prefix error histogram, отдельно на pillar-уровне.
- **Failure mode taxonomy**: ручной разбор 100 ошибок (по 50 для слабой и сильной
  policy) с категориями: pillar-miss-at-L0, mid-level-miss, reward-misranking,
  search-collapse (поиск ушёл от верного раннего кандидата), ambiguous-ground-truth.

---

## 8. Воспроизводимость

Чек-лист на момент submission:

- [ ] Полный код в публичном репо (anonymized для double-blind).
- [ ] `pyproject.toml` с пин-версиями; Docker для open-weight inference.
- [ ] Hashed snapshots датасетов и дерева CWE (SHA-256).
- [ ] `scripts/reproduce_table_N.sh` / `reproduce_figure_N.sh` на каждую таблицу/рисунок.
- [ ] Random seeds зафиксированы и задокументированы.
- [ ] JSONL-лог всех LLM-вызовов (prompt + response + model_id + seed + tokens).
- [ ] Datasheet for Datasets на собранный бенчмарк.
- [ ] Snapshot model_id + provider + дата на каждую фазу.

---

## 9. Риски и mitigations

| Риск | Вероятность | Mitigation |
|---|---|---|
| MCTS не даёт прироста при равном `B` | **Высокая (и это ОК)** | Это и есть результат; статья диагностическая, нарратив строится вокруг «почему» |
| Эффект порога `α*` не виден на cheap-tier | Средняя | Обязательно добавить premium-точку (сильный конец оси) в headline-фазе |
| LLM видела CVE из test в pre-training | Высокая | Репортить cutoff; срез CVE после cutoff модели (§2.4) |
| Дисбаланс CWE искажает macro-метрики | Средняя | Стратификация теста; репорт и micro, и macro; head/mid/tail breakdown |
| Latency MCTS делает сравнение «нечестным» в практике | Средняя | Репортить отдельно latency-Pareto; отметить параллелизуемость SC |
| Reasoning-search (M5) трудно воспроизводим (свободный формат) | Средняя | Жёсткая JSON-схема шагов; фиксированный набор действий |
| Compute-бюджет не хватит на full | Средняя | Приоритет: RQ1+RQ2 на cheap; RQ3/RQ4 урезать по samples, не по seeds |

---

## 10. Timeline (12 недель до submission)

| Неделя | Milestone |
|---|---|
| 1 | Setup репо, литобзор (test-time scaling, MCTS-for-LLM, vuln-detection), сбор данных |
| 2 | MinHash-дедуп, построение дерева CWE, фиксация сплитов и статистики |
| 3 | Реализация policy-шага, M0/M1/M2 + reward-интерфейс, dev-итерация промптов |
| 4 | Реализация M3 (beam), M4 (MCTS-taxonomy); измерение tokens/sample под budget-grid |
| 5 | Реализация M5 (MCTS-reasoning); pilot-прогон RQ1 на cheap-tier |
| 6 | RQ2 (4 модели, ось `α`), кривые gain-vs-budget, первые числа |
| 7 | RQ3 (self-eval vs judge), RQ4 (M4 vs M5) |
| 8 | Statistical analysis, порог `α*`, cascade/failure-mode taxonomy |
| 9 | Headline-фаза: premium-точка; графики и таблицы |
| 10 | Draft: problem → experiments → method → analysis (в этом порядке) |
| 11 | Draft intro/related work; internal review; добор недостающих прогонов |
| 12 | Final polish, supplementary, submission |

---

## 11. Что нужно немедленно

1. Скачать PrimeVul + DiverseVul + BigVul + CVEFixes; MinHash-дедуп между ними.
2. Построить и зафиксировать дерево CWE из MITRE (pillars + depth 3), SHA-256.
3. Реализовать reward-интерфейс (self-eval / judge) и измерить `tokens/sample` для
   каждого метода → построить budget-grid (без этого RQ1 невозможен).
4. Запустить pilot на $100 согласно §5.3 — кривые `Acc(B)` на cheap-tier.
5. Согласовать с научником/PI авторство и ревью до submission.

---

## 12. Зафиксированная политика (lock file)

Не пересматривается без явного решения пользователя.

### 12.1 Научная политика

- Статья **диагностическая**: вопрос «когда/почему помогает test-time search»,
  **не** «наш метод лучше». Нулевой/отрицательный результат по MCTS — валидный исход.
- **Compute-matched** сравнение обязательно: все методы сравниваются в точках
  равного бюджета токенов, а не равного числа итераций.
- Две изолируемые оси обязательны: **policy strength** (RQ2) и **reward fidelity**
  (RQ3). Без обеих статья теряет диагностическую ценность.
- **Только публичные данные** (PrimeVul/DiverseVul/BigVul/CVEFixes + MITRE).
  Внутренний контур и корпоративные модели не используются и не цитируются.
- **PRM в этой статье не обучаем** — reward только self-eval и external-judge.
  PRM — Future Work.

### 12.2 Бюджетная политика

- Pilot — $100 OpenRouter, только cheap-tier (§3.7).
- Premium-tier подключается только после pilot и только для закрытия сильного конца
  оси `α(π)` (нужно для H2/H3).
- Open-weight (Qwen/Llama/DeepSeek) — через OpenRouter, свой GPU не обязателен.

### 12.3 Воспроизводимость

- Все LLM-calls в JSONL (prompt + response + model_id + timestamp + seed + tokens).
- Дисковый кэш по `hash(prompt + model_id)` обязателен.
- Snapshot моделей фиксируется на старте каждой фазы.

### 12.4 Условия пересмотра политики

- Смена целевой конференции.
- Появление бюджета > $500 сверх pilot или доступа к HPC (тогда можно вернуть PRM).
- Изменение цен OpenRouter > 30% по cheap-tier.

### 12.5 Источники цен (snapshot apr 2026)

- GPT-4.1-nano: https://openrouter.ai/openai/gpt-4.1-nano ($0.10 / $0.40)
- Gemini 2.5 Flash-Lite: https://openrouter.ai/google/gemini-2.5-flash-lite ($0.10 / $0.40)
- DeepSeek V3.1: https://openrouter.ai/deepseek/deepseek-chat-v3.1 ($0.15 / $0.75)
- Claude Haiku 4.5: https://openrouter.ai/anthropic/claude-haiku-4.5 ($1.00 / $5.00)
- OpenRouter pricing index: https://openrouter.ai/pricing
