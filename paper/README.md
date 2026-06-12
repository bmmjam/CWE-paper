# Paper scaffold

LaTeX-скелет статьи под arXiv / AAAI / ICLR.
Структура максимально нейтральна: `article` class + стандартные пакеты,
чтобы перенос на стиль конкретной конференции (NeurIPS, AAAI, ICLR) сводился
к замене `\documentclass` и пары команд.

## Структура

```
paper/
├── main.tex                    # точка входа, \input всех секций
├── references.bib              # библиография (placeholder ключи TODO_*)
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_problem.tex
│   ├── 04_method.tex
│   ├── 05_experiments.tex
│   ├── 06_analysis.tex
│   ├── 07_conclusion.tex
│   ├── A1_dataset_details.tex  # appendix
│   ├── A2_prompts.tex
│   └── A3_extra_results.tex
├── figures/                    # .pdf / .png рисунков
├── tables/                     # .tex автогенерированных таблиц
└── README.md
```

## Сборка

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

или классически:

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Перенос на стиль конференции

- **NeurIPS**: положить `neurips_2025.sty` рядом с `main.tex`,
  заменить заголовок документа на `\documentclass{article}\usepackage{neurips_2025}`.
- **AAAI**: заменить на `aaai26.sty`, добавить `\usepackage{aaai26}`,
  убрать `\maketitle` руками если стиль им управляет.
- **ICLR**: аналогично с `iclr2026_conference.sty`.
- **arXiv**: текущая конфигурация уже подходит, ничего менять не надо.

## Маркеры в тексте

- `\TODO{...}` — красные плейсхолдеры. Полнотекстовый grep:
  ```bash
  grep -rn "TODO" sections/ references.bib
  ```
- `TODO_*` ключи в `references.bib` нужно заменять на полноценные записи
  по мере того как related work выкристаллизовывается.

## Что писать первым

Не abstract и не introduction. Порядок, который реально работает:

1. `03_problem.tex` — формализуй задачу до кода. Это упрётся во все остальное.
2. `05_experiments.tex` — заполни Setup пока эксперименты ещё свежие в голове.
3. `04_method.tex` — описывай метод после того, как зафиксирован setup.
4. `06_analysis.tex` — пиши параллельно с прогоном экспериментов.
5. `02_related_work.tex` — последним, чтобы не цементировать позиционирование рано.
6. `01_introduction.tex` и `00_abstract.tex` — самым последним.

## Чек-лист перед submission

- [ ] Все `\TODO{...}` сняты или сознательно оставлены как known limitations.
- [ ] Все `TODO_*` bib-ключи заменены на реальные.
- [ ] Каждая таблица и рисунок упомянуты в тексте.
- [ ] Каждое утверждение про числа подтверждено цифрой в таблице.
- [ ] Anonymization для double-blind: убраны имена, ссылки на репо, благодарности.
- [ ] Supplementary в отдельном PDF, если требует CFP.
- [ ] Reproducibility checklist (специфический для каждой конференции) приложен.
