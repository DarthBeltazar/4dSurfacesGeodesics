# 4D-гиперповерхности и геодезические: рендерер червоточин

Искривлённое 3D-пространство задаётся гиперповерхностью M = {x ∈ R⁴ : f(x) = 0}; лучи света —
геодезические индуцированной метрики. Полное задание — [TASK.md](TASK.md), текущий статус —
[docs/PROGRESS.md](docs/PROGRESS.md).

## Сборка и запуск (этап 1: прототип Python + JAX, CPU, float64)

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux: .venv/bin/python
.venv/Scripts/python -m pytest tests -q                    # все тесты (~4 мин)
.venv/Scripts/python -m pytest tests/test_u1_jet.py -q     # один файл
.venv/Scripts/python -m pytest "tests/test_u2_reference.py::test_vs_dop853[ellis]"   # один случай
```

Эксперименты этапа 1 (каждый пишет каталог `results/stage-1/<дата>_<имя>/` с config/meta/cmd):

```sh
.venv/Scripts/python prototype/scripts/convergence.py        # порядок сходимости
.venv/Scripts/python prototype/scripts/benchmark.py          # вызовы SDF на луч, метод × фигура
.venv/Scripts/python prototype/scripts/ellis_deflection.py   # угол отклонения против квадратуры
.venv/Scripts/python prototype/scripts/invariant_drift.py    # U3: графики дрейфа
.venv/Scripts/python prototype/scripts/render_ellis.py       # 512² Эллис + тепловая карта шагов
```

Этап 2 (`results/stage-2/`):

```sh
.venv/Scripts/python prototype/scripts/transfer_accuracy.py  # матрица переноса против разностей, стоимость Jet15
.venv/Scripts/python prototype/scripts/picture_tests.py      # картинки тестов S³ и S¹×R²
.venv/Scripts/python prototype/scripts/modea_vs_reference.py # режим A против 1024 spp (~10 мин CPU)
.venv/Scripts/python prototype/scripts/footprint_check.py    # след Якоби против ковариации стохастики
```

На Windows для кириллицы в консоли: `PYTHONIOENCODING=utf-8`.

## Обзор архитектуры

`prototype/geo4d/`:
- `jet.py` — jet-числа Jet7 (f, ∇f, ∇f·v, vᵀHv за один проход) и Jet15 (полный гессиан), пространство
  имён `ops`; SDF пишутся один раз и работают на массивах и jet-числах.
- `shapes.py` — тестовые фигуры и червоточины (`SHAPES`), обёртки для U5/U6.
- `rhs.py` — правая часть геодезической со стабилизацией Баумгарта и проекция.
- `integrators.py` — BS32/DOPRI5 с FSAL и адаптивным шагом, RK4 и наивная схема; `lax.while_loop` + `vmap`.
- `reference.py` — эталоны: DOP853 на автодифференцировании, квадратуры углов отклонения.
- `wormholes.py` — начальные состояния, условия выхода и асимптотические направления червоточин.
- `render.py` — процедурные фоны по листам, рендер этапа 1.
- `jacobi.py`, `camera.py`, `transfer.py` — поля Якоби, камера/тонкая линза, матрица переноса.
- `markers.py` — светящиеся маркеры для компактных фигур.
- `cubemap.py` — кубмапы, мипы, фильтр по эллипсу, .hdr.
- `modea.py` — режим A, детектор, фолбэк, эталонный стохастический рендерер.
- `results.py` — каталоги воспроизводимых результатов.

## Документы

- Формулы и конвенции: [docs/math.md](docs/math.md)
- Дизайн: [DD-001 интегратор](docs/design/DD-001-integrator.md),
  [DD-002 компактная червоточина и фигуры](docs/design/DD-002-wormhole-f-and-shapes.md),
  [DD-003 Якоби и режим A](docs/design/DD-003-jacobi-mode-a.md),
  [DD-006 тесты и эталоны](docs/design/DD-006-test-infrastructure.md)
- Решения: [docs/adr/](docs/adr/)
- Отчёты по этапам: [docs/reports/](docs/reports/)
- Журнал сессий: [docs/log/](docs/log/), известные проблемы: [docs/known-issues.md](docs/known-issues.md),
  изменения: [CHANGELOG.md](CHANGELOG.md)
