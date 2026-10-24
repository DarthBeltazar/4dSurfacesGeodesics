# 4D-гиперповерхности и геодезические: рендерер червоточин

Искривлённое 3D-пространство задаётся гиперповерхностью M = {x ∈ R⁴ : f(x) = 0}; лучи света —
геодезические индуцированной метрики. Полное задание — [TASK.md](TASK.md), текущий статус —
[docs/PROGRESS.md](docs/PROGRESS.md).

## Сборка и запуск (прототип Python + JAX, CPU, float64)

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux: .venv/bin/python
.venv/Scripts/python -m pytest tests -q                    # все тесты (~8 мин)
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

## Этап 3: GPU (Slang → Vulkan compute, float32)

Нужны Vulkan SDK 1.4.328.1 (slangc, volk, VMA; путь из `VULKAN_SDK`), MSVC 2022, CMake/Ninja (берутся из CLion).

```sh
gpu\build.cmd                                               # Release → gpu/build/geo4d_gpu.exe
gpu\build\geo4d_gpu.exe info                                # устройство и драйвер
.venv/Scripts/python -m pytest tests -q -m gpu             # GPU-тесты (~30 с); без бинарника пропускаются
.venv/Scripts/python prototype/scripts/gpu_vs_prototype.py  # таблица GPU / прототип-f32 / f64 по фигурам
.venv/Scripts/python prototype/scripts/gpu_kappa_alpha.py   # выбор κ и α во float32
.venv/Scripts/python prototype/scripts/gpu_profile.py       # профиль режима A 1080p/4K
.venv/Scripts/python prototype/scripts/gpu_modea_quality.py # качество режима A против эталона этапа 2
```

## Этап 4: режим B — трассировка путей (`results/stage-4/`)

Денойзер OptiX: при первом конфигурировании CMake скачивает заголовки `NVIDIA/optix-dev` (нужна сеть); CUDA Toolkit
не нужен (ADR-0007). Без OptiX: `cmake -DGEO4D_OPTIX=OFF`.

```sh
.venv/Scripts/python -m pytest tests/test_objects.py tests/test_pathtrace.py -q   # прототип: попадания, печь, MIS
.venv/Scripts/python -m pytest tests/test_gpu_pt.py -q                            # GPU против прототипа, денойзер
.venv/Scripts/python prototype/scripts/gpu_pt_quality.py   # смещение конусов/допуска, денойзер, кадр против эталона
.venv/Scripts/python prototype/scripts/gpu_pt_profile.py   # 1080p/4K: 1 spp и 64 spp + денойзер
```

Шейдеры компилирует Python (`geo4d.gpu.compile_shader`, кэш в `gpu/build/spv/`); хост исполняет job-файлы
(`run`) и графы диспетчеризаций (`graph`), данные — через `.npy`.

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
- `gpu.py`, `gpu_modea.py` — стенд GPU: компиляция шейдеров, запуск `geo4d_gpu`, трассировка и режим A на GPU.
- `objects.py` — объекты 4D SDF ∩ M: предел шага, эрмитово уточнение попадания, кадр оттенения (M26–M28).
- `pathtrace.py` — трассировщик путей прототипа (плоские зоны, BSDF, NEE/MIS — M29), сцены режима B;
  `euclid_pt.py` — независимый евклидов эталон; `gpu_pt.py` — генерация сцены для GPU, кадр режима B, денойзер.

`gpu/`: `src/` — хост (Vulkan-контекст, `run`/`graph`, .npy); `shaders/` — `jet.slang` (jet-числа за `IReal`),
`scenes.slang` (SDF), `integrator.slang`, `trace.slang`, `modea.slang`, `wavefront.slang`, `bc6h_*.slang`.

## Документы

- Формулы и конвенции: [docs/math.md](docs/math.md)
- Дизайн: [DD-001 интегратор](docs/design/DD-001-integrator.md),
  [DD-002 компактная червоточина и фигуры](docs/design/DD-002-wormhole-f-and-shapes.md),
  [DD-003 Якоби и режим A](docs/design/DD-003-jacobi-mode-a.md),
  [DD-004 архитектура GPU](docs/design/DD-004-gpu-architecture.md),
  [DD-005 path tracing](docs/design/DD-005-path-tracing.md),
  [DD-006 тесты и эталоны](docs/design/DD-006-test-infrastructure.md)
- Решения: [docs/adr/](docs/adr/)
- Отчёты по этапам: [docs/reports/](docs/reports/)
- Журнал сессий: [docs/log/](docs/log/), известные проблемы: [docs/known-issues.md](docs/known-issues.md),
  изменения: [CHANGELOG.md](CHANGELOG.md)
