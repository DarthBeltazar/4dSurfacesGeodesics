"""Генерирует .run/*.run.xml — общие конфигурации запуска JetBrains (CLion, PyCharm): окна сцен, замер, ролик,
тесты, хост под отладчиком. Интерпретатор — .venv проекта (в CLion Python-интерпретатор обычно не настроен), путь
зависит от ОС, поэтому на Linux/macOS конфигурации нужно один раз перегенерировать:

    .venv/bin/python prototype/scripts/ide_configs.py          # Linux/macOS
    .venv/Scripts/python prototype/scripts/ide_configs.py      # Windows (так сгенерированы файлы в репозитории)
"""
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr as qa, escape

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / ".run"
R.mkdir(exist_ok=True)
VENV_PY = "$PROJECT_DIR$/.venv/" + ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
for f in R.glob("*.run.xml"):
    f.unlink()
MOD = "4dSurfacesHeodesics"

def py(fname, name, folder, script, params, before=()):
    tasks = "".join(f'\n      <option name="RunConfigurationTask" enabled="true" run_configuration_name={qa(b)} '
                    f'run_configuration_type="PythonConfigurationType" />' for b in before)
    (R / f"{fname}.run.xml").write_text(f"""<component name="ProjectRunConfigurationManager">
  <configuration default="false" name={qa(name)} type="PythonConfigurationType" factoryName="Python" folderName={qa(folder)}>
    <module name="{MOD}" />
    <option name="ENV_FILES" value="" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs>
      <env name="PYTHONUNBUFFERED" value="1" />
      <env name="PYTHONIOENCODING" value="utf-8" />
    </envs>
    <option name="SDK_HOME" value="{VENV_PY}" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="IS_MODULE_SDK" value="false" />
    <option name="ADD_CONTENT_ROOTS" value="true" />
    <option name="ADD_SOURCE_ROOTS" value="true" />
    <option name="SCRIPT_NAME" value="$PROJECT_DIR$/{script}" />
    <option name="PARAMETERS" value={qa(params)} />
    <option name="SHOW_COMMAND_LINE" value="false" />
    <option name="EMULATE_TERMINAL" value="false" />
    <option name="MODULE_MODE" value="false" />
    <option name="REDIRECT_INPUT" value="false" />
    <option name="INPUT_FILE" value="" />
    <method v="2">{tasks}
    </method>
  </configuration>
</component>
""", encoding="utf-8")

def pytest(fname, name, args):
    q = lambda s: escape(f'"{s}"', {'"': "&quot;"})
    (R / f"{fname}.run.xml").write_text(f"""<component name="ProjectRunConfigurationManager">
  <configuration default="false" name={qa(name)} type="tests" factoryName="py.test" folderName="Тесты">
    <module name="{MOD}" />
    <option name="ENV_FILES" value="" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs>
      <env name="PYTHONIOENCODING" value="utf-8" />
    </envs>
    <option name="SDK_HOME" value="{VENV_PY}" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="IS_MODULE_SDK" value="false" />
    <option name="ADD_CONTENT_ROOTS" value="true" />
    <option name="ADD_SOURCE_ROOTS" value="true" />
    <option name="_new_keywords" value="{q('')}" />
    <option name="_new_parameters" value="{q('')}" />
    <option name="_new_additionalArguments" value="{q(args)}" />
    <option name="_new_target" value="{q('$PROJECT_DIR$/tests')}" />
    <option name="_new_targetType" value="{q('PATH')}" />
    <method v="2" />
  </configuration>
</component>
""", encoding="utf-8")

def host(fname, name, params, before=()):
    tasks = "".join(f'\n      <option name="RunConfigurationTask" enabled="true" run_configuration_name={qa(b)} '
                    f'run_configuration_type="PythonConfigurationType" />' for b in before)
    (R / f"{fname}.run.xml").write_text(f"""<component name="ProjectRunConfigurationManager">
  <configuration default="false" name={qa(name)} type="CMakeRunConfiguration" factoryName="Application" folderName="Хост (C++)" PROGRAM_PARAMS={qa(params)} REDIRECT_INPUT="false" ELEVATE="false" USE_EXTERNAL_CONSOLE="false" EMULATE_TERMINAL="false" WORKING_DIR="file://$PROJECT_DIR$" PASS_PARENT_ENVS_2="true" PROJECT_NAME="geo4d_gpu" TARGET_NAME="geo4d_gpu" RUN_TARGET_PROJECT_NAME="geo4d_gpu" RUN_TARGET_NAME="geo4d_gpu">
    <method v="2">{tasks}
      <option name="com.jetbrains.cidr.execution.CidrBuildBeforeRunTaskProvider$BuildBeforeRunTask" enabled="true" />
    </method>
  </configuration>
</component>
""", encoding="utf-8")

SCENES = [("two_wormholes", "две червоточины (сцена 3)", "1280x720"), ("ellis", "Эллис", "1280x720"),
          ("S3", "S³", "1280x720"), ("tiger", "T³ («тигр»)", "1280x720"),
          ("spheritorus", "сферитор", "1280x720"), ("torisphere", "ториосфера", "1280x720")]
for key, title, res in SCENES:
    py(f"window_{key}", f"Окно: {title}", "Окно", "prototype/scripts/app.py", f"--scene {key} --res {res}")
py("bench_two_wormholes_1080p", "Замер окна: две червоточины 1080p", "Замеры", "prototype/scripts/app.py",
   "--bench --scene two_wormholes --res 1920x1080")
py("video_1080p", "Ролик: сквозь горловину 1080p, 64 spp", "Замеры", "prototype/scripts/video.py", "--res 1920x1080 --spp 64")
py("graph_two_wormholes", "Граф для хоста: две червоточины", "Хост (C++)", "prototype/scripts/app.py",
   "--graph-only gpu/build-debug/app/two_wormholes --res 1280x720")
host("host_info", "geo4d_gpu info", "info")
host("host_app_two_wormholes", "geo4d_gpu app: две червоточины (отладка хоста)",
     "app $PROJECT_DIR$/gpu/build-debug/app/two_wormholes/app_graph.txt", before=["Граф для хоста: две червоточины"])
pytest("tests_all", "Тесты: все (~15 мин)", "-q")
pytest("tests_gpu", "Тесты: GPU (~5 мин)", "-q -m gpu")
print("\n".join(sorted(p.name for p in R.iterdir())))
