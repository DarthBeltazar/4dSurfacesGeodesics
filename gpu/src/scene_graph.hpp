// Граф сцены (DD-011 §1, §3): библиотека нод (nodes/*.json), документ `.g4d` (version 2), перевод сцен version 1
// (geo4d.scenefile) в граф и сборка модели окна (scene.hpp) из графа с проверкой типов связей.
//
// Нода — запись библиотеки: тип, выход (manifold | shape | material | object), входы, параметры (float, int, vec3,
// vec4, color, string), формула на языке выражений (expr.hpp), для нынешних видов — legacy_kind (вид модели окна) и
// rows (параметр → поле строк объектов DD-008 в кодгене). Связь — "from": "<id>.out", "to": "<id>.<вход>".
#pragma once

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "scene.hpp"

namespace geo4d {

struct NodeParamDef {
    std::string name, type;                       // float | int | vec3 | vec4 | color | string
    nlohmann::json def;
    double min = 0, max = 0;
    bool structural = false;                      // меняет текст шейдера (ось полупространства)
    bool hidden = false;                          // не показывать в инспекторе
};

struct NodeInputDef {
    std::string name, type;
    bool multi = false;                           // несколько связей (объекты сцены), порядок — порядок связей
};

struct NodeDef {
    std::string type, group, title, output, formula, legacy_kind, material;
    std::vector<NodeParamDef> params;
    std::vector<NodeInputDef> inputs;
    std::map<std::string, std::string> rows;      // параметр → выражение строк объекта в obj_g (c, s.y, …)
    const NodeParamDef* param(const std::string& n) const;
    const NodeInputDef* input(const std::string& n) const;
};

class NodeLibrary {
public:
    void load_dir(const std::string& dir);        // все *.json: объект или массив объектов
    const NodeDef* find(const std::string& type) const;
    const std::vector<NodeDef>& defs() const { return defs_; }

private:
    std::vector<NodeDef> defs_;
};

// библиотека процесса: nodes/ из каталогов программы (Dirs); загружается при первом обращении
NodeLibrary& node_library();
void set_node_library_dir(const std::string& dir);

// значения параметров ноды (по умолчанию из библиотеки, поверх — из графа); векторы — компонентами c.x, c.y …
std::map<std::string, double> node_values(const NodeDef& def, const nlohmann::json& params);

// документ version 1 (scenes/*.json) → version 2 (граф); значения v1 переносятся явно (без умолчаний нод)
nlohmann::json migrate_v1(const nlohmann::json& v1);
// документ (version 1 или 2) → модель окна; ошибки графа — исключение с id ноды
Scene scene_from_document(const nlohmann::json& doc, const std::string& path, bool project);

// geo4d_gpu migrate <сцена.json> <сцена.g4d>
int run_migrate(int argc, char** argv);

}  // namespace geo4d
