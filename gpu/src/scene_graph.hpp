// Граф сцены (DD-011 §1, §3): библиотека нод (nodes/*.json), документ `.g4d` (version 2), перевод сцен version 1
// (geo4d.scenefile) в граф и сборка модели окна (scene.hpp) из графа с проверкой типов связей.
//
// Нода — запись библиотеки: тип, выход (manifold | shape | field | inherit | material | object), входы, параметры (float, int, vec3,
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
    std::string role;                             // параметр одной роли поля (manifold | object, DD-015); пусто — обеих
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
    bool hidden = false;                          // прежние ноды (DD-015): загружаются, в меню их нет
    // поле (DD-015): роль → нода, которой оно читается в этой роли («as», пусто — свой код), переименование параметров
    struct RoleAlias { std::string as; std::map<std::string, std::string> rename; };
    std::map<std::string, RoleAlias> roles;
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

// Правило типов портов (DD-015 §1): вход manifold принимает manifold | field, shape — shape | field, any — всё
// геометрическое; выход «inherit» (xf.*) — тип его входа «shape» вверх по цепочке (не подключён — field). Пусто —
// связь допустима, иначе — сообщение. nodes/links — графа документа (без проверяемой связи).
std::string link_error(const NodeLibrary& lib, const nlohmann::json& nodes, const nlohmann::json& links,
                       const std::string& from, const std::string& to, const std::string& port);
// тип выхода ноды с учётом «inherit» (по связям графа)
std::string resolved_output(const NodeLibrary& lib, const nlohmann::json& nodes, const nlohmann::json& links, const std::string& id);
// роли поля по графу: биты 1 — многообразие, 2 — объект (обход вниз по связям до выходов сцены и объектов)
int field_roles(const NodeLibrary& lib, const nlohmann::json& nodes, const nlohmann::json& links, const std::string& id);

// документ version 1 (scenes/*.json) → version 2 (граф); значения v1 переносятся явно (без умолчаний нод)
nlohmann::json migrate_v1(const nlohmann::json& v1);
// документ (version 1 или 2) → модель окна; ошибки графа — исключение с id ноды
Scene scene_from_document(const nlohmann::json& doc, const std::string& path, bool project);

// geo4d_gpu migrate <сцена.json> <сцена.g4d>
int run_migrate(int argc, char** argv);

}  // namespace geo4d
