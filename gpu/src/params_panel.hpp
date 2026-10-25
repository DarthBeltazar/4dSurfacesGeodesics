// Временная панель «Параметры» (DD-011, шаг 4; её заменит инспектор редактора нод, шаг 6): все параметры нод графа
// сцены. Правка числа применяется сразу, если структура шейдеров та же (слоты, codegen.hpp::structure_key), иначе —
// перегенерацией быстрого графа при отпускании ползунка; решает окно (app.cpp), панель только правит документ.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

class ParamsPanel {
public:
    // документ сцены: .g4d или version 1 (переводится в граф); false — ошибка в status()
    bool open(const std::string& path);
    const std::string& path() const { return path_; }
    void set_path(const std::string& p) { path_ = p; }
    const nlohmann::json& doc() const { return doc_; }
    bool dirty() const { return dirty_; }         // документ правился панелью (или сценарием)
    const std::string& status() const { return status_; }
    void set_status(const std::string& s) { status_ = s; }

    struct Change {
        bool changed = false;                     // значение изменилось в этом кадре
        bool commit = false;                      // правка завершена (ползунок отпущен, Enter) — можно перегенерировать
    };
    // enabled = false — только просмотр (режим редактирования DD-008: его граф без слотов)
    Change draw(bool enabled);
    // сценарий (директива setparam): число или вектор; false — нет ноды или параметра, сообщение в err
    bool set(const std::string& node, const std::string& param, const std::vector<double>& v, std::string& err);

private:
    std::string path_, status_;
    nlohmann::json doc_;
    bool dirty_ = false;
};
