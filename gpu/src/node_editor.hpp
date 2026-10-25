// Редактор нод (DD-011 §8; холст — imgui-node-editor, ADR-0009): граф сцены `.g4d` на холсте — ноды с портами
// (цвет — тип), связи только между совместимыми портами, добавление (контекстное меню по группам и поиск), удаление,
// перемещение, выделение; инспектор выбранной ноды; undo/redo — снимки документа; открыть/сохранить. Редактор правит
// только документ: применяет правку окно (app.cpp: слоты — сразу, иначе предпросмотр на интерпретаторе и сборка в
// фоне). Положения нод — в документе (ui.pos), у нод без положения — раскладка по глубине от ноды «Сцена».
#pragma once

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ax::NodeEditor { struct EditorContext; }

class NodeEditor {
public:
    NodeEditor();
    ~NodeEditor();
    // документ сцены: .g4d или version 1 (переводится в граф); false — ошибка в status()
    bool open(const std::string& path);
    const std::string& path() const { return path_; }
    const nlohmann::json& doc() const { return doc_; }
    bool dirty() const { return dirty_; }         // документ правился (окно применило правку хотя бы раз)
    const std::string& status() const { return status_; }
    // сообщение окна; ошибка графа «нода «id»: …» — красная рамка этой ноды
    void set_status(const std::string& s);

    struct Change {
        bool changed = false;                     // документ изменился в этом кадре
        bool commit = false;                      // правка завершена (ползунок отпущен, связь, нода, undo)
        bool opened = false;                      // открыт другой файл сцены (камера — из него)
        std::string save_path;                    // сохранить документ с текущей камерой сюда
    };
    // окно «Граф»: холст и инспектор; enabled = false — только просмотр
    Change draw(bool enabled);
    // сценарий: setparam — число или вектор; addnode / link / unlink / delnode — как действия на холсте
    bool set(const std::string& node, const std::string& param, const std::vector<double>& v, std::string& err);
    bool add_node(const std::string& id, const std::string& type, std::string& err);
    bool link(const std::string& from, const std::string& to, std::string& err);       // «id.out» → «id.вход»
    bool unlink(const std::string& to, std::string& err);                              // все связи ко входу
    bool del_node(const std::string& id, std::string& err);
    void undo();
    void redo();
    const std::string& selected() const { return sel_; }
    // нода «Объект» (out.object) объекта сцены k — порядок связей входа scene.objects; пусто — нет
    std::string object_node(int k) const;
    void select(const std::string& id) { sel_ = id; select_req_ = true; }

private:
    std::string path_, status_, error_node_, sel_;
    nlohmann::json doc_;
    bool dirty_ = false, select_req_ = false;
    std::vector<nlohmann::json> undo_, redo_;
    ax::NodeEditor::EditorContext* ed_ = nullptr;
    std::map<std::string, int> nid_;              // id ноды → номер на холсте (не переиспользуются)
    int next_nid_ = 1;
    bool place_ = true;                           // задать положения нод на холсте из документа
    int fit_ = 2;                                 // показать весь граф (через кадр: размеры нод известны после отрисовки)
    char open_path_[512] = {}, search_[64] = {};

    int nid(const std::string& id);
    void snapshot();                              // перед структурной правкой: в undo
    void layout_missing();                        // ui.pos нодам без положения
    nlohmann::json& nodes() { return doc_["graphs"]["main"]["nodes"]; }
    nlohmann::json& links() { return doc_["graphs"]["main"]["links"]; }
    Change inspector(bool enabled);
};
