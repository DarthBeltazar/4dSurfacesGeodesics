// Редактор нод (DD-011 §8) — см. node_editor.hpp.
#include "node_editor.hpp"

#include <imgui.h>
#include <imgui_node_editor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <set>

#include "codegen.hpp"
#include "scene_graph.hpp"

namespace ed = ax::NodeEditor;
using geo4d::NodeDef;
using geo4d::NodeParamDef;
using geo4d::node_library;

namespace {

int components(const std::string& t) { return t == "vec3" || t == "color" ? 3 : t == "vec4" ? 4 : 1; }

// цвет порта — тип значения (DD-011 §8)
ImVec4 type_color(const std::string& t) {
    if (t == "manifold") return ImVec4(0.35f, 0.75f, 1.0f, 1);
    if (t == "shape") return ImVec4(1.0f, 0.65f, 0.25f, 1);
    if (t == "material") return ImVec4(0.95f, 0.45f, 0.85f, 1);
    if (t == "object") return ImVec4(0.45f, 0.95f, 0.45f, 1);
    return ImVec4(0.8f, 0.8f, 0.8f, 1);
}

constexpr int kOut = 255;                         // номер порта выхода; входы — 0…254

// значок порта: кружок цвета типа, у входа-списка (объекты сцены) — квадрат
void pin_mark(const ImVec4& c, bool multi) {
    const float h = ImGui::GetTextLineHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(h, h));
    const ImVec2 m(p.x + 0.5f * h, p.y + 0.5f * h);
    const ImU32 col = ImGui::ColorConvertFloat4ToU32(c);
    if (multi) ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(m.x - 0.3f * h, m.y - 0.3f * h), ImVec2(m.x + 0.3f * h, m.y + 0.3f * h), col);
    else ImGui::GetWindowDrawList()->AddCircleFilled(m, 0.3f * h, col);
}

std::pair<std::string, std::string> split_port(const std::string& s) {
    auto d = s.rfind('.');
    return d == std::string::npos ? std::make_pair(s, std::string()) : std::make_pair(s.substr(0, d), s.substr(d + 1));
}

}  // namespace

NodeEditor::NodeEditor() {
    ed::Config cfg;
    cfg.SettingsFile = nullptr;                   // положения нод — в документе (ui.pos), ADR-0009
    ed_ = ed::CreateEditor(&cfg);
}

NodeEditor::~NodeEditor() {
    if (ed_) ed::DestroyEditor(ed_);
}

bool NodeEditor::open(const std::string& path) {
    try {
        std::ifstream f(geo4d::from_u8(path), std::ios::binary);
        if (!f) throw std::runtime_error("не открыть " + path);
        nlohmann::json d = nlohmann::json::parse(f);
        doc_ = d.contains("manifold") && !d.contains("graphs") ? geo4d::migrate_v1(d) : d;
        if (!doc_["graphs"]["main"].contains("links")) links() = nlohmann::json::array();
        path_ = path;
        std::snprintf(open_path_, sizeof(open_path_), "%s", path.c_str());
        dirty_ = false;
        status_.clear();
        error_node_.clear();
        sel_.clear();
        undo_.clear();
        redo_.clear();
        layout_missing();
        place_ = true;
        fit_ = 2;
        return true;
    } catch (const std::exception& e) {
        status_ = e.what();
        return false;
    }
}

void NodeEditor::set_status(const std::string& s) {
    status_ = s;
    error_node_.clear();
    auto a = s.find("нода «");
    if (a != std::string::npos) {                 // ошибка графа (scene_graph.cpp: «граф сцены, нода «id»: …»)
        a += std::string("нода «").size();
        auto b = s.find("»", a);
        if (b != std::string::npos) error_node_ = s.substr(a, b - a);
    }
}

int NodeEditor::nid(const std::string& id) {
    auto it = nid_.find(id);
    if (it != nid_.end()) return it->second;
    return nid_[id] = next_nid_++;
}

void NodeEditor::snapshot() {
    undo_.push_back(doc_);
    if (undo_.size() > 200) undo_.erase(undo_.begin());
    redo_.clear();
}

void NodeEditor::undo() {
    if (undo_.empty()) return;
    redo_.push_back(doc_);
    doc_ = undo_.back();
    undo_.pop_back();
    place_ = true;
}

void NodeEditor::redo() {
    if (redo_.empty()) return;
    undo_.push_back(doc_);
    doc_ = redo_.back();
    redo_.pop_back();
    place_ = true;
}

// раскладка: столбцы по глубине (число связей до ноды «Сцена»), справа налево; строки — по порядку в документе
void NodeEditor::layout_missing() {
    std::map<std::string, int> depth;
    for (auto& n : nodes())
        if (n.value("type", std::string()) == "out.scene") depth[n.value("id", std::string())] = 0;
    for (int it = 0; it < 64; ++it) {
        bool ch = false;
        for (auto& l : links()) {
            const std::string f = split_port(l.value("from", std::string())).first, t = split_port(l.value("to", std::string())).first;
            if (!depth.count(t)) continue;
            const int dd = depth[t] + 1;
            if (!depth.count(f) || depth[f] < dd) { depth[f] = dd; ch = true; }
        }
        if (!ch) break;
    }
    std::map<int, int> rows;
    for (auto& n : nodes()) {
        const std::string id = n.value("id", std::string());
        const int d = depth.count(id) ? depth[id] : 0;
        const int r = rows[d]++;
        if (n.contains("ui") && n["ui"].contains("pos")) continue;
        n["ui"]["pos"] = {-340.0 * d, 150.0 * r};
    }
}

bool NodeEditor::set(const std::string& node, const std::string& param, const std::vector<double>& v, std::string& err) {
    nlohmann::json* n = nullptr;
    for (auto& x : nodes())
        if (x.value("id", std::string()) == node) n = &x;
    if (!n) { err = "нет ноды «" + node + "»"; return false; }
    const NodeDef* def = node_library().find(n->value("type", std::string()));
    const NodeParamDef* pd = def ? def->param(param) : nullptr;
    if (!pd || pd->type == "string") { err = "у ноды «" + node + "» нет числового параметра «" + param + "»"; return false; }
    const int c = components(pd->type);
    if (int(v.size()) != c) { err = "параметр «" + param + "»: нужно чисел — " + std::to_string(c); return false; }
    snapshot();
    (*n)["params"][param] = c == 1 ? (pd->type == "int" ? nlohmann::json(int(v[0])) : nlohmann::json(v[0])) : nlohmann::json(v);
    dirty_ = true;
    return true;
}

bool NodeEditor::add_node(const std::string& id, const std::string& type, std::string& err) {
    if (!node_library().find(type)) { err = "нет типа ноды «" + type + "»"; return false; }
    for (auto& n : nodes())
        if (n.value("id", std::string()) == id) { err = "id «" + id + "» уже есть"; return false; }
    snapshot();
    nodes().push_back({{"id", id}, {"type", type}, {"params", nlohmann::json::object()}});
    layout_missing();
    place_ = true;
    dirty_ = true;
    return true;
}

bool NodeEditor::link(const std::string& from, const std::string& to, std::string& err) {
    auto [fn, fp] = split_port(from);
    auto [tn, tp] = split_port(to);
    const NodeDef *fd = nullptr, *td = nullptr;
    for (auto& n : nodes()) {
        const std::string id = n.value("id", std::string());
        if (id == fn) fd = node_library().find(n.value("type", std::string()));
        if (id == tn) td = node_library().find(n.value("type", std::string()));
    }
    if (!fd || !td) { err = "нет ноды связи " + from + " → " + to; return false; }
    const geo4d::NodeInputDef* in = td->input(tp);
    if (fp != "out" || !in) { err = "нет порта " + (fp != "out" ? from : to); return false; }
    if (in->type != fd->output) { err = "типы портов не совпадают: " + fd->output + " → " + in->type; return false; }
    // цикл: выход tn уже ведёт (по связям вниз) к fn
    std::function<bool(const std::string&)> reaches = [&](const std::string& a) {
        if (a == fn) return true;
        for (auto& l : links())
            if (split_port(l.value("from", std::string())).first == a && reaches(split_port(l.value("to", std::string())).first)) return true;
        return false;
    };
    if (fn == tn || reaches(tn)) { err = "связь " + from + " → " + to + " замкнула бы цикл"; return false; }
    snapshot();
    if (!in->multi) {                             // вход на одну связь — прежняя заменяется
        auto& L = links();
        for (size_t i = L.size(); i-- > 0;)
            if (L[i].value("to", std::string()) == to) L.erase(L.begin() + i);
    }
    links().push_back({{"from", from}, {"to", to}});
    dirty_ = true;
    return true;
}

bool NodeEditor::unlink(const std::string& to, std::string& err) {
    auto& L = links();
    size_t n0 = L.size();
    nlohmann::json before = doc_;
    for (size_t i = L.size(); i-- > 0;)
        if (L[i].value("to", std::string()) == to) L.erase(L.begin() + i);
    if (L.size() == n0) { err = "нет связей ко входу " + to; return false; }
    undo_.push_back(before);
    redo_.clear();
    dirty_ = true;
    return true;
}

bool NodeEditor::del_node(const std::string& id, std::string& err) {
    auto& N = nodes();
    auto it = std::find_if(N.begin(), N.end(), [&](const nlohmann::json& n) { return n.value("id", std::string()) == id; });
    if (it == N.end()) { err = "нет ноды «" + id + "»"; return false; }
    if (it->value("type", std::string()) == "out.scene") { err = "ноду «Сцена» не удалить"; return false; }
    snapshot();
    N.erase(it);
    auto& L = links();
    for (size_t i = L.size(); i-- > 0;)
        if (split_port(L[i].value("from", std::string())).first == id || split_port(L[i].value("to", std::string())).first == id)
            L.erase(L.begin() + i);
    if (sel_ == id) sel_.clear();
    dirty_ = true;
    return true;
}

// инспектор выбранной ноды: ползунки параметров (диапазоны из библиотеки); числа — сразу, «*» — меняют шейдер
NodeEditor::Change NodeEditor::inspector(bool enabled) {
    Change ch;
    nlohmann::json* n = nullptr;
    for (auto& x : nodes())
        if (x.value("id", std::string()) == sel_) n = &x;
    if (!n) {
        ImGui::TextDisabled("выберите ноду на холсте");
        return ch;
    }
    const NodeDef* def = node_library().find(n->value("type", std::string()));
    if (!def) return ch;
    ImGui::TextUnformatted(def->title.c_str());
    ImGui::TextDisabled("%s · %s", sel_.c_str(), def->type.c_str());
    if (sel_ == error_node_) ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%s", status_.c_str());
    ImGui::Separator();
    ImGui::BeginDisabled(!enabled);
    nlohmann::json& params = (*n)["params"];
    if (params.is_null()) params = nlohmann::json::object();
    ImGui::PushID(sel_.c_str());
    ImGui::PushItemWidth(-70);
    for (const NodeParamDef& p : def->params) {
        if (p.hidden) continue;
        const nlohmann::json cur = params.contains(p.name) ? params[p.name] : p.def;
        bool changed = false;
        nlohmann::json val;
        const std::string lab = p.structural || p.type == "int" || (p.type == "string" && p.name == "formula") ? p.name + " *" : p.name;
        if (p.type == "float") {
            float x = cur.is_number() ? cur.get<float>() : 0.0f;
            const float speed = p.max > p.min ? float(p.max - p.min) / 400.0f : 0.005f;
            changed = ImGui::DragFloat(lab.c_str(), &x, speed, float(p.min), float(p.max), "%.4g");
            val = double(x);
        } else if (p.type == "int") {
            int x = cur.is_number() ? cur.get<int>() : 0;
            changed = ImGui::InputInt(lab.c_str(), &x);
            if (p.max > p.min) x = std::clamp(x, int(p.min), int(p.max));
            val = x;
        } else if (p.type == "vec3" || p.type == "vec4" || p.type == "color") {
            const int c = components(p.type);
            float x[4] = {0, 0, 0, 0};
            for (int i = 0; i < c && cur.is_array() && i < int(cur.size()); ++i) x[i] = cur[i].get<float>();
            if (p.type == "color") changed = ImGui::ColorEdit3(lab.c_str(), x, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
            else if (c == 3) changed = ImGui::DragFloat3(lab.c_str(), x, 0.005f, 0, 0, "%.4g");
            else changed = ImGui::DragFloat4(lab.c_str(), x, 0.005f, 0, 0, "%.4g");
            val = nlohmann::json::array();
            for (int i = 0; i < c; ++i) val.push_back(double(x[i]));
        } else if (p.type == "string") {
            std::string s = cur.is_string() ? cur.get<std::string>() : "";
            char buf[1024];
            std::snprintf(buf, sizeof(buf), "%s", s.c_str());
            if (p.name == "formula") ImGui::InputTextMultiline(lab.c_str(), buf, sizeof(buf), ImVec2(-70, 60));
            else ImGui::InputText(lab.c_str(), buf, sizeof(buf));
            if (ImGui::IsItemDeactivatedAfterEdit() && s != buf) {
                snapshot();
                params[p.name] = std::string(buf);
                dirty_ = true;
                ch.changed = ch.commit = true;
            }
            continue;
        } else continue;
        if (ImGui::IsItemActivated()) snapshot(); // начало правки — снимок для undo (вся правка ползунком — один шаг)
        if (changed) {
            params[p.name] = val;
            dirty_ = true;
            ch.changed = true;
        }
        // числа применяются сразу; перегенерация (если структура другая) — когда правка завершена
        if (ImGui::IsItemDeactivatedAfterEdit() || (changed && p.type == "int")) ch.commit = true;
    }
    ImGui::PopItemWidth();
    ImGui::PopID();
    if (def->type != "out.scene" && ImGui::Button("удалить ноду")) {
        std::string err;
        if (del_node(sel_, err)) ch.changed = ch.commit = true;
    }
    ImGui::EndDisabled();
    return ch;
}

std::string NodeEditor::object_node(int k) const {
    if (!doc_.contains("graphs")) return "";
    std::string scene;
    for (auto& n : doc_["graphs"]["main"]["nodes"])
        if (n.value("type", std::string()) == "out.scene") scene = n.value("id", std::string());
    int i = 0;
    for (auto& l : doc_["graphs"]["main"].value("links", nlohmann::json::array()))
        if (l.value("to", std::string()) == scene + ".objects" && i++ == k) return split_port(l.value("from", std::string())).first;
    return "";
}

NodeEditor::Change NodeEditor::draw(bool enabled) {
    Change ch;
    if (!ImGui::Begin("Граф")) {
        ImGui::End();
        return ch;
    }
    if (!doc_.contains("graphs")) {
        ImGui::TextWrapped("%s", status_.empty() ? "нет графа сцены" : status_.c_str());
        ImGui::End();
        return ch;
    }
    // строка инструментов: undo/redo, файл, сообщение
    ImGui::BeginDisabled(!enabled || undo_.empty());
    if (ImGui::Button("отменить")) { undo(); ch.changed = ch.commit = true; }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!enabled || redo_.empty());
    if (ImGui::Button("повторить")) { redo(); ch.changed = ch.commit = true; }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(120.0f, ImGui::GetContentRegionAvail().x * 0.35f));
    ImGui::InputText("##file", open_path_, sizeof(open_path_));
    ImGui::SameLine();
    ImGui::BeginDisabled(!enabled);
    if (ImGui::Button("открыть")) {
        if (open(open_path_)) ch.changed = ch.commit = ch.opened = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("сохранить")) ch.save_path = open_path_;
    ImGui::EndDisabled();
    if (!status_.empty()) {
        ImGui::SameLine();
        if (error_node_.empty()) ImGui::TextDisabled("%s", status_.c_str());
        else ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%s", status_.c_str());
    }
    // Ctrl+Z / Ctrl+Y — в окне графа
    if (enabled && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::GetIO().KeyCtrl) {
        if (ImGui::IsKeyPressed(ImGuiKey_Z) && !undo_.empty()) { undo(); ch.changed = ch.commit = true; }
        if (ImGui::IsKeyPressed(ImGuiKey_Y) && !redo_.empty()) { redo(); ch.changed = ch.commit = true; }
    }

    const float insp_w = 320.0f;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::BeginChild("canvas", ImVec2(std::max(100.0f, avail.x - insp_w), 0));
    ed::SetCurrentEditor(ed_);
    ed::Begin("graph");
    struct PinRef { std::string node, port, type; bool out; bool multi; };
    std::map<uintptr_t, PinRef> pins;
    for (auto& n : nodes()) {
        const std::string id = n.value("id", std::string());
        const NodeDef* def = node_library().find(n.value("type", std::string()));
        if (!def) continue;
        const int k = nid(id);
        if (place_ && n.contains("ui") && n["ui"].contains("pos"))
            ed::SetNodePosition(ed::NodeId(k), ImVec2(n["ui"]["pos"][0].get<float>(), n["ui"]["pos"][1].get<float>()));
        const bool err = id == error_node_;
        if (err) ed::PushStyleColor(ed::StyleColor_NodeBorder, ImVec4(1, 0.25f, 0.25f, 1));
        ed::BeginNode(ed::NodeId(k));
        ImGui::TextUnformatted(def->title.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("%s", id.c_str());
        for (size_t j = 0; j < def->inputs.size() && j < size_t(kOut); ++j) {
            const auto& in = def->inputs[j];
            const uintptr_t pid = uintptr_t(k) * 256 + j;
            pins[pid] = {id, in.name, in.type, false, in.multi};
            ed::BeginPin(ed::PinId(pid), ed::PinKind::Input);
            pin_mark(type_color(in.type), in.multi);
            ImGui::SameLine();
            ImGui::TextUnformatted(in.name.c_str());
            ed::EndPin();
        }
        if (!def->output.empty()) {
            const uintptr_t pid = uintptr_t(k) * 256 + kOut;
            pins[pid] = {id, "out", def->output, true, false};
            ed::BeginPin(ed::PinId(pid), ed::PinKind::Output);
            ImGui::TextDisabled("%s", def->output.c_str());
            ImGui::SameLine();
            pin_mark(type_color(def->output), false);
            ed::EndPin();
        }
        ed::EndNode();
        if (err) ed::PopStyleColor();
    }
    place_ = false;
    // связи: номер — позиция в списке документа (+1)
    auto pin_of = [&](const std::string& port, bool out) -> uintptr_t {
        auto [n, p] = split_port(port);
        if (!nid_.count(n)) return 0;
        for (auto& x : nodes())
            if (x.value("id", std::string()) == n) {
                const NodeDef* def = node_library().find(x.value("type", std::string()));
                if (!def) return 0;
                if (out) return uintptr_t(nid_[n]) * 256 + kOut;
                for (size_t j = 0; j < def->inputs.size(); ++j)
                    if (def->inputs[j].name == p) return uintptr_t(nid_[n]) * 256 + j;
            }
        return 0;
    };
    for (size_t i = 0; i < links().size(); ++i) {
        const auto& l = links()[i];
        const uintptr_t a = pin_of(l.value("from", std::string()), true), b = pin_of(l.value("to", std::string()), false);
        if (a && b) ed::Link(ed::LinkId(i + 1), ed::PinId(a), ed::PinId(b), pins.count(a) ? type_color(pins[a].type) : ImVec4(1, 1, 1, 1));
    }
    if (enabled) {
        // новая связь: от выхода к входу того же типа (иначе — красная и не создаётся)
        if (ed::BeginCreate()) {
            ed::PinId a, b;
            if (ed::QueryNewLink(&a, &b) && a && b && pins.count(a.Get()) && pins.count(b.Get())) {
                PinRef pa = pins[a.Get()], pb = pins[b.Get()];
                if (!pa.out) std::swap(pa, pb);
                std::string err;
                if (!pa.out || pb.out || pa.type != pb.type) ed::RejectNewItem(ImVec4(1, 0.3f, 0.3f, 1), 2.0f);
                else if (ed::AcceptNewItem(ImVec4(0.4f, 1, 0.4f, 1), 3.0f)) {
                    if (link(pa.node + ".out", pb.node + "." + pb.port, err)) ch.changed = ch.commit = true;
                    else set_status(err);
                }
            }
        }
        ed::EndCreate();
        // удаление: связи и ноды (Delete на выделении); «Сцену» не удалить
        if (ed::BeginDelete()) {
            std::vector<size_t> dl;
            std::vector<std::string> dn;
            ed::LinkId l;
            while (ed::QueryDeletedLink(&l))
                if (ed::AcceptDeletedItem()) dl.push_back(size_t(l.Get()) - 1);
            ed::NodeId nd;
            while (ed::QueryDeletedNode(&nd)) {
                std::string id;
                for (auto& [s, v] : nid_) if (uintptr_t(v) == nd.Get()) id = s;
                bool scene = false;
                for (auto& x : nodes()) scene = scene || (x.value("id", std::string()) == id && x.value("type", std::string()) == "out.scene");
                if (!id.empty() && !scene && ed::AcceptDeletedItem()) dn.push_back(id);
                else ed::RejectDeletedItem();
            }
            if (!dl.empty() || !dn.empty()) {
                snapshot();
                std::sort(dl.rbegin(), dl.rend());
                for (size_t i : dl) if (i < links().size()) links().erase(links().begin() + i);
                for (auto& id : dn) {
                    std::string err;
                    if (del_node(id, err)) undo_.pop_back();   // свой снимок del_node — лишний: всё удаление — один шаг
                }
                dirty_ = true;
                ch.changed = ch.commit = true;
            }
        }
        ed::EndDelete();
    }
    // контекстное меню холста: добавить ноду (группы библиотеки, поиск по имени)
    const ImVec2 mouse = ImGui::GetMousePos();
    ed::Suspend();
    if (enabled && ed::ShowBackgroundContextMenu()) {
        ImGui::OpenPopup("add_node");
        search_[0] = 0;
    }
    static ImVec2 at;
    if (ImGui::BeginPopup("add_node")) {
        if (ImGui::IsWindowAppearing()) { at = mouse; ImGui::SetKeyboardFocusHere(); }
        ImGui::InputText("поиск", search_, sizeof(search_));
        std::map<std::string, std::vector<const NodeDef*>> groups;
        const std::string q = search_;
        for (auto& d : node_library().defs())
            if (q.empty() || d.title.find(q) != std::string::npos || d.type.find(q) != std::string::npos) groups[d.group].push_back(&d);
        const NodeDef* pick = nullptr;
        for (auto& [g, v] : groups) {
            if (!q.empty()) {
                for (auto* d : v) if (ImGui::MenuItem(d->title.c_str())) pick = d;
            } else if (ImGui::BeginMenu(g.c_str())) {
                for (auto* d : v) if (ImGui::MenuItem(d->title.c_str())) pick = d;
                ImGui::EndMenu();
            }
        }
        if (pick) {
            std::string base = pick->type.substr(pick->type.find('.') + 1), id = base;
            for (int i = 2; std::any_of(nodes().begin(), nodes().end(), [&](const nlohmann::json& n) { return n.value("id", std::string()) == id; }); ++i)
                id = base + "_" + std::to_string(i);
            std::string err;
            if (add_node(id, pick->type, err)) {
                const ImVec2 p = ed::ScreenToCanvas(at);
                nodes().back()["ui"]["pos"] = {p.x, p.y};
                place_ = true;
                sel_ = id;
                ch.changed = ch.commit = true;
            }
        }
        ImGui::EndPopup();
    }
    ed::Resume();
    // выделение (одна нода — в инспекторе) и положения нод — в документ (без применения правки)
    if (select_req_ && nid_.count(sel_)) {
        ed::SelectNode(ed::NodeId(nid_[sel_]));
        ed::NavigateToSelection(false, 0.2f);
        select_req_ = false;
    } else if (ed::GetSelectedObjectCount() > 0) {
        ed::NodeId sel[1];
        if (ed::GetSelectedNodes(sel, 1) == 1)
            for (auto& [s, v] : nid_) if (uintptr_t(v) == sel[0].Get()) sel_ = s;
    }
    for (auto& n : nodes()) {
        const std::string id = n.value("id", std::string());
        if (!nid_.count(id)) continue;
        const ImVec2 p = ed::GetNodePosition(ed::NodeId(nid_[id]));
        if (p.x < 1e30f) n["ui"]["pos"] = {std::round(p.x), std::round(p.y)};
    }
    if (fit_ > 0 && --fit_ == 0) ed::NavigateToContent(0.0f);
    ed::End();
    ed::SetCurrentEditor(nullptr);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("inspector", ImVec2(0, 0), ImGuiChildFlags_Borders);
    Change ci = inspector(enabled);
    ImGui::EndChild();
    ImGui::End();
    ch.changed = ch.changed || ci.changed;
    ch.commit = ch.commit || ci.commit;
    return ch;
}
