// Панель «Параметры» (DD-011, шаг 4) — см. params_panel.hpp.
#include "params_panel.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "codegen.hpp"
#include "scene_graph.hpp"

using geo4d::NodeDef;
using geo4d::NodeParamDef;
using geo4d::node_library;

bool ParamsPanel::open(const std::string& path) {
    try {
        std::ifstream f(geo4d::from_u8(path), std::ios::binary);
        if (!f) throw std::runtime_error("не открыть " + path);
        nlohmann::json d = nlohmann::json::parse(f);
        doc_ = d.contains("manifold") && !d.contains("graphs") ? geo4d::migrate_v1(d) : d;
        path_ = path;
        dirty_ = false;
        status_.clear();
        return true;
    } catch (const std::exception& e) {
        status_ = e.what();
        doc_ = nlohmann::json();
        return false;
    }
}

namespace {

int components(const std::string& t) { return t == "vec3" || t == "color" ? 3 : t == "vec4" ? 4 : 1; }

nlohmann::json* find_node(nlohmann::json& doc, const std::string& id) {
    if (!doc.contains("graphs")) return nullptr;
    for (auto& n : doc["graphs"]["main"]["nodes"])
        if (n.value("id", std::string()) == id) return &n;
    return nullptr;
}

}  // namespace

bool ParamsPanel::set(const std::string& node, const std::string& param, const std::vector<double>& v, std::string& err) {
    nlohmann::json* n = find_node(doc_, node);
    if (!n) { err = "нет ноды «" + node + "»"; return false; }
    const NodeDef* def = node_library().find(n->value("type", std::string()));
    const NodeParamDef* pd = def ? def->param(param) : nullptr;
    if (!pd || pd->type == "string") { err = "у ноды «" + node + "» нет числового параметра «" + param + "»"; return false; }
    const int c = components(pd->type);
    if (int(v.size()) != c) { err = "параметр «" + param + "»: нужно чисел — " + std::to_string(c); return false; }
    nlohmann::json val = c == 1 ? (pd->type == "int" ? nlohmann::json(int(v[0])) : nlohmann::json(v[0])) : nlohmann::json(v);
    (*n)["params"][param] = val;
    dirty_ = true;
    return true;
}

ParamsPanel::Change ParamsPanel::draw(bool enabled) {
    Change ch;
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 430), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330, 420), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Параметры")) {
        ImGui::End();
        return ch;
    }
    if (!doc_.contains("graphs")) {
        ImGui::TextWrapped("%s", status_.empty() ? "нет графа сцены" : status_.c_str());
        ImGui::End();
        return ch;
    }
    if (!enabled) ImGui::TextDisabled("в режиме редактирования (Tab) — только просмотр");
    if (!status_.empty()) ImGui::TextWrapped("%s", status_.c_str());
    ImGui::BeginDisabled(!enabled);
    for (auto& n : doc_["graphs"]["main"]["nodes"]) {
        const std::string id = n.value("id", std::string()), type = n.value("type", std::string());
        const NodeDef* def = node_library().find(type);
        if (!def) continue;
        bool any = false;
        for (auto& p : def->params) any = any || (!p.hidden && p.type != "string");
        if (!any && type != "mf.formula") continue;
        if (!ImGui::CollapsingHeader((id + " — " + def->title + "###" + id).c_str())) continue;
        ImGui::PushID(id.c_str());
        nlohmann::json& params = n["params"];
        if (params.is_null()) params = nlohmann::json::object();
        for (const NodeParamDef& p : def->params) {
            if (p.hidden) continue;
            const nlohmann::json cur = params.contains(p.name) ? params[p.name] : p.def;
            bool changed = false;
            nlohmann::json val;
            // структурный параметр (меняет текст шейдера) — по завершении правки; отмечен «*»
            const std::string lab = p.structural || p.type == "int" || p.type == "string" ? p.name + " *" : p.name;
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
            } else if (p.type == "string" && p.name == "formula") {
                std::string s = cur.is_string() ? cur.get<std::string>() : "";
                char buf[1024];
                std::snprintf(buf, sizeof(buf), "%s", s.c_str());
                ImGui::InputTextMultiline(lab.c_str(), buf, sizeof(buf), ImVec2(-1, 60));
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    params[p.name] = std::string(buf);
                    dirty_ = true;
                    ch.changed = ch.commit = true;
                }
                continue;
            } else continue;
            if (changed) {
                params[p.name] = val;
                dirty_ = true;
                ch.changed = true;
            }
            // числа применяются сразу; перегенерация (если структура другая) — когда правка завершена
            if (ImGui::IsItemDeactivatedAfterEdit() || (changed && p.type == "int")) ch.commit = true;
        }
        ImGui::PopID();
    }
    ImGui::EndDisabled();
    ImGui::End();
    return ch;
}
