// Редактор сцен (DD-008) — см. editor.hpp.
#include "editor.hpp"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

const char* kKinds[] = {"hyperplane", "S3", "spheritorus", "torisphere", "tiger", "ellis", "compact_wh", "two_compact_wh"};
const char* kKindNames[] = {"гиперплоскость R³", "S³", "сферитор", "ториосфера", "T³ («тигр»)", "червоточина Эллиса",
                            "компактная червоточина", "две компактные червоточины"};

// параметры по умолчанию — как geo4d.scenefile.MANIFOLDS
std::vector<ManifoldParam> default_params(const std::string& kind) {
    if (kind == "S3") return {{"R", {1.0f}}};
    if (kind == "spheritorus" || kind == "torisphere") return {{"R", {1.0f}}, {"r", {0.4f}}};
    if (kind == "tiger") return {{"R1", {1.2f}}, {"R2", {0.8f}}, {"r", {0.3f}}};
    if (kind == "ellis") return {{"b", {1.0f}}};
    if (kind == "compact_wh") return {{"b", {1.0f}}, {"rho0", {3.0f}}, {"W", {1.5f}}};
    if (kind == "two_compact_wh")
        return {{"c1", {-4, 0, 0}}, {"c2", {4, 0, 0}}, {"b", {1.0f}}, {"rho0", {3.0f}}, {"W", {1.5f}}};
    return {};
}

std::string num(float v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.9g", double(v));
    std::string s = b;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";      // как в Python: 1.0, не 1
    return s;
}

std::string vec(const float* v, int n) {
    std::string s = "[";
    for (int i = 0; i < n; ++i) s += (i ? ", " : "") + num(v[i]);
    return s + "]";
}

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') o += '\\';
        o += ch;
    }
    return o + "\"";
}

float max3(const float* v) { return std::max(v[0], std::max(v[1], v[2])); }

}  // namespace

// ---- модель: M34, упаковка строк -------------------------------------------------------------------

std::vector<int> nee_lights(const std::vector<EdObject>& objs, const EdFlat& flat) {
    std::vector<int> out;
    if (!flat.has) return out;
    for (int k = 0; k < int(objs.size()); ++k) {
        const EdObject& o = objs[k];
        if (o.mat != 2 || o.geom != 0) continue;
        float dw = 1e30f;
        for (float s : flat.sheets) dw = std::min(dw, std::fabs(o.c[3] - s));
        if (dw >= 1e-6f) continue;
        bool clear = true;
        for (auto& c : flat.balls) {
            double dx = o.c[0] - c[0], dy = o.c[1] - c[1], dz = o.c[2] - c[2];
            if (std::sqrt(dx * dx + dy * dy + dz * dz) < double(flat.rho0) + o.a) clear = false;
        }
        if (clear) out.push_back(k);
    }
    return out;
}

std::vector<float> pack_rows(const std::vector<EdObject>& objs, const EdFlat& flat) {
    if (objs.size() > size_t(kMaxObj)) throw std::runtime_error("объектов больше PT_MAXOBJ");
    std::vector<float> R(size_t(kObjRows) * 4, 0.0f);
    auto row = [&](int i) { return &R[size_t(i) * 4]; };
    auto lights = nee_lights(objs, flat);
    row(0)[0] = float(objs.size());
    row(0)[1] = float(lights.size());
    for (size_t i = 0; i < lights.size(); ++i) R[4 + i] = float(lights[i]);
    for (int k = 0; k < int(objs.size()); ++k) {
        const EdObject& o = objs[k];
        int b = 9 + 6 * k;
        float bound = 1e30f;
        if (o.geom == 0) {
            std::copy(o.c, o.c + 4, row(b));
            row(b + 1)[0] = 0; row(b + 1)[1] = o.a;
            bound = o.a;
        } else if (o.geom == 1) {
            std::copy(o.c, o.c + 4, row(b));
            row(b + 1)[0] = 1; row(b + 1)[1] = o.a; row(b + 1)[2] = o.b;
            bound = float(double(o.a) + double(o.b));
        } else {
            float pl[4] = {2, float(o.axis), o.offset, o.sign};
            std::copy(pl, pl + 4, row(b + 1));
            pl[0] = 1;
            std::copy(pl, pl + 4, row(b + 2));
        }
        std::copy(o.albedo, o.albedo + 3, row(b + 3));
        row(b + 3)[3] = float(o.mat);
        std::copy(o.emit, o.emit + 3, row(b + 4));
        row(b + 4)[3] = o.rough;
        row(b + 5)[0] = bound;
        row(b + 5)[1] = o.on_M ? 1.0f : 0.0f;
    }
    return R;
}

std::vector<EdObject> unpack_rows(const float* R, const std::vector<std::string>& names) {
    int n = int(R[0]);
    std::vector<EdObject> objs(n);
    for (int k = 0; k < n; ++k) {
        const float* r = R + size_t(9 + 6 * k) * 4;
        EdObject& o = objs[k];
        o.name = k < int(names.size()) ? names[k] : "obj" + std::to_string(k);
        o.geom = int(r[4]);
        std::copy(r, r + 4, o.c);
        if (o.geom == 2) { o.axis = int(r[5]); o.offset = r[6]; o.sign = r[7]; }
        else { o.a = r[5]; o.b = r[6]; }
        std::copy(r + 12, r + 15, o.albedo);
        o.mat = int(r[15]);
        std::copy(r + 16, r + 19, o.emit);
        o.rough = r[19];
        o.on_M = r[21] != 0;
    }
    return objs;
}

// ---- конфигурация из директив графа ------------------------------------------------------------------

EditorConfig parse_editor_config(const std::vector<std::vector<std::string>>& extra) {
    EditorConfig c;
    for (auto& w : extra) {
        if (w[0] == "editor") {               // editor <сцена.json> <рабочий каталог> <fov>
            c.enabled = true;
            c.scene_path = w.at(1);
            c.work_dir = w.at(2);
            c.fov = std::stof(w.at(3));
        } else if (w[0] == "manifold") {      // manifold <вид> <env> k=v k=v1,v2,v3 …
            c.kind = w.at(1);
            c.env = w.at(2) == "-" ? "" : w.at(2);
            for (size_t i = 3; i < w.size(); ++i) {
                auto eq = w[i].find('=');
                ManifoldParam p{w[i].substr(0, eq), {}};
                std::stringstream ss(w[i].substr(eq + 1));
                for (std::string t; std::getline(ss, t, ',');) p.v.push_back(std::stof(t));
                c.params.push_back(p);
            }
        } else if (w[0] == "flat") {          // flat <n листов> w… <n шаров> ρ₀ cx cy cz …
            size_t i = 1;
            c.flat.has = true;
            int ns = std::stoi(w.at(i++));
            for (int k = 0; k < ns; ++k) c.flat.sheets.push_back(std::stof(w.at(i++)));
            int nb = std::stoi(w.at(i++));
            c.flat.rho0 = std::stof(w.at(i++));
            for (int k = 0; k < nb; ++k) {
                std::array<float, 3> b{};
                for (auto& v : b) v = std::stof(w.at(i++));
                c.flat.balls.push_back(b);
            }
        } else if (w[0] == "objname") {       // objname <k> <имя>
            size_t k = std::stoul(w.at(1));
            if (c.names.size() <= k) c.names.resize(k + 1);
            c.names[k] = w.size() > 2 ? w[2] : "";
        } else if (w[0] == "rebuild") {       // rebuild <python> <скрипт> <аргументы…>
            c.python = w.at(1);
            c.script = w.at(2);
            c.rebuild_args.assign(w.begin() + 3, w.end());
        } else if (w[0] == "speed") {
            c.speed = std::stof(w.at(1));
        }
    }
    return c;
}

// ---- JSON (geo4d.scenefile) ---------------------------------------------------------------------------

std::string scene_json(const std::string& kind, const std::vector<ManifoldParam>& params, const std::string& env,
                       const std::vector<EdObject>& objs, const CameraState& cam, float fov, float speed) {
    static const char* geom[] = {"ball4", "torus", "plane"};
    static const char* mats[] = {"diffuse", "metal", "emit"};
    std::ostringstream o;
    o << "{\n \"manifold\": {\n  \"kind\": " << json_str(kind) << ",\n  \"params\": {";
    for (size_t i = 0; i < params.size(); ++i)
        o << (i ? "," : "") << "\n   " << json_str(params[i].key) << ": "
          << (params[i].v.size() == 1 ? num(params[i].v[0]) : vec(params[i].v.data(), int(params[i].v.size())));
    o << (params.empty() ? "}" : "\n  }") << "\n },\n";
    if (!env.empty()) o << " \"env\": " << json_str(env) << ",\n";
    o << " \"objects\": [";
    for (size_t k = 0; k < objs.size(); ++k) {
        const EdObject& b = objs[k];
        o << (k ? "," : "") << "\n  {\n   \"name\": " << json_str(b.name) << ",\n   \"kind\": " << json_str(geom[b.geom]) << ",\n";
        if (b.geom == 2)
            o << "   \"axis\": " << b.axis << ",\n   \"offset\": " << num(b.offset) << ",\n   \"sign\": " << num(b.sign) << ",\n";
        else {
            o << "   \"center\": " << vec(b.c, 4) << ",\n";
            if (b.geom == 0) o << "   \"r\": " << num(b.a) << ",\n";
            else o << "   \"R\": " << num(b.a) << ",\n   \"r\": " << num(b.b) << ",\n";
        }
        o << "   \"on_M\": " << (b.on_M ? "true" : "false") << ",\n   \"material\": {\n    \"kind\": " << json_str(mats[b.mat])
          << ",\n    \"albedo\": " << vec(b.albedo, 3) << ",\n    \"rough\": " << num(b.rough) << ",\n    \"emit\": "
          << vec(b.emit, 3) << "\n   }\n  }";
    }
    o << (objs.empty() ? "]" : "\n ]") << ",\n \"camera\": {\n  \"x\": " << vec(cam.x, 4) << ",\n  \"f\": " << vec(cam.f, 4)
      << ",\n  \"up\": " << vec(cam.u, 4) << ",\n  \"right\": " << vec(cam.r, 4) << ",\n  \"fov\": " << num(fov)
      << "\n },\n \"speed\": " << num(speed) << "\n}\n";
    return o.str();
}

// ---- редактор -------------------------------------------------------------------------------------------

Editor::Editor(const EditorConfig& cfg) : cfg_(cfg) {
    std::snprintf(save_path_, sizeof(save_path_), "%s", cfg.scene_path.c_str());
    kind_edit_ = cfg.kind;
    params_edit_ = cfg.params;
}

Editor::~Editor() {
    if (worker_ && worker_->joinable()) worker_->join();
}

void Editor::set_config(const EditorConfig& cfg, const float* rows) {
    std::string wd = cfg_.work_dir;                    // перегенерации — в каталоге первого графа (без вложенности)
    cfg_ = cfg;
    if (!wd.empty()) cfg_.work_dir = wd;
    kind_edit_ = cfg.kind;
    params_edit_ = cfg.params;
    objs_ = unpack_rows(rows, cfg.names);
    if (sel_ >= int(objs_.size()) || sel_ < 0) sel_ = objs_.empty() ? -1 : 0;
}

void Editor::save(const std::string& path, const CameraState& cam) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { status_ = "не удалось записать " + path; return; }
    f << scene_json(cfg_.kind, cfg_.params, cfg_.env, objs_, cam, cfg_.fov, cfg_.speed);
    status_ = "сохранено: " + path;
}

void Editor::start_rebuild(const CameraState& cam) {
    if (cfg_.python.empty()) { status_ = "перегенерация недоступна (нет директивы rebuild)"; return; }
    ++rebuild_n_;
    std::string base = cfg_.work_dir + "/edit_" + std::to_string(rebuild_n_);
    std::string json = base + ".json";
    graph_out_ = base;
    log_path_ = base + ".log";
    std::string env = kind_edit_ == cfg_.kind ? cfg_.env : "";        // другой вид — фон по умолчанию для вида
    {
        std::ofstream f(json, std::ios::binary);
        f << scene_json(kind_edit_, params_edit_, env, objs_, cam, cfg_.fov, cfg_.speed);
    }
    auto q = [](const std::string& s) { return "\"" + s + "\""; };
    std::string cmd = q(cfg_.python) + " " + q(cfg_.script) + " --scene " + q(json) + " --graph-only " + q(base);
    for (auto& a : cfg_.rebuild_args) cmd += " " + a;
    cmd += " > " + q(log_path_) + " 2>&1";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";                       // cmd /c снимает внешние кавычки
#endif
    done_ = false;
    status_ = "перегенерация: шейдеры и граф (Python)…";
    worker_ = std::make_unique<std::thread>([this, cmd]() {
        rc_ = std::system(cmd.c_str());
        done_ = true;
    });
}

void Editor::rebuild(const std::string& kind, const std::vector<ManifoldParam>& params, const CameraState& cam, bool wait) {
    kind_edit_ = kind;
    params_edit_ = default_params(kind);
    for (auto& p : params)
        for (auto& q : params_edit_)
            if (q.key == p.key) q.v = p.v;
    start_rebuild(cam);
    while (wait && worker_ && !done_) std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

std::string Editor::poll_rebuild() {
    if (!worker_ || !done_) return "";
    worker_->join();
    worker_.reset();
    if (rc_ == 0) {
        status_ = "многообразие обновлено";
        return graph_out_ + "/app_graph.txt";
    }
    std::ifstream f(log_path_);
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto p = all.rfind("Error");
    if (p == std::string::npos) p = all.size() > 600 ? all.size() - 600 : 0;
    status_ = "ошибка перегенерации (сцена не изменена):\n" + all.substr(p, 600);
    return "";
}

UiActions Editor::draw(const FrameInfo& fi, const CameraState& cam, bool& use_den, float& exposure, float& speed) {
    UiActions act;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Сцена (F1 — скрыть панели)");
    ImGui::Text("%.1f мс (%.0f к/с), накоплено %u spp", fi.ms, fi.ms > 0 ? 1000.0 / fi.ms : 0.0, fi.spp);
    ImGui::TextDisabled("x = (%.2f, %.2f, %.2f, %.2f)", cam.x[0], cam.x[1], cam.x[2], cam.x[3]);

    if (ImGui::CollapsingHeader("Рендер", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (fi.has_den) ImGui::Checkbox("денойзер (R)", &use_den);
        else ImGui::TextDisabled("денойзер недоступен");
        ImGui::SliderFloat("экспозиция", &exposure, 0.05f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("скорость", &speed, 0.05f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::TextDisabled("WASD, Q/E — полёт; ЛКМ — взгляд мышью; Z/C — крен");
    }

    const bool busy = rebuilding();
    if (ImGui::CollapsingHeader("Многообразие", ImGuiTreeNodeFlags_DefaultOpen)) {
        int cur = 0;
        for (int i = 0; i < IM_ARRAYSIZE(kKinds); ++i) if (kind_edit_ == kKinds[i]) cur = i;
        ImGui::BeginDisabled(busy);
        if (ImGui::Combo("вид", &cur, kKindNames, IM_ARRAYSIZE(kKindNames)) && kind_edit_ != kKinds[cur]) {
            kind_edit_ = kKinds[cur];
            params_edit_ = kind_edit_ == cfg_.kind ? cfg_.params : default_params(kind_edit_);
        }
        for (auto& p : params_edit_) {
            if (p.v.size() == 3) ImGui::DragFloat3(p.key.c_str(), p.v.data(), 0.02f);
            else ImGui::DragFloat(p.key.c_str(), p.v.data(), 0.01f, 0.0f, 0.0f, "%.3f");
        }
        if (ImGui::Button("Применить (перекомпиляция)")) start_rebuild(cam);
        ImGui::SameLine();
        if (ImGui::Button("Сброс")) { kind_edit_ = cfg_.kind; params_edit_ = cfg_.params; }
        ImGui::EndDisabled();
        if (busy) ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "идёт перегенерация…");
    }

    if (ImGui::CollapsingHeader("Файл", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputText("##path", save_path_, sizeof(save_path_));
        ImGui::BeginDisabled(busy);
        if (ImGui::Button("Сохранить сцену")) save(save_path_, cam);
        ImGui::EndDisabled();
    }
    if (!status_.empty()) ImGui::TextWrapped("%s", status_.c_str());
    ImGui::End();

    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(disp.x - 370, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Объекты");
    {
        ImGui::BeginDisabled(busy);
        auto lights = nee_lights(objs_, cfg_.flat);
        auto is_light = [&](int k) { return std::find(lights.begin(), lights.end(), k) != lights.end(); };
        if (ImGui::BeginListBox("##objs", ImVec2(-FLT_MIN, 6.5f * ImGui::GetTextLineHeightWithSpacing()))) {
            for (int k = 0; k < int(objs_.size()); ++k) {
                std::string label = objs_[k].name + (is_light(k) ? "  [NEE]" : "") + "##" + std::to_string(k);
                if (ImGui::Selectable(label.c_str(), sel_ == k)) sel_ = k;
            }
            ImGui::EndListBox();
        }
        auto add = [&](int geom) {
            if (objs_.size() >= size_t(kMaxObj)) { status_ = "не больше 32 объектов"; return; }
            EdObject o;
            o.geom = geom;
            o.name = geom == 0 ? "ball" : geom == 1 ? "torus" : "plane";
            if (geom == 2) { o.axis = 2; o.offset = cam.x[2] - 1.0f; }
            else {
                float d = geom == 0 ? 2.5f : 4.0f;
                for (int i = 0; i < 4; ++i) o.c[i] = cam.x[i] + d * cam.f[i];
                o.on_M = true;
                if (geom == 1) { o.a = 0.6f; o.b = 0.15f; }
                act.snap = true;
            }
            objs_.push_back(o);
            sel_ = int(objs_.size()) - 1;
            act.objects_changed = true;
        };
        if (ImGui::Button("+ шар")) add(0);
        ImGui::SameLine();
        if (ImGui::Button("+ тор")) add(1);
        ImGui::SameLine();
        if (ImGui::Button("+ плоскость")) add(2);
        if (sel_ >= 0) {
            if (ImGui::Button("копия") && objs_.size() < size_t(kMaxObj)) {
                EdObject o = objs_[sel_];
                o.name += "_copy";
                if (o.geom != 2) o.c[0] += 2.2f * o.a;
                objs_.push_back(o);
                sel_ = int(objs_.size()) - 1;
                act.objects_changed = true;
                act.snap = o.on_M;
            }
            ImGui::SameLine();
            if (ImGui::Button("удалить")) {
                objs_.erase(objs_.begin() + sel_);
                sel_ = -1;
                act.objects_changed = true;
            }
        }
        if (sel_ >= 0 && sel_ < int(objs_.size())) {
            EdObject& o = objs_[sel_];
            bool ch = false, moved = false;
            ImGui::SeparatorText("свойства");
            char name[64];
            std::snprintf(name, sizeof(name), "%s", o.name.c_str());
            if (ImGui::InputText("имя", name, sizeof(name))) {
                o.name = name;
                std::replace(o.name.begin(), o.name.end(), ' ', '_');   // имена — слова директив графа
            }
            const char* geoms[] = {"4D-шар", "тор в листе", "полупространство"};
            if (ImGui::Combo("форма", &o.geom, geoms, 3)) {
                ch = true;
                if (o.geom == 1 && o.b <= 0) o.b = 0.15f;
            }
            if (o.geom != 2) {
                moved |= ImGui::DragFloat4("центр", o.c, 0.01f);
                if (ImGui::Checkbox("на M (проекция центра)", &o.on_M)) moved = true;
                if (ImGui::Button("перед камерой")) {
                    float d = std::max(2.0f, 4.0f * o.a);
                    for (int i = 0; i < 4; ++i) o.c[i] = cam.x[i] + d * cam.f[i];
                    o.on_M = true;
                    moved = true;
                }
                if (cfg_.flat.has) {
                    ImGui::SameLine();
                    if (ImGui::Button("на другой лист")) { o.c[3] = -o.c[3]; moved = true; }
                }
                if (o.geom == 0) ch |= ImGui::DragFloat("r", &o.a, 0.005f, 0.01f, 100.0f);
                else {
                    ch |= ImGui::DragFloat("R", &o.a, 0.005f, 0.01f, 100.0f);
                    ch |= ImGui::DragFloat("r##t", &o.b, 0.005f, 0.005f, 100.0f);
                }
            } else {
                const char* axes[] = {"x", "y", "z", "w"};
                ch |= ImGui::Combo("ось", &o.axis, axes, 4);
                ch |= ImGui::DragFloat("смещение", &o.offset, 0.01f);
                bool up = o.sign > 0;
                if (ImGui::Checkbox("внутрь — в сторону «−»", &up)) { o.sign = up ? 1.0f : -1.0f; ch = true; }
            }
            const char* mats[] = {"диффузный", "металл", "излучатель"};
            ch |= ImGui::Combo("материал", &o.mat, mats, 3);
            if (o.mat != 2) ch |= ImGui::ColorEdit3("альбедо", o.albedo);
            if (o.mat == 1) ch |= ImGui::SliderFloat("шероховатость", &o.rough, 0.01f, 1.0f);
            if (o.mat == 2) {
                float s = max3(o.emit), col[3] = {1, 1, 1};
                if (s > 0) for (int i = 0; i < 3; ++i) col[i] = o.emit[i] / s;
                bool ec = ImGui::ColorEdit3("цвет", col);
                ec |= ImGui::DragFloat("яркость", &s, 0.1f, 0.0f, 1000.0f);
                if (ec) {
                    for (int i = 0; i < 3; ++i) o.emit[i] = col[i] * s;
                    ch = true;
                }
                if (is_light(sel_)) ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "NEE-источник (M34)");
                else ImGui::TextDisabled(cfg_.flat.has ? "не в плоской зоне / не на листе: только BSDF (M34)"
                                                       : "плоских зон нет: свет только через BSDF");
            }
            if (ch || moved) act.objects_changed = true;
            if (moved && o.on_M && o.geom != 2) act.snap = true;
        }
        ImGui::EndDisabled();
    }
    ImGui::End();
    return act;
}
