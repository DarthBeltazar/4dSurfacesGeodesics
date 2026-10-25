#pragma once

#include <string>
#include <vector>

#include "editor.hpp"

// rebuild — перегенерация в процессе (окно без Python, DD-010): редактор вызывает её вместо app.py; args — как у app.py
int run_app(const std::string& graph_path, RebuildFn rebuild = {}, std::vector<std::string> rebuild_args = {});
