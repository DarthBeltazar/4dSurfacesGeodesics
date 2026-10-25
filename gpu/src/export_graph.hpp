// geo4d_gpu export-graph (DD-010, шаг 1) — см. export_graph.cpp.
#pragma once

int run_export_graph(int argc, char** argv);
// geo4d_gpu eval-graph <сцена> <вход.json> <выход.json>: многообразие графа на CPU (f, ∇f, H в точках), метаданные,
// mf_scene.slang и каталог заголовка (для сверки GPU с прототипом, DD-011 шаг 3)
int run_eval_graph(int argc, char** argv);
