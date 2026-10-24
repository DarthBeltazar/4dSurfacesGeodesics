// Слова директив графа и job-файлов: через пробел; "…" — одно слово с пробелами (пути вида C:\My Programs\…).
// protect_quotes снимает кавычки и заменяет пробелы внутри них на \x1f — строку можно читать std::istringstream;
// прочитанные пути возвращает к исходному виду unprotect. Пишет кавычки geo4d.gpu.qpath.
#pragma once

#include <string>

inline std::string protect_quotes(const std::string& line) {
    std::string out;
    bool in_q = false;
    for (char ch : line) {
        if (ch == '"') { in_q = !in_q; continue; }
        out += (in_q && (ch == ' ' || ch == '\t')) ? '\x1f' : ch;
    }
    return out;
}

inline std::string unprotect(std::string s) {
    for (auto& ch : s)
        if (ch == '\x1f') ch = ' ';
    return s;
}
