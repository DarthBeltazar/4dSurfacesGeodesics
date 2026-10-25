// Файлы графа в памяти (DD-010, шаг 3) — см. memfs.hpp.
#include "memfs.hpp"

#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>

namespace memfs {

namespace {

std::mutex mtx;
std::map<std::string, std::string> texts;
std::map<std::string, NpyArray> arrays;

}  // namespace

void put_text(const std::string& path, const std::string& text) {
    std::lock_guard<std::mutex> lock(mtx);
    texts[path] = text;
}

void put_npy(const std::string& path, NpyArray a) {
    std::lock_guard<std::mutex> lock(mtx);
    arrays[path] = std::move(a);
}

std::string read_text(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = texts.find(path);
        if (it != texts.end()) return it->second;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("нет файла: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

NpyArray read_npy(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = arrays.find(path);
        if (it != arrays.end()) return it->second;
    }
    return npy_read(path);
}

}  // namespace memfs
