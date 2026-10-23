// Чтение/запись .npy (C-порядок, little-endian; float32, float64, int32, uint32).
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct NpyArray {
    std::string dtype;               // "<f4", "<f8", "<i4", "<u4"
    std::vector<size_t> shape;
    std::vector<uint8_t> data;

    size_t count() const {
        size_t n = 1;
        for (size_t s : shape) n *= s;
        return n;
    }
    size_t itemsize() const { return dtype == "<f8" ? 8 : 4; }
};

inline NpyArray npy_read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("не открыть " + path);
    char magic[6];
    f.read(magic, 6);
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        uint16_t h16;
        f.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, ' ');
    f.read(header.data(), hlen);
    NpyArray a;
    auto p = header.find("'descr':");
    auto q1 = header.find('\'', p + 8);
    auto q2 = header.find('\'', q1 + 1);
    a.dtype = header.substr(q1 + 1, q2 - q1 - 1);
    if (header.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("fortran_order не поддерживается");
    auto s1 = header.find('(', header.find("'shape':"));
    auto s2 = header.find(')', s1);
    std::string sh = header.substr(s1 + 1, s2 - s1 - 1);
    size_t pos = 0;
    while (pos < sh.size()) {
        while (pos < sh.size() && (sh[pos] == ' ' || sh[pos] == ',')) ++pos;
        if (pos >= sh.size()) break;
        size_t e = pos;
        while (e < sh.size() && isdigit(static_cast<unsigned char>(sh[e]))) ++e;
        a.shape.push_back(std::stoull(sh.substr(pos, e - pos)));
        pos = e;
    }
    if (a.dtype != "<f4" && a.dtype != "<f8" && a.dtype != "<i4" && a.dtype != "<u4")
        throw std::runtime_error("dtype не поддерживается: " + a.dtype);
    a.data.resize(a.count() * a.itemsize());
    f.read(reinterpret_cast<char*>(a.data.data()), a.data.size());
    return a;
}

inline void npy_write(const std::string& path, const std::string& dtype, const std::vector<size_t>& shape,
                      const void* data, size_t bytes) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
    sh += ")";
    std::string header = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t total = 10 + header.size() + 1;
    header += std::string((64 - total % 64) % 64, ' ') + "\n";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("не записать " + path);
    f.write("\x93NUMPY\x01\x00", 8);
    uint16_t hl = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(header.data(), header.size());
    f.write(static_cast<const char*>(data), bytes);
}
