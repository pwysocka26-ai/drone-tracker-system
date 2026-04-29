// CSV-based gimbal data source dla integracji z encoderami glowicy.
//
// Format CSV (header obowiazkowy, kolejnosc kolumn dowolna):
//   frame_idx,fov_h_deg,axis_az_mrad,axis_el_mrad
//   0,8.0,1500.0,200.0
//   1,8.0,1502.5,200.1
//   ...
//
// Lookup strategy: exact match po frame_idx; brak wpisu dla danej klatki
// -> uzyj nearest-PREVIOUS valid entry (gimbal moze raportowac rzadziej niz
// FPS video). Brak wpisow przed pierwsza klatka video -> nullopt.
//
// Header-only (CSV parser ~50 LOC, brak zewnetrznych deps).
#pragma once

#include <algorithm>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "dtracker/angular.hpp"

namespace dtracker {

class GimbalCsvSource {
public:
    // Load CSV z pliku. Throws std::runtime_error przy brakach kolumn / parse errors.
    explicit GimbalCsvSource(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("GimbalCsvSource: cannot open " + path);
        }
        std::string line;
        if (!std::getline(f, line)) {
            throw std::runtime_error("GimbalCsvSource: empty file " + path);
        }
        // Parse header — column index lookup
        std::unordered_map<std::string, int> col;
        {
            std::stringstream ss(line);
            std::string cell;
            int idx = 0;
            while (std::getline(ss, cell, ',')) {
                // trim whitespace
                while (!cell.empty() && (cell.back() == '\r' || cell.back() == ' ')) cell.pop_back();
                while (!cell.empty() && cell.front() == ' ') cell.erase(0, 1);
                col[cell] = idx++;
            }
        }
        const auto require = [&](const char* name) -> int {
            auto it = col.find(name);
            if (it == col.end()) {
                throw std::runtime_error(std::string("GimbalCsvSource: missing column ") + name);
            }
            return it->second;
        };
        const int c_frame = require("frame_idx");
        const int c_fov = require("fov_h_deg");
        const int c_az = require("axis_az_mrad");
        const int c_el = require("axis_el_mrad");

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<std::string> cells;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) cells.push_back(cell);
            const int max_idx = std::max({c_frame, c_fov, c_az, c_el});
            if (static_cast<int>(cells.size()) <= max_idx) continue;  // skip short row
            Entry e;
            e.frame_idx = std::stoi(cells[c_frame]);
            e.fov_h_deg = std::stof(cells[c_fov]);
            e.axis_az_mrad = std::stof(cells[c_az]);
            e.axis_el_mrad = std::stof(cells[c_el]);
            entries_.push_back(e);
        }
        std::sort(entries_.begin(), entries_.end(),
                  [](const Entry& a, const Entry& b) { return a.frame_idx < b.frame_idx; });
    }

    // Zwraca GimbalSnapshot dla danej klatki video.
    // - Exact match po frame_idx -> ten wpis.
    // - Brak match -> nearest-PREVIOUS valid entry.
    // - Brak wpisow przed frame_idx (np. CSV zaczyna sie od frame 100, pytanie o frame 50)
    //   -> nullopt.
    // Image_w/h potrzebne tylko jesli chcesz fov_v_rad wyliczone z aspect.
    std::optional<GimbalSnapshot> lookup(int frame_idx, int image_w, int image_h) const {
        if (entries_.empty()) return std::nullopt;
        // upper_bound: pierwszy entry > frame_idx
        auto it = std::upper_bound(entries_.begin(), entries_.end(), frame_idx,
                                    [](int v, const Entry& e) { return v < e.frame_idx; });
        if (it == entries_.begin()) return std::nullopt;  // wszystkie wpisy po frame_idx
        --it;  // nearest-previous (lub exact match)
        GimbalSnapshot g;
        g.fov_h_rad = deg_to_rad(it->fov_h_deg);
        g.fov_v_rad = fov_v_from_h(g.fov_h_rad, image_w, image_h);
        g.axis_az_mrad = it->axis_az_mrad;
        g.axis_el_mrad = it->axis_el_mrad;
        return g;
    }

    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }

private:
    struct Entry {
        int frame_idx;
        float fov_h_deg;
        float axis_az_mrad;
        float axis_el_mrad;
    };
    std::vector<Entry> entries_;
};

}  // namespace dtracker
