#include "slm/frequency_store.h"
#include <fstream>
#include <algorithm>
#include <numeric>

namespace slm {

void FrequencyStore::increment(const std::string& key, uint64_t count) {
    store_[key] += count;
}

uint64_t FrequencyStore::get_count(const std::string& key) const {
    auto it = store_.find(key);
    return it != store_.end() ? it->second : 0;
}

std::vector<std::pair<std::string, uint64_t>> FrequencyStore::get_top_k(
    const std::string& prefix, size_t k) const {

    std::vector<std::pair<std::string, uint64_t>> matches;

    for (const auto& [key, count] : store_) {
        if (key.size() >= prefix.size() &&
            key.compare(0, prefix.size(), prefix) == 0) {
            matches.emplace_back(key, count);
        }
    }

    // Partial sort to get top-k
    if (matches.size() > k) {
        std::partial_sort(matches.begin(), matches.begin() + k, matches.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        matches.resize(k);
    } else {
        std::sort(matches.begin(), matches.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    return matches;
}

const std::unordered_map<std::string, uint64_t>& FrequencyStore::entries() const {
    return store_;
}

size_t FrequencyStore::size() const {
    return store_.size();
}

uint64_t FrequencyStore::total_count() const {
    uint64_t total = 0;
    for (const auto& [key, count] : store_) {
        total += count;
    }
    return total;
}

void FrequencyStore::clear() {
    store_.clear();
}

bool FrequencyStore::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return false;

    // Write magic bytes "SFS1"
    out.write("SFS1", 4);

    // Write number of entries
    uint64_t count = store_.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Write each entry: key_len, key, value
    for (const auto& [key, value] : store_) {
        uint32_t key_len = static_cast<uint32_t>(key.size());
        out.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        out.write(key.data(), key_len);
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    return out.good();
}

bool FrequencyStore::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;

    // Check magic bytes
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "SFS1") return false;

    // Read number of entries
    uint64_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    store_.clear();
    store_.reserve(static_cast<size_t>(count));

    // Read entries
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t key_len;
        in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));

        std::string key(key_len, '\0');
        in.read(&key[0], key_len);

        uint64_t value;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));

        store_[key] = value;
    }

    return in.good();
}

} // namespace slm
