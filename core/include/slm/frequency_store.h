#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <utility>

namespace slm {

class FrequencyStore {
public:
    FrequencyStore() = default;

    // Increment count for an n-gram key
    void increment(const std::string& key, uint64_t count = 1);

    // Get count for a key
    uint64_t get_count(const std::string& key) const;

    // Get top-k entries matching a prefix
    std::vector<std::pair<std::string, uint64_t>> get_top_k(
        const std::string& prefix, size_t k) const;

    // Get all entries (for serialization)
    const std::unordered_map<std::string, uint64_t>& entries() const;

    // Total unique entries
    size_t size() const;

    // Total count across all entries
    uint64_t total_count() const;

    // Clear all data
    void clear();

    // Binary serialization
    bool save(const std::string& filepath) const;
    bool load(const std::string& filepath);

private:
    std::unordered_map<std::string, uint64_t> store_;
};

} // namespace slm
