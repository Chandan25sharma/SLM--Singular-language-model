#pragma once

#include "slm/ngram_model.h"
#include <string>
#include <vector>
#include <random>

namespace slm {

class Generator {
public:
    explicit Generator(const NGramModel& model);

    // Generate continuation from prompt tokens
    std::vector<std::string> generate(
        const std::vector<std::string>& prompt_tokens,
        size_t max_length = 50,
        double temperature = 0.7) const;

private:
    const NGramModel& model_;
    mutable std::mt19937 rng_;

    // Weighted random selection with temperature
    std::string sample(
        const std::vector<std::pair<std::string, double>>& candidates,
        double temperature) const;
};

} // namespace slmsdvdfcb
