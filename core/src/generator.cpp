#include "slm/generator.h"
#include <cmath>
#include <numeric>
#include <chrono>

namespace slm {

Generator::Generator(const NGramModel& model)
    : model_(model)
    , rng_(static_cast<unsigned>(
          std::chrono::steady_clock::now().time_since_epoch().count()))
{
}

std::string Generator::sample(
    const std::vector<std::pair<std::string, double>>& candidates,
    double temperature) const {

    if (candidates.empty()) return "";
    if (candidates.size() == 1) return candidates[0].first;

    // Apply temperature scaling
    std::vector<double> weights;
    weights.reserve(candidates.size());

    for (const auto& [word, prob] : candidates) {
        double adjusted = std::pow(prob, 1.0 / temperature);
        weights.push_back(adjusted);
    }

    // Normalize
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sum <= 0) return candidates[0].first;

    for (auto& w : weights) {
        w /= sum;
    }

    // Weighted random selection
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    double cumulative = 0.0;

    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (r <= cumulative) {
            return candidates[i].first;
        }
    }

    return candidates.back().first;
}

std::vector<std::string> Generator::generate(
    const std::vector<std::string>& prompt_tokens,
    size_t max_length,
    double temperature) const {

    std::vector<std::string> result = prompt_tokens;
    
    // End-of-sentence markers
    const std::vector<std::string> eos_hints = {".", "!", "?"};

    for (size_t i = 0; i < max_length; ++i) {
        // Build context from last 2 tokens
        std::vector<std::string> context;
        if (result.size() >= 2) {
            context.push_back(result[result.size() - 2]);
            context.push_back(result[result.size() - 1]);
        } else if (!result.empty()) {
            context.push_back(result.back());
        }

        auto candidates = model_.predict_next(context, 10);
        if (candidates.empty()) break;

        std::string next_word = sample(candidates, temperature);
        if (next_word.empty()) break;

        result.push_back(next_word);

        // Natural stopping: if we generated a reasonable amount and hit end-of-sentence territory
        if (i > 5) {
            // Check if recent word looks like a sentence ender
            // (in our tokenizer punctuation is stripped, so we rely on common sentence-ending patterns)
            bool should_stop = false;
            
            // Random chance to stop after minimum length, increasing with length
            if (i > 15) {
                std::uniform_real_distribution<double> stop_dist(0.0, 1.0);
                double stop_chance = (static_cast<double>(i) - 15.0) / (max_length * 2.0);
                if (stop_dist(rng_) < stop_chance) {
                    should_stop = true;
                }
            }
            
            if (should_stop) break;
        }
    }

    return result;
}

} // namespace slm
