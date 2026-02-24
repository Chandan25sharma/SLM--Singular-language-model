#pragma once

#include "slm/frequency_store.h"
#include <string>
#include <vector>
#include <utility>

namespace slm {

class NGramModel {
public:
    NGramModel() = default;

    // Train on a sequence of tokens
    void train(const std::vector<std::string>& tokens);

    // Predict next word given context (last 1-2 tokens)
    // Returns vector of (word, probability) sorted by probability descending
    std::vector<std::pair<std::string, double>> predict_next(
        const std::vector<std::string>& context, size_t num_candidates = 5) const;

    // Save/load model
    bool save_model(const std::string& path) const;
    bool load_model(const std::string& path);

    // Stats
    size_t vocab_size() const;
    size_t unigram_count() const;
    size_t bigram_count() const;
    size_t trigram_count() const;
    uint64_t total_tokens_trained() const;

    // Get top vocabulary words
    std::vector<std::pair<std::string, uint64_t>> top_vocab(size_t n) const;

    // Clear all model data
    void reset();

private:
    FrequencyStore unigrams_;
    FrequencyStore bigrams_;
    FrequencyStore trigrams_;
    uint64_t total_tokens_ = 0;

    // N-gram key builders
    static std::string bigram_key(const std::string& w1, const std::string& w2);
    static std::string trigram_key(const std::string& w1, const std::string& w2, const std::string& w3);
    static std::string bigram_prefix(const std::string& w1);
    static std::string trigram_prefix(const std::string& w1, const std::string& w2);
    static std::string extract_last_word(const std::string& key);
};

} // namespace slm
