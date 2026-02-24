#include "slm/ngram_model.h"
#include <fstream>
#include <algorithm>
#include <cmath>

namespace slm {

// Key format: "word1|word2" for bigrams, "word1|word2|word3" for trigrams
std::string NGramModel::bigram_key(const std::string& w1, const std::string& w2) {
    return w1 + "|" + w2;
}

std::string NGramModel::trigram_key(const std::string& w1, const std::string& w2, const std::string& w3) {
    return w1 + "|" + w2 + "|" + w3;
}

std::string NGramModel::bigram_prefix(const std::string& w1) {
    return w1 + "|";
}

std::string NGramModel::trigram_prefix(const std::string& w1, const std::string& w2) {
    return w1 + "|" + w2 + "|";
}

std::string NGramModel::extract_last_word(const std::string& key) {
    auto pos = key.rfind('|');
    if (pos == std::string::npos) return key;
    return key.substr(pos + 1);
}

void NGramModel::train(const std::vector<std::string>& tokens) {
    if (tokens.empty()) return;

    total_tokens_ += tokens.size();

    // Unigrams
    for (const auto& token : tokens) {
        unigrams_.increment(token);
    }

    // Bigrams
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        bigrams_.increment(bigram_key(tokens[i], tokens[i + 1]));
    }

    // Trigrams
    for (size_t i = 0; i + 2 < tokens.size(); ++i) {
        trigrams_.increment(trigram_key(tokens[i], tokens[i + 1], tokens[i + 2]));
    }
}

std::vector<std::pair<std::string, double>> NGramModel::predict_next(
    const std::vector<std::string>& context, size_t num_candidates) const {

    std::vector<std::pair<std::string, double>> results;

    // Try trigram first (if we have 2+ context words)
    if (context.size() >= 2) {
        const auto& w1 = context[context.size() - 2];
        const auto& w2 = context[context.size() - 1];
        auto prefix = trigram_prefix(w1, w2);
        auto top = trigrams_.get_top_k(prefix, num_candidates * 2);

        if (!top.empty()) {
            // Compute context count (bigram of w1,w2)
            double context_count = static_cast<double>(bigrams_.get_count(bigram_key(w1, w2)));
            if (context_count > 0) {
                for (const auto& [key, count] : top) {
                    std::string word = extract_last_word(key);
                    double prob = static_cast<double>(count) / context_count;
                    results.emplace_back(word, prob);
                }
            }
        }
    }

    // Backoff to bigram (if we have 1+ context words)
    if (results.empty() && !context.empty()) {
        const auto& w1 = context.back();
        auto prefix = bigram_prefix(w1);
        auto top = bigrams_.get_top_k(prefix, num_candidates * 2);

        if (!top.empty()) {
            double context_count = static_cast<double>(unigrams_.get_count(w1));
            if (context_count > 0) {
                for (const auto& [key, count] : top) {
                    std::string word = extract_last_word(key);
                    double prob = static_cast<double>(count) / context_count;
                    results.emplace_back(word, prob);
                }
            }
        }
    }

    // Backoff to unigram
    if (results.empty()) {
        auto top = unigrams_.get_top_k("", num_candidates * 2);
        double total = static_cast<double>(unigrams_.total_count());
        if (total > 0) {
            for (const auto& [key, count] : top) {
                double prob = static_cast<double>(count) / total;
                results.emplace_back(key, prob);
            }
        }
    }

    // Sort by probability and trim
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    if (results.size() > num_candidates) {
        results.resize(num_candidates);
    }

    return results;
}

bool NGramModel::save_model(const std::string& path) const {
    // Save as three files: path.uni, path.bi, path.tri
    if (!unigrams_.save(path + ".uni")) return false;
    if (!bigrams_.save(path + ".bi")) return false;
    if (!trigrams_.save(path + ".tri")) return false;

    // Save metadata
    std::ofstream meta(path + ".meta", std::ios::binary);
    if (!meta) return false;
    meta.write("SLM1", 4);
    meta.write(reinterpret_cast<const char*>(&total_tokens_), sizeof(total_tokens_));

    return meta.good();
}

bool NGramModel::load_model(const std::string& path) {
    if (!unigrams_.load(path + ".uni")) return false;
    if (!bigrams_.load(path + ".bi")) return false;
    if (!trigrams_.load(path + ".tri")) return false;

    std::ifstream meta(path + ".meta", std::ios::binary);
    if (!meta) return false;
    char magic[4];
    meta.read(magic, 4);
    if (std::string(magic, 4) != "SLM1") return false;
    meta.read(reinterpret_cast<char*>(&total_tokens_), sizeof(total_tokens_));

    return meta.good();
}

size_t NGramModel::vocab_size() const {
    return unigrams_.size();
}

size_t NGramModel::unigram_count() const {
    return unigrams_.size();
}

size_t NGramModel::bigram_count() const {
    return bigrams_.size();
}

size_t NGramModel::trigram_count() const {
    return trigrams_.size();
}

uint64_t NGramModel::total_tokens_trained() const {
    return total_tokens_;
}

std::vector<std::pair<std::string, uint64_t>> NGramModel::top_vocab(size_t n) const {
    return unigrams_.get_top_k("", n);
}

void NGramModel::reset() {
    unigrams_.clear();
    bigrams_.clear();
    trigrams_.clear();
    total_tokens_ = 0;
}

} // namespace slm
