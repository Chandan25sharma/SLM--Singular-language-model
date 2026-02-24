#include "slm/tokenizer.h"
#include <algorithm>
#include <sstream>
#include <cctype>

namespace slm {

std::string Tokenizer::to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string Tokenizer::strip_punct(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') {
            result += c;
        }
    }
    return result;
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        word = to_lower(word);
        word = strip_punct(word);
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }

    return tokens;
}

std::string Tokenizer::detokenize(const std::vector<std::string>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += ' ';
        result += tokens[i];
    }
    return result;
}

} // namespace slm
