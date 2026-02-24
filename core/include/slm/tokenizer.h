#pragma once

#include <string>
#include <vector>

namespace slm {

class Tokenizer {
public:
    // Tokenize text into lowercase words, stripping punctuation
    static std::vector<std::string> tokenize(const std::string& text);

    // Join tokens back into text
    static std::string detokenize(const std::vector<std::string>& tokens);

private:
    static std::string to_lower(const std::string& s);
    static std::string strip_punct(const std::string& s);
};

} // namespace slm
