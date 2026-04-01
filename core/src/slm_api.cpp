#define SLM_EXPORTS
#include "slm/slm_api.h"
#include "slm/tokenizer.h"
#include "slm/ngram_model.h"
#include "slm/generator.h"

#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>

struct SLMContext {
    slm::NGramModel model;
};

static char* alloc_string(const std::string& s) {
    char* buf = static_cast<char*>(std::malloc(s.size() + 1));
    if (buf) {
        std::memcpy(buf, s.c_str(), s.size() + 1);
    }
    return buf;
}

extern "C" {

slm_handle slm_create() {
    return new SLMContext();
}

void slm_destroy(slm_handle handle) {
    delete static_cast<SLMContext*>(handle);
}

int slm_train_file(slm_handle handle, const char* filepath) {
    if (!handle || !filepath) return -1;

    auto* ctx = static_cast<SLMContext*>(handle);

    std::ifstream file(filepath);
    if (!file) return -1;

    std::stringstream ss;
    ss << file.rdbuf();
    std::string text = ss.str();

    auto tokens = slm::Tokenizer::tokenize(text);
    if (tokens.empty()) return -1;

    ctx->model.train(tokens);
    return 0;
}

int slm_train_text(slm_handle handle, const char* text) {
    if (!handle || !text) return -1;

    auto* ctx = static_cast<SLMContext*>(handle);
    auto tokens = slm::Tokenizer::tokenize(std::string(text));
    if (tokens.empty()) return -1;

    ctx->model.train(tokens);
    return 0;
}

int slm_save_model(slm_handle handle, const char* path) {
    if (!handle || !path) return -1;
    auto* ctx = static_cast<SLMContext*>(handle);
    return ctx->model.save_model(std::string(path)) ? 0 : -1;
}

int slm_load_model(slm_handle handle, const char* path) {
    if (!handle || !path) return -1;
    auto* ctx = static_cast<SLMContext*>(handle);
    return ctx->model.load_model(std::string(path)) ? 0 : -1;
}

const char* slm_generate(slm_handle handle, const char* prompt, int max_len, double temperature) {
    if (!handle || !prompt) return alloc_string("");

    auto* ctx = static_cast<SLMContext*>(handle);
    auto prompt_tokens = slm::Tokenizer::tokenize(std::string(prompt));

    slm::Generator gen(ctx->model);
    auto result_tokens = gen.generate(prompt_tokens,
        max_len > 0 ? static_cast<size_t>(max_len) : 50,
        temperature > 0 ? temperature : 0.7);

    std::string result = slm::Tokenizer::detokenize(result_tokens);
    return alloc_string(result);
}

const char* slm_predict_next(slm_handle handle, const char* context, int num_candidates) {
    if (!handle || !context) return alloc_string("[]");

    auto* ctx = static_cast<SLMContext*>(handle);
    auto tokens = slm::Tokenizer::tokenize(std::string(context));

    auto predictions = ctx->model.predict_next(tokens,
        num_candidates > 0 ? static_cast<size_t>(num_candidates) : 5);

    // Build JSON array
    std::string json = "[";
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (i > 0) json += ",";
        json += "{\"word\":\"" + predictions[i].first + "\",\"prob\":"
             + std::to_string(predictions[i].second) + "}";
    }
    json += "]";

    return alloc_string(json);
}

int slm_get_vocab_size(slm_handle handle) {
    if (!handle) return 0;
    auto* ctx = static_cast<SLMContext*>(handle);
    return static_cast<int>(ctx->model.vocab_size());
}

int slm_get_total_tokens(slm_handle handle) {
    if (!handle) return 0;
    auto* ctx = static_cast<SLMContext*>(handle);
    return static_cast<int>(ctx->model.total_tokens_trained());
}

const char* slm_get_top_vocab(slm_handle handle, int n) {
    if (!handle) return alloc_string("[]");

    auto* ctx = static_cast<SLMContext*>(handle);
    auto top = ctx->model.top_vocab(n > 0 ? static_cast<size_t>(n) : 20);

    // Build JSON array
    std::string json = "[";
    for (size_t i = 0; i < top.size(); ++i) {
        if (i > 0) json += ",";
        json += "{\"word\":\"" + top[i].first + "\",\"count\":"
             + std::to_string(top[i].second) + "}";
    }
    json += "]";

    return alloc_string(json);
}

void slm_free_string(const char* str) {
    std::free(const_cast<char*>(str));
}

void slm_reset(slm_handle handle) {
    if (!handle) return;
    auto* ctx = static_cast<SLMContext*>(handle);
    ctx->model.reset();
}

} // extern "C"
