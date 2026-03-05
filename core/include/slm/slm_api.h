#pragma once

#ifdef _WIN32
    #ifdef SLM_EXPORTS
        #define SLM_API __declspec(dllexport)
    #else
        #define SLM_API __declspec(dllimport)
    #endif
#else
    #define SLM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef void* slm_handle;

// Lifecycle
SLM_API slm_handle slm_create();
SLM_API void slm_destroy(slm_handle handle);

// Training
SLM_API int slm_train_file(slm_handle handle, const char* filepath);
SLM_API int slm_train_text(slm_handle handle, const char* text);

// Model persistence
SLM_API int slm_save_model(slm_handle handle, const char* path);
SLM_API int slm_load_model(slm_handle handle, const char* path);

// Generation
SLM_API const char* slm_generate(slm_handle handle, const char* prompt, int max_len, double temperature);
SLM_API const char* slm_predict_next(slm_handle handle, const char* context, int num_candidates);

// Info
SLM_API int slm_get_vocab_size(slm_handle handle);
SLM_API int slm_get_total_tokens(slm_handle handle);
SLM_API const char* slm_get_top_vocab(slm_handle handle, int n);

// Memory
SLM_API void slm_free_string(const char* str);

// Reset
SLM_API void slm_reset(slm_handle handle);

#ifdef __cplusplus
}
#endif
