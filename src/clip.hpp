#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace pentimento {
	inline constexpr int kEmbeddingSize = 512;

#ifdef _WIN32
    inline std::wstring to_ort_path(const std::string& s) {
        return std::wstring(s.begin(), s.end());
    } 
#else
    inline std::string to_ort_path(const std::string& s) {
        return s;
    }
#endif

    class ClipEncoder {
    public: 
		explicit ClipEncoder(const std::string& model_path, int threads = 0);
		void encode(const float* input, std::size_t count, std::vector<float>& output);
    private:
		Ort::Env env_;
		Ort::SessionOptions options_;
		Ort::Session session_;
    };

}