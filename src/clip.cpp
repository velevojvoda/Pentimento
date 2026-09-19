#include "clip.hpp"

#include "image.hpp"

#include <array>
#include <cstdint>

namespace pentimento {

	namespace {
		constexpr const char* kInputName = "pixel_values";
		constexpr const char* kOutputName = "image_embeds";
	}
	
	ClipEncoder::ClipEncoder(const std::string& model_path, int threads)
		: env_(ORT_LOGGING_LEVEL_WARNING, "pentimento"), options_(), session_(nullptr) {
		if (threads > 0) {
			options_.SetIntraOpNumThreads(threads);
		}
		options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		session_ = Ort::Session(env_, to_ort_path(model_path).c_str(), options_);	
	}

	void ClipEncoder::encode(const float* input, std::size_t count, std::vector<float>& out) {
		const std::array<std::int64_t, 4> shape{ static_cast<std::int64_t>(count), 3, kImageSize, kImageSize };

		const std::size_t values = count * 3 * static_cast<std::size_t>(kImageSize) * static_cast<std::size_t>(kImageSize);

		auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		Ort::Value tensor = Ort::Value::CreateTensor<float>
			(
				memory,
				const_cast<float*>(input),
				values,
				shape.data(),
				shape.size()
			);

		const char* input_names[] = { kInputName };
		const char* output_names[] = { kOutputName };

		auto output_tensors = session_.Run(Ort::RunOptions{ nullptr }, input_names, &tensor, 1, output_names, 1);

		const float* data = output_tensors[0].GetTensorData<float>();
		out.assign(data, data + count * kEmbeddingSize);
	}

}