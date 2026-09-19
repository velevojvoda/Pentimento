#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "image.hpp"
#include <chrono>
#include <cmath>
#include "clip.hpp"
#include "embed.hpp"

namespace {
#ifdef _WIN32
	std::wstring to_ort_path(const std::string& s) {
		return std::wstring(s.begin(), s.end());
	}
#else 
	std::string to_ort_path(const std::string& s) {
		return s;
	}
#endif

	std::string shape_to_string(const std::vector<int64_t>& shape) {
		std::string out = "[";
		for (std::size_t i = 0; i < shape.size(); ++i) {
			if (i > 0) out += ", ";
			out += shape[i] < 0 ? std::string("?") : std::to_string(shape[i]);	
		}
		return out + "]";
	}

	const char* element_type_name(ONNXTensorElementDataType type) {
		switch (type) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
			default: return "other";
		}
	}

	void describe(Ort::Session& session, bool inputs) {
		Ort::AllocatorWithDefaultOptions allocator;
		const std::size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();

		std::cout << (inputs ? "Inputs: " : "Outputs: ") << count << "\n";
		for (std::size_t i = 0; i < count; ++i) {
			auto name = inputs ? session.GetInputNameAllocated(i, allocator)
				: session.GetOutputNameAllocated(i, allocator);
			auto info = inputs ? session.GetInputTypeInfo(i)
				: session.GetOutputTypeInfo(i);
			auto tensor = info.GetTensorTypeAndShapeInfo();

			std::cout << "  " << name.get() << "  " << shape_to_string(tensor.GetShape())
				<< "  " << element_type_name(tensor.GetElementType()) << "\n";
		}
	}
}

int cmd_inspect(const char* model_path) {
	try {
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pentimento");
		Ort::SessionOptions options;
		Ort::Session session(env, to_ort_path(model_path).c_str(), options);

		std::cout << "model: " << model_path << "\n";
		describe(session, true);
		describe(session, false);
	}
	catch (const Ort::Exception& e) {
		std::cerr << "onnxruntime error: " << e.what() << "\n";
		return 1;
	}
	return 0;
}

int cmd_imgstat(const char* image_path) {
	std::vector<float> tensor;
	if (!pentimento::load_image_tensor(image_path, tensor)) {
		std::cerr << "cannot read image: " << image_path << "\n";
		return 1;
	}

	const auto [lo, hi] = std::minmax_element(tensor.begin(), tensor.end());
	const double sum = std::accumulate(tensor.begin(), tensor.end(), 0.0);
	const std::size_t expected =
		static_cast<std::size_t>(3) * pentimento::kImageSize * pentimento::kImageSize;

	std::cout << "image: " << image_path << "\n"
		<< "values: " << tensor.size() << "  (expected " << expected << ")\n"
		<< "min: " << *lo << "  max: " << *hi
		<< "  mean: " << sum / tensor.size() << "\n";
	return 0;
}

int cmd_embed1(const char* model_path, const char* image_path) {
	std::vector<float> pixels;
	if (!pentimento::load_image_tensor(image_path, pixels)) {
		std::cerr << "cannot read image: " << image_path << "\n";
		return 1;
	}

	try {
		pentimento::ClipEncoder clip(model_path);

		const auto start = std::chrono::steady_clock::now();
		std::vector<float> embedding;

		clip.encode(pixels.data(), 1, embedding);
		const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - start).count();

		double sum_sq = 0.0;
		for (const float v : embedding) sum_sq += static_cast<double>(v) * v;
		std::cout << "values: " << embedding.size() << "  in " << ms << " ms\n"
			<< "L2 norm: " << std::sqrt(sum_sq) << "\nfirst 8:";
		for (std::size_t i = 0; i < 8 && i < embedding.size(); ++i) {
			std::cout << " " << embedding[i];
		}
		std::cout << "\n";
	}
	catch (const Ort::Exception& e) {
		std::cerr << "onnxruntime error: " << e.what() << "\n";
		return 1;
	}

	return 0;
}

int cmd_embed(int argc, char** argv) {
	pentimento::EmbedConfig config;
	config.model_path = argv[2];
	config.data_dir = argv[3];
	config.out_path = argv[4];
	if (argc > 5) config.batch_size = std::stoi(argv[5]);
	if (argc > 6) config.threads = std::stoi(argv[6]);

	try {
		return pentimento::run_embed(config);
	}
	catch (const Ort::Exception& e) {
		std::cerr << "\nonnxruntime error: " << e.what() << "\n";
		return 1;
	}
}

int main(int argc, char** argv) {
	const std::string cmd = argc > 1 ? argv[1] : "";

	if (cmd == "inspect" && argc == 3) return cmd_inspect(argv[2]);
	if (cmd == "imgstat" && argc == 3) return cmd_imgstat(argv[2]);
	if (cmd == "embed1" && argc == 4) return cmd_embed1(argv[2], argv[3]);
	if (cmd == "embed" && argc >= 5) return cmd_embed(argc, argv);

	std::cerr << "usage:\n"
		<< "  pentimento inspect <model.onnx>\n"
		<< "  pentimento imgstat <image>\n";
	return 1;
}