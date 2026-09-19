#pragma once

#include <string>
#include <vector>

namespace pentimento {
	inline constexpr int kImageSize = 224;
	inline constexpr float kMean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
	inline constexpr float kStd[3] = { 0.26862954f, 0.26130258f, 0.27577711f };

	bool load_image_tensor(const std::string& path, std::vector<float>& out);

}