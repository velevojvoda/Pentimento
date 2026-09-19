#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace pentimento {
	namespace { constexpr int kChannels = 3; }

	bool load_image_tensor(const std::string& path, std::vector<float>& out) {
		int w = 0, h = 0, channels_in_file = 0;

		unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels_in_file, kChannels);
		if (pixels == nullptr || w <= 0 || h <= 0) {
			if (pixels != nullptr) stbi_image_free(pixels);
			return false;
		}

		const double scale = static_cast<double>(kImageSize) / std::max(w, h);
		const int rw = std::max(kImageSize, static_cast<int>(std::lround(w * scale)));
		const int rh = std::max(kImageSize, static_cast<int>(std::lround(h * scale)));

		std::vector<unsigned char> resized(static_cast<size_t>(rw) * rh * kChannels);
		const bool resize_ok = stbir_resize_uint8_linear(pixels, w, h, 0, resized.data(), rw, rh, 0, STBIR_RGB) != 0;
		stbi_image_free(pixels);
		if (!resize_ok) {
			return false;
		}

		const int x0 = (rw - kImageSize) / 2;
		const int y0 = (rh - kImageSize) / 2;

		out.resize(static_cast<size_t>(kImageSize) * kImageSize * kChannels);
		for (int c = 0; c < kChannels; ++c) {
			for (int y = 0; y < kImageSize; ++y) {
				for (int x = 0; x < kImageSize; ++x) {
					const std::size_t src =
						(static_cast<std::size_t>(y0 + y) * rw + (x0 + x)) * kChannels + c;
					const std::size_t dst =
						(static_cast<std::size_t>(c) * kImageSize + y) * kImageSize + x;
					out[dst] = (resized[src] / 255.0f - kMean[c]) / kStd[c];
				}
			}
		}
		return true;

	}
}