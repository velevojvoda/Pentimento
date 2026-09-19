#include "embed.hpp"

#include "clip.hpp"
#include "csv.hpp"
#include "image.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace pentimento {

    namespace fs = std::filesystem;

    namespace {

        constexpr std::size_t kPixelsPerImage =
            3 * static_cast<std::size_t>(kImageSize) * kImageSize;
        constexpr std::size_t kRowBytes = kEmbeddingSize * sizeof(float);

        bool read_file_column(const fs::path& csv_path, std::vector<std::string>& files) {
            std::ifstream in(csv_path, std::ios::binary);
            if (!in) return false;

            CsvReader reader(in);
            const int col = reader.column("file");
            if (col < 0) return false;

            std::vector<std::string> row;
            while (reader.next(row)) {
                if (row.size() != reader.header().size()) {
                    std::cerr << "malformed CSV row " << reader.record_count() << "\n";
                    return false;
                }
                files.push_back(row[col]);
            }
            return true;
        }

        std::size_t resume_point(const fs::path& out_path) {
            if (!fs::exists(out_path)) return 0;
            const auto size = fs::file_size(out_path);
            const std::size_t rows = size / kRowBytes;
            if (size % kRowBytes != 0) fs::resize_file(out_path, rows * kRowBytes);
            return rows;
        }

    }  

    int run_embed(const EmbedConfig& config) {
        const fs::path data_dir(config.data_dir);
        const fs::path out_path(config.out_path);

        std::vector<std::string> files;
        if (!read_file_column(data_dir / "wikiart.csv", files)) {
            std::cerr << "cannot read " << (data_dir / "wikiart.csv").string() << "\n";
            return 1;
        }

        if (out_path.has_parent_path()) fs::create_directories(out_path.parent_path());
        std::size_t done = resume_point(out_path);
        if (done >= files.size()) {
            std::cout << "already complete: " << done << " rows\n";
            return 0;
        }
        std::cout << files.size() << " images, resuming at " << done << "\n";

        ClipEncoder clip(config.model_path, config.threads);
        std::ofstream out(out_path, std::ios::binary | std::ios::app);

        const std::size_t batch_size = static_cast<std::size_t>(config.batch_size);
        std::vector<float> batch(batch_size * kPixelsPerImage);
        std::vector<float> pixels;
        std::vector<float> embeddings;
        std::size_t failed = 0;

        const std::size_t first = done;
        const auto start = std::chrono::steady_clock::now();

        while (done < files.size()) {
            const std::size_t n = std::min(batch_size, files.size() - done);
            std::vector<bool> ok(n, true);

            for (std::size_t i = 0; i < n; ++i) {
                const fs::path path = data_dir / "images" / files[done + i];
                float* slot = batch.data() + i * kPixelsPerImage;
                if (load_image_tensor(path.string(), pixels)) {
                    std::copy(pixels.begin(), pixels.end(), slot);
                }
                else {
                    ok[i] = false;
                    ++failed;
                    std::fill_n(slot, kPixelsPerImage, 0.0f);
                    std::cerr << "\ncannot read " << path.string() << "\n";
                }
            }

            clip.encode(batch.data(), n, embeddings);

            for (std::size_t i = 0; i < n; ++i) {
                if (!ok[i]) std::fill_n(embeddings.begin() + i * kEmbeddingSize, kEmbeddingSize, 0.0f);
            }

            out.write(reinterpret_cast<const char*>(embeddings.data()),
                static_cast<std::streamsize>(n * kRowBytes));
            out.flush();
            done += n;

            const double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const double rate = (done - first) / secs;
            const double eta_min = (files.size() - done) / rate / 60.0;
            std::cout << "\r  " << done << " / " << files.size()
                << "   " << static_cast<int>(rate) << " img/s"
                << "   ~" << static_cast<int>(eta_min) << " min left   " << std::flush;
        }

        std::cout << "\ndone, " << failed << " unreadable\n";
        return 0;
    }

}  