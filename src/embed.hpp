#pragma once

#include <string>

namespace pentimento {

    struct EmbedConfig {
        std::string model_path;
        std::string data_dir;   
        std::string out_path;   
        int batch_size = 32;
        int threads = 0;        
    };

    int run_embed(const EmbedConfig& config);

}  