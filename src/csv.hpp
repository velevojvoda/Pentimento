#pragma once

#include <cstddef>
#include <istream>
#include <string>
#include <string_view>
#include <vector>

namespace pentimento {

    class CsvReader {
    public:
        explicit CsvReader(std::istream& in);


        bool next(std::vector<std::string>& fields);

        const std::vector<std::string>& header() const noexcept { return header_; }

        int column(std::string_view name) const noexcept;

        std::size_t record_count() const noexcept { return records_; }

    private:
        void refill();
        int get();
        int peek();

        std::istream& in_;
        std::vector<std::string> header_;
        std::vector<char> buf_;
        std::size_t pos_ = 0;
        std::size_t len_ = 0;
        std::size_t records_ = 0;
    };

}  