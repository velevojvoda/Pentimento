#include "csv.hpp"

namespace pentimento {

	namespace {
		constexpr std::size_t kBufferSize = 1 << 20;
	}

	CsvReader::CsvReader(std::istream& in) : in_(in), buf_(kBufferSize) {
		refill();
	
		if (len_ >= 3 && static_cast<unsigned char>(buf_[0]) == 0xEF
			&& static_cast<unsigned char>(buf_[1]) == 0xBB
			&& static_cast<unsigned char>(buf_[2]) == 0xBF) {
			pos_ = 3;
		}

		next(header_);
		records_ = 0;
	}

	void CsvReader::refill() {
		in_.read(buf_.data(), static_cast<std::streamsize>(buf_.size()));
		len_ = static_cast<std::size_t>(in_.gcount());
		pos_ = 0;
	}
	
	int CsvReader::get() {
		if (pos_ == len_) {
			refill();
			if (len_ == 0)
				return -1;
		}
		return static_cast<unsigned char>(buf_[pos_++]);
	}

	int CsvReader::peek() {
		if (pos_ == len_) {
			refill();
			if (len_ == 0)
				return -1;
		}
		return static_cast<unsigned char>(buf_[pos_]);
	}

	bool CsvReader::next(std::vector<std::string>& fields) {
		fields.clear();

		if(peek() < 0)
			return false;

		std::string field;
		bool quoted = false;

		for (;;) {
			const int c = get();
			if (c < 0) {
				fields.push_back(std::move(field));
				break;
			}

			if (quoted) {
				if (c != '"') {
					field.push_back(static_cast<char>(c));
				}
				else if (peek() == '"') {
					get();
					field.push_back('"');
				}
				else {
					quoted = false;
				}
				continue;
			}

			if (c == '"' && field.empty()) {
				quoted = true;
			}
			else if (c == ',') {
				fields.push_back(std::move(field));
				field.clear();
			}
			else if (c == '\n') {
				fields.push_back(std::move(field));
				break;
			}
			else if (c == '\r') {
				if (peek() == '\n') get();
				fields.push_back(std::move(field));
				break;
			}
			else {
				field.push_back(static_cast<char>(c));
			}
		}

		++records_;
		return true;
	}
	
	int CsvReader::column(std::string_view name) const noexcept {
		for (std::size_t i = 0; i < header_.size(); ++i) {
			if (header_[i] == name)
				return static_cast<int>(i);
		}
		return -1;
	}
}