#include<bits/stdc++.h>

using namespace std;
class BitUnpacker {
private:
    const vector<uint8_t>& data;

    size_t byte_pos = 0;

    uint64_t buffer = 0;
    int bits = 0;   // buffer 中还有多少有效 bit

public:
    BitUnpacker(const vector<uint8_t>& data)
        : data(data) {}

    // 读取 bit_count 个 bit
    // 成功返回 true，读取结果放到 value
    bool unpack(uint64_t& value, int bit_count) {

        if (bit_count <= 0 || bit_count > 64)
            return false;

        value = 0;
        int value_bits = 0;

        while (value_bits < bit_count) {

            // buffer 没有数据，从下一个 byte 读取
            if (bits == 0) {

                if (byte_pos >= data.size())
                    return false;

                buffer = data[byte_pos++];
                bits = 8;
            }

            int take = min(bits, bit_count - value_bits);

            uint64_t mask;

            if (take == 64)
                mask = UINT64_MAX;
            else
                mask = (1ULL << take) - 1;

            // 取出最低 take bit
            uint64_t part = buffer & mask;

            // 放入结果
            value |= part << value_bits;

            // 消耗 buffer 中的数据
            buffer >>= take;
            bits -= take;

            value_bits += take;
        }

        return true;
    }

    // 直接返回结果
    uint64_t unpack(int bit_count) {

        uint64_t value;

        if (!unpack(value, bit_count))
            throw runtime_error("BitUnpacker: unexpected EOF");

        return value;
    }

    // 当前还剩多少 bit
    size_t remaining_bits() const {

        return (data.size() - byte_pos) * 8 + bits;
    }
};

int main(){
	
	    // =========================
	    // 1. 打开 events.dat
	    // =========================
	
	    ifstream in("result/events.dat", ios::binary);
	
	    if (!in) {
	        cerr << "Cannot open event.dat\n";
	        return 1;
	    }
	
	
	    // =========================
	    // 2. 读取 Header
	    // =========================
	
	    uint32_t magic;
	    uint32_t version;
	    uint64_t start_time;
	
	    uint16_t max_time_diff;
	    uint16_t width;
	    uint16_t height;
	
	    uint8_t TimestampType;
	
	    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	    in.read(reinterpret_cast<char*>(&version), sizeof(version));
	    in.read(reinterpret_cast<char*>(&start_time), sizeof(start_time));
	    in.read(reinterpret_cast<char*>(&max_time_diff),
	            sizeof(max_time_diff));
	    in.read(reinterpret_cast<char*>(&width), sizeof(width));
	    in.read(reinterpret_cast<char*>(&height), sizeof(height));
	    in.read(reinterpret_cast<char*>(&TimestampType),
	            sizeof(TimestampType));
	
	
	    // =========================
	    // 3. 检查 Header
	    // =========================
	
	    if (magic != 20260916) {
	        cerr << "Invalid file format\n";
	        return 1;
	    }
	
	    cout << "version = " << version << '\n';
	    cout << "start_time = " << start_time << '\n';
	    cout << "max_time_diff = " << max_time_diff << '\n';
	    cout << "width = " << width << '\n';
	    cout << "height = " << height << '\n';
	
	
	    // =========================
	    // 4. 读取剩余的 event data
	    // =========================
	
	    vector<uint8_t> data(
	        (istreambuf_iterator<char>(in)),
	        istreambuf_iterator<char>()
	    );
	
	    in.close();
	
	
	    // =========================
	    // 5. 计算 pixel bit 数
	    // =========================
	
	    uint32_t pixel_count =
	        static_cast<uint32_t>(width) *
	        static_cast<uint32_t>(height);
	
	    uint32_t dot_len =
	        32 - __builtin_clz(pixel_count - 1);
	
	    uint32_t event_bits =
	        max_time_diff + dot_len + 1;
	
	
	    cout << "dot_len = " << dot_len << '\n';
	    cout << "event_bits = " << event_bits << '\n';
	
	
	    // =========================
	    // 6. 创建 BitUnpacker
	    // =========================
	
	    BitUnpacker unpacker(data);
	
	
	    // =========================
	    // 7. 打开输出 txt
	    // =========================
	
	    ofstream out("result/decoded.txt");
	
	    if (!out) {
	        cerr << "Cannot create decoded.txt\n";
	        return 1;
	    }
	
	
	    // =========================
	    // 8. 解压
	    // =========================
	
	    uint64_t timestamp = start_time;
	
	    size_t event_count = 0;
	
	    while (unpacker.remaining_bits() >= event_bits) {
	
	        // dt
	        uint64_t dt =
	            unpacker.unpack(max_time_diff);
	
	        // pixel
	        uint64_t pixel =
	            unpacker.unpack(dot_len);
	
	        // polarity
	        uint64_t p =
	            unpacker.unpack(1);
	
	
	        // =========================
	        // 9. 恢复 timestamp
	        // =========================
	
	        timestamp += dt;
	
	
	        // =========================
	        // 10. 恢复 x, y
	        // =========================
	
	        uint32_t x =
	            static_cast<uint32_t>(pixel / height);
	
	        uint32_t y =
	            static_cast<uint32_t>(pixel % height);
	
	
	        // =========================
	        // 11. 写入 txt
	        // =========================
	
	        out << x << ' '
	            << y << ' '
	            << p << ' '
	            << timestamp << '\n';
	
	        event_count++;
	    }
	
	    out.close();
	
	
	    // =========================
	    // 12. 输出统计
	    // =========================
	
	    cout << "Decoded events: "
	         << event_count << '\n';
	
	    cout << "Output: decoded.txt\n";
	
	    cout << "Remaining bits: "
	         << unpacker.remaining_bits() << '\n';
	
	
	    return 0;
}
