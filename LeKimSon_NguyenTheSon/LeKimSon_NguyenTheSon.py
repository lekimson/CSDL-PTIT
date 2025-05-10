import heapq  # Thư viện để tạo hàng đợi ưu tiên (priority queue) cho thuật toán Huffman
import os  # Thư viện để thao tác với hệ thống file (mở, đọc, ghi file)
import pickle  # Thư viện để lưu và tải dữ liệu (dùng để lưu bảng mã Huffman)
import time  # Thư viện để đo thời gian xử lý (nén và giải nén)
import tkinter as tk  # Thư viện để tạo giao diện đồ họa (GUI)
from tkinter import ttk, filedialog, messagebox  # Các thành phần giao diện (nút, hộp thoại, thông báo)
from collections import Counter  # Thư viện để đếm tần suất ký tự trong văn bản
import matplotlib.pyplot as plt  # Thư viện để vẽ biểu đồ thống kê
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Để nhúng biểu đồ vào giao diện Tkinter
import numpy as np  # Thư viện để xử lý mảng và tính toán số học (dùng cho biểu đồ)
from PIL import Image, ImageTk  # Thư viện để xử lý và hiển thị ảnh
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import logging  # Thư viện để ghi log hoạt động của ứng dụng

# Cấu hình logging để ghi lại các sự kiện (nén, giải nén, lỗi, v.v.) vào file 'huffman_app.log'
logging.basicConfig(filename='huffman_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Đa ngôn ngữ
LANGUAGES = {
    'vi': {
        'export_pdf': "Xuất Báo cáo PDF",
        'success_pdf': "Báo cáo PDF đã được lưu tại: {}",
        'no_stats': "Không có dữ liệu thống kê để xuất!"
    },
    'en': {
        'export_pdf': "Export PDF Report",
        'success_pdf': "PDF report saved to: {}",
        'no_stats': "No statistical data to export!"
    }
}
current_lang = 'vi'
def _(key): return LANGUAGES.get(current_lang, LANGUAGES['vi']).get(key, key)

# Lớp HuffmanNode: Đại diện cho một nút trong cây Huffman
class HuffmanNode:
    # Hàm khởi tạo: Tạo một nút với ký tự (char) và tần suất (freq)
    def __init__(self, char, freq):
        self.char = char  # Ký tự (hoặc None nếu là nút trung gian)
        self.freq = freq  # Tần suất xuất hiện của ký tự
        self.left = None  # Con trái của nút
        self.right = None  # Con phải của nút
    
    # Hàm so sánh: Dùng để sắp xếp các nút trong hàng đợi ưu tiên (dựa trên tần suất)
    def __lt__(self, other):
        return self.freq < other.freq

# Lớp HuffmanCoding: Xử lý thuật toán nén và giải nén Huffman
class HuffmanCoding:
    # Hàm khởi tạo: Khởi tạo các thuộc tính cần thiết cho thuật toán Huffman
    def __init__(self):
        self.heap = []  # Hàng đợi ưu tiên để lưu các nút của cây Huffman
        self.codes = {}  # Từ điển lưu mã Huffman cho từng ký tự
        self.reverse_codes = {}  # Từ điển lưu ký tự tương ứng với mã Huffman (dùng để giải nén)
        self.original_size = 0  # Kích thước dữ liệu gốc (trước khi nén)
        self.compressed_size = 0  # Kích thước dữ liệu sau khi nén
        self.encoding_time = 0  # Thời gian nén dữ liệu
        self.decoding_time = 0  # Thời gian giải nén dữ liệu
        self.frequency = None  # Từ điển tần suất ký tự
        self.compressed_data = None  # Dữ liệu đã nén (dạng byte array)
        self.char_to_bytes = {}  # Từ điển ánh xạ ký tự sang bytes
    
    # Hàm tạo từ điển tần suất: Đếm số lần xuất hiện của mỗi ký tự trong văn bản
    def make_frequency_dict(self, text):
        self.frequency = Counter(text)  # Đếm tần suất trên chuỗi ký tự gốc
        return self.frequency
    
    # Hàm tạo ánh xạ ký tự sang bytes: Lưu cách mã hóa từng ký tự thành bytes
    def build_char_to_bytes(self, text):
        self.char_to_bytes = {}
        for char in set(text):  # Duyệt qua các ký tự duy nhất
            self.char_to_bytes[char] = char.encode('utf-8')
    
    # Hàm tạo hàng đợi ưu tiên: Chuyển từ điển tần suất thành các nút và đưa vào hàng đợi
    def make_heap(self, frequency):
        for char, freq in frequency.items():
            heapq.heappush(self.heap, HuffmanNode(char, freq))  # Đưa từng nút vào hàng đợi
    
    # Hàm hợp nhất các nút: Xây dựng cây Huffman bằng cách hợp nhất các nút có tần suất nhỏ nhất
    def merge_nodes(self):
        while len(self.heap) > 1:  # Lặp cho đến khi chỉ còn 1 nút (gốc của cây)
            node1, node2 = heapq.heappop(self.heap), heapq.heappop(self.heap)  # Lấy 2 nút có tần suất nhỏ nhất
            merged = HuffmanNode(None, node1.freq + node2.freq)  # Tạo nút mới với tần suất là tổng
            merged.left, merged.right = node1, node2  # Gán con trái và con phải
            heapq.heappush(self.heap, merged)  # Đưa nút mới vào hàng đợi
    
    # Hàm đệ quy tạo mã Huffman: Duyệt cây Huffman để tạo mã nhị phân cho từng ký tự
    def make_codes_helper(self, node, current_code):
        if not node:  # Nếu nút rỗng, thoát
            return
        if node.char is not None:  # Nếu là nút lá (có ký tự)
            self.codes[node.char] = current_code or "0"  # Gán mã cho ký tự (mặc định là "0" nếu không có mã)
            self.reverse_codes[current_code or "0"] = node.char  # Lưu ngược lại để giải nén
            return
        # Đệ quy: Thêm "0" cho nhánh trái, "1" cho nhánh phải
        self.make_codes_helper(node.left, current_code + "0")
        self.make_codes_helper(node.right, current_code + "1")
    
    # Hàm tạo mã Huffman: Lấy nút gốc của cây và tạo mã cho tất cả ký tự
    def make_codes(self):
        if not self.heap:  # Nếu hàng đợi rỗng, thoát
            return
        root = heapq.heappop(self.heap)  # Lấy nút gốc
        self.make_codes_helper(root, "")  # Tạo mã từ nút gốc
    
    # Hàm mã hóa văn bản: Chuyển văn bản thành chuỗi nhị phân dựa trên bảng mã Huffman
    def get_encoded_text(self, text):
        return "".join(self.codes[char] for char in text)  # Ghép mã của từng ký tự thành chuỗi
    
    # Hàm thêm padding: Đảm bảo chuỗi nhị phân có độ dài chia hết cho 8 (để chuyển thành byte)
    def pad_encoded_text(self, encoded_text):
        padding = 8 - (len(encoded_text) % 8)  # Tính số bit cần thêm
        encoded_text += "0" * padding  # Thêm bit 0 vào cuối
        return f"{padding:08b}" + encoded_text  # Thêm 8 bit đầu tiên để lưu số padding
    
    # Hàm chuyển chuỗi nhị phân thành byte array: Chuyển từng nhóm 8 bit thành 1 byte
    def get_byte_array(self, padded_encoded_text):
        return bytearray(int(padded_encoded_text[i:i+8], 2) for i in range(0, len(padded_encoded_text), 8))
    
    # Hàm nén dữ liệu: Thực hiện toàn bộ quá trình nén dữ liệu bằng thuật toán Huffman
    def compress(self, text, progress_callback=None):
        # Đặt lại trạng thái trước khi nén
        self.heap = []
        self.codes = {}
        self.reverse_codes = {}
        self.compressed_data = None
        
        start_time = time.perf_counter()  # Bắt đầu đo thời gian nén
        
        # Tính kích thước gốc (sau khi mã hóa thành bytes)
        text_bytes = text.encode('utf-8')
        self.original_size = len(text_bytes)
        
        # Đếm tần suất trên ký tự gốc và tạo ánh xạ ký tự sang bytes
        frequency = self.make_frequency_dict(text)
        self.build_char_to_bytes(text)
        
        # Xây dựng cây Huffman và tạo mã
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()
        
        # Mã hóa văn bản thành chuỗi nhị phân
        encoded_text = self.get_encoded_text(text)
        padded_encoded_text = self.pad_encoded_text(encoded_text)
        bytes_array = self.get_byte_array(padded_encoded_text)
        self.compressed_size = len(bytes_array)  # Lưu kích thước dữ liệu nén
        self.encoding_time = time.perf_counter() - start_time  # Tính thời gian nén
        
        self.compressed_data = bytes_array  # Lưu dữ liệu đã nén
        
        if progress_callback:  # Cập nhật thanh tiến trình (nếu có)
            progress_callback(100)
        return bytes_array, frequency  # Trả về dữ liệu nén và từ điển tần suất
    
    # Hàm bỏ padding: Loại bỏ các bit padding để lấy lại chuỗi nhị phân gốc
    def remove_padding(self, bit_string):
        padding = int(bit_string[:8], 2)  # Đọc 8 bit đầu tiên để biết số padding
        bit_string = bit_string[8:]  # Bỏ 8 bit đầu
        return bit_string[:-padding] if padding > 0 else bit_string  # Bỏ bit padding
    
    # Hàm giải mã văn bản: Chuyển chuỗi nhị phân thành văn bản gốc dựa trên bảng mã
    def decode_text(self, encoded_text):
        current_code, decoded_text = "", []
        for bit in encoded_text:  # Duyệt từng bit
            current_code += bit  # Thêm bit vào chuỗi mã
            if current_code in self.reverse_codes:  # Nếu chuỗi mã khớp với một ký tự
                decoded_text.append(self.reverse_codes[current_code])  # Thêm ký tự vào kết quả
                current_code = ""  # Đặt lại chuỗi mã
        return decoded_text
    
    # Hàm giải nén dữ liệu: Thực hiện toàn bộ quá trình giải nén
    def decompress(self, byte_array, reverse_codes, char_to_bytes, progress_callback=None):
        self.reverse_codes = reverse_codes  # Lưu bảng mã ngược
        self.char_to_bytes = char_to_bytes  # Lưu ánh xạ ký tự sang bytes
        bit_string = "".join(bin(byte)[2:].rjust(8, '0') for byte in byte_array)  # Chuyển byte array thành chuỗi nhị phân
        
        if progress_callback:  # Cập nhật thanh tiến trình (nếu có)
            progress_callback(100)
        
        encoded_text = self.remove_padding(bit_string)  # Bỏ padding
        decoded_chars = self.decode_text(encoded_text)  # Giải mã thành danh sách ký tự
        
        # Chuyển danh sách ký tự thành bytes
        decoded_bytes = bytearray()
        for char in decoded_chars:
            decoded_bytes.extend(self.char_to_bytes[char])
        
        return decoded_bytes

# Lớp HuffmanGUI: Xây dựng giao diện đồ họa cho ứng dụng
class HuffmanGUI(tk.Tk):
    # Hàm khởi tạo: Thiết lập cửa sổ chính và các thuộc tính ban đầu
    def __init__(self):
        super().__init__()
        self.title("Ứng dụng Nén và Giải nén Huffman")  # Đặt tiêu đề cửa sổ
        self.geometry("1000x800")  # Kích thước cửa sổ
        self.configure(bg="#f0f0f0")  # Màu nền
        
        self.huffman = HuffmanCoding()  # Khởi tạo đối tượng HuffmanCoding
        self.compressed_file = None  # Lưu đường dẫn file nén
        self.codes_file = None  # Lưu đường dẫn file bảng mã
        self.frequency = None  # Lưu từ điển tần suất ký tự
        self.original_text = None  # Lưu văn bản gốc
        
        self.history = []  # Lưu lịch sử nén/giải nén
        self.create_widgets()  # Tạo các thành phần giao diện
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.create_language_menu()  # Xử lý khi người dùng đóng cửa sổ
    
    # Hàm tạo giao diện: Tạo các tab và thành phần giao diện (nút, ô nhập liệu, biểu đồ, v.v.)
    def create_widgets(self):
        # Tạo giao diện với các định dạng (style) cho nút, nhãn
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=5)  # Định dạng nút
        style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")  # Định dạng nhãn
        style.configure("Header.TLabel", font=("Arial", 16, "bold"), background="#f0f0f0")  # Định dạng tiêu đề
        
        # Tạo khung chính
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Tiêu đề chính
        header_label = ttk.Label(main_frame, text="HỆ THỐNG NÉN DỮ LIỆU HUFFMAN", style="Header.TLabel")
        header_label.pack(pady=10)
        
        # Tạo các tab (Nén Văn bản, Nén File, Thống kê, Hướng dẫn)
        tab_control = ttk.Notebook(main_frame)
        
        self.text_tab = ttk.Frame(tab_control)  # Tab 1: Nén và Giải nén Văn bản
        self.file_tab = ttk.Frame(tab_control)  # Tab 2: Nén và Giải nén File
        self.stats_tab = ttk.Frame(tab_control)  # Tab 3: Phân tích & Thống kê
        self.help_tab = ttk.Frame(tab_control)  # Tab 4: Hướng dẫn
        
        tab_control.add(self.text_tab, text="Nén và Giải nén Văn bản")
        tab_control.add(self.file_tab, text="Nén và Giải nén File")
        tab_control.add(self.stats_tab, text="Phân tích & Thống kê")
        tab_control.add(self.help_tab, text="Hướng dẫn")
        tab_control.pack(expand=1, fill="both")
        
        # Tab 1: Nén và Giải nén Văn bản
        # Khung nhập văn bản
        input_frame = ttk.LabelFrame(self.text_tab, text="Văn bản đầu vào")
        input_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.input_text = tk.Text(input_frame, height=10, width=80, font=("Arial", 12))
        self.input_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Khung chứa các nút điều khiển
        button_frame = ttk.Frame(self.text_tab)
        button_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(button_frame, text="Nén Văn bản", command=self.compress_text).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Giải nén Văn bản", command=self.decompress_text).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Xóa", command=self.clear_text).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Tạo Văn bản Mẫu", command=self.generate_sample_text).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Lưu Kết Quả", command=self.save_text_result).pack(side="left", padx=5)
        
        # Khung hiển thị kết quả
        output_frame = ttk.LabelFrame(self.text_tab, text="Kết quả")
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.output_text = tk.Text(output_frame, height=10, width=80, font=("Arial", 12))
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 2: Nén và Giải nén File
        # Khung chọn file
        file_frame_top = ttk.Frame(self.file_tab)
        file_frame_top.pack(fill="x", padx=10, pady=20)
        ttk.Label(file_frame_top, text="Chọn tệp để nén hoặc giải nén:").pack(side="left", padx=5)
        self.file_path_var = tk.StringVar()  # Biến lưu đường dẫn file
        ttk.Entry(file_frame_top, textvariable=self.file_path_var, width=50).pack(side="left", padx=5)
        ttk.Button(file_frame_top, text="Duyệt...", command=self.browse_file).pack(side="left", padx=5)
        
        # Khung chứa nút nén và giải nén file
        file_frame_buttons = ttk.Frame(self.file_tab)
        file_frame_buttons.pack(fill="x", padx=10, pady=10)
        ttk.Button(file_frame_buttons, text="Nén File", command=self.compress_file).pack(side="left", padx=5)
        ttk.Button(file_frame_buttons, text="Giải nén File", command=self.decompress_file).pack(side="left", padx=5)
        
        # Khung hiển thị thanh tiến trình
        progress_frame = ttk.LabelFrame(self.file_tab, text="Tiến trình")
        progress_frame.pack(fill="x", padx=10, pady=10)
        self.progress_var = tk.DoubleVar()  # Biến lưu giá trị tiến trình
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=10, pady=10)
        
        # Khung hiển thị thông tin file
        file_info_frame = ttk.LabelFrame(self.file_tab, text="Thông tin")
        file_info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.file_info_text = tk.Text(file_info_frame, height=15, width=80, font=("Arial", 12))
        self.file_info_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 3: Phân tích & Thống kê
        self.stats_frame = ttk.Frame(self.stats_tab)
        self.stats_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Khung chứa biểu đồ
        graph_frame = ttk.LabelFrame(self.stats_frame, text="Phân tích hiệu suất")
        graph_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 5))  # Tạo 3 biểu đồ
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)  # Nhúng biểu đồ vào giao diện
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Khung hiển thị thông tin thống kê
        stats_info_frame = ttk.LabelFrame(self.stats_frame, text="Thông tin chi tiết")
        stats_info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.stats_info_text = tk.Text(stats_info_frame, height=8, width=80, font=("Arial", 12))
        self.stats_info_text.pack(fill="both", expand=True, padx=10, pady=5)
        # Khung hiển thị lịch sử nén/giải nén
        history_label = ttk.Label(stats_info_frame, text="Lịch sử nén/giải nén (5 lần gần nhất):", font=("Arial", 11, "bold"))
        history_label.pack(anchor="w", padx=5, pady=(10, 0))

        self.history_text = tk.Text(stats_info_frame, height=6, font=("Arial", 11))
        self.history_text.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Button(stats_info_frame, text="Xuất Báo cáo Thống kê", command=self.export_stats).pack(side="left", pady=5)
        ttk.Button(stats_info_frame, text=_("export_pdf"), command=self.export_stats_pdf).pack(side="left", pady=5, padx=5)
        ttk.Button(stats_info_frame, text="Xóa Dữ Liệu", command=self.clear_stats).pack(side="left", pady=5)
        
        # Tab 4: Hướng dẫn
        help_text = """
        HƯỚNG DẪN SỬ DỤNG ỨNG DỤNG NÉN DỮ LIỆU HUFFMAN
        
        1. Nén và Giải nén Văn bản:
           - Nhập văn bản vào ô "Văn bản đầu vào"
           - Nhấn "Tạo Văn bản Mẫu" để tạo văn bản mẫu
           - Nhấn "Nén Văn bản" để thực hiện nén
           - Nhấn "Giải nén Văn bản" để khôi phục văn bản gốc
           - Nhấn "Lưu Kết Quả" để lưu văn bản đã nén/giải nén
           - Nhấn "Xóa" để xóa nội dung các ô
        
        2. Nén và Giải nén File:
           - Nhấn "Duyệt..." để chọn file (hỗ trợ văn bản, ảnh PNG/JPEG, v.v.)
           - Nhấn "Nén File" để tạo file nén (.bin) và file mã (.codes)
           - Nhấn "Giải nén File" để khôi phục file gốc từ file đã nén
           - Thanh tiến trình hiển thị trạng thái xử lý
        
        3. Phân tích & Thống kê:
           - Hiển thị thông số về kích thước trước và sau khi nén
           - Hiển thị tỷ lệ nén, thời gian xử lý, và biểu đồ phân tích
           - Nhấn "Xóa Dữ Liệu" để xóa biểu đồ và thông tin thống kê
           - Xuất báo cáo thống kê bằng nút "Xuất Báo cáo Thống kê"
           - Xuất Báo cáo PDF bằng nút "Xuất Báo cáo PDF"
           - Hiển thị lịch sử nén/giải nén ( 5 lần gần nhất)
        
        Thuật toán Huffman:
        - Xây dựng cây Huffman dựa trên tần suất xuất hiện của các ký tự
        - Mã hóa các ký tự bằng mã nhị phân (0, 1) theo nguyên tắc:
          + Ký tự xuất hiện nhiều có mã ngắn hơn
          + Không có mã nào là tiền tố của mã khác
        - Độ phức tạp: O(n log n) với n là số ký tự khác nhau
        """
        help_scroll = ttk.Scrollbar(self.help_tab)  # Thanh cuộn
        help_scroll.pack(side="right", fill="y")
        help_text_widget = tk.Text(self.help_tab, font=("Arial", 12), yscrollcommand=help_scroll.set)
        help_text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        help_text_widget.insert("1.0", help_text)  # Chèn nội dung hướng dẫn
        help_text_widget.config(state="disabled")  # Không cho chỉnh sửa
        help_scroll.config(command=help_text_widget.yview)
    
    # Hàm tạo văn bản mẫu: Tạo một đoạn văn bản mẫu để người dùng thử nghiệm
    def generate_sample_text(self):
        sample_text = "Chúng tôi là Nguyễn Thế Sơn và Lê Kim Sơn, sinh viên lớp D22CQDK01_N. Đây là ứng dụng về thuật toán greedy cho bài toán Human coding trong nén dử liệu."
        self.input_text.delete("1.0", tk.END)  # Xóa nội dung cũ
        self.input_text.insert("1.0", sample_text)  # Thêm văn bản mẫu
        logging.info("Generated sample text")  # Ghi log
    
    # Hàm xử lý khi đóng ứng dụng: Hỏi người dùng trước khi thoát
    def on_closing(self):
        if messagebox.askyesno("Xác nhận Thoát", "Bạn có chắc chắn muốn thoát ứng dụng?"):
            logging.info("Application closed by user")  # Ghi log
            self.destroy()  # Đóng ứng dụng
    
    # Hàm cập nhật thanh tiến trình: Cập nhật giá trị thanh tiến trình trong Tab 2
    def update_progress(self, value):
        self.progress_var.set(value)
        self.update_idletasks()
    
    # Hàm chọn file: Mở hộp thoại để người dùng chọn file cần nén hoặc giải nén
    def browse_file(self):
        filename = filedialog.askopenfilename(title="Chọn file", filetypes=[("All files", "*.*"), ("Text files", "*.txt"), ("Image files", "*.png;*.jpg;*.jpeg")])
        if filename:
            self.file_path_var.set(filename)  # Lưu đường dẫn file
            self.file_info_text.delete("1.0", tk.END)  # Xóa thông tin cũ
            self.file_info_text.insert(tk.END, f"Đã chọn file: {filename}\nKích thước: {os.path.getsize(filename)} bytes\n")  # Hiển thị thông tin file
            logging.info(f"Selected file: {filename}")  # Ghi log
    
    # Hàm nén văn bản: Nén văn bản trong ô nhập liệu và hiển thị kết quả
    def compress_text(self):
        text = self.input_text.get("1.0", "end-1c")  # Lấy văn bản từ ô nhập liệu
        if not text:  # Kiểm tra xem có văn bản không
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập văn bản để nén!")
            logging.warning("Attempted to compress empty text")
            return
        
        try:
            self.original_text = text  # Lưu văn bản gốc
            compressed_data, frequency = self.huffman.compress(text)  # Nén dữ liệu
            self.frequency = frequency  # Lưu từ điển tần suất
            
            # Hiển thị kết quả nén
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Văn bản đã được nén!\n\n")
            self.output_text.insert(tk.END, f"Kích thước gốc: {self.huffman.original_size} bytes\n")
            self.output_text.insert(tk.END, f"Kích thước sau nén: {self.huffman.compressed_size} bytes\n")
            compression_ratio = (1 - self.huffman.compressed_size / self.huffman.original_size) * 100
            self.output_text.insert(tk.END, f"Tỷ lệ nén: {compression_ratio:.2f}%\n")
            self.output_text.insert(tk.END, f"Thời gian nén: {self.huffman.encoding_time:.6f} giây\n\n")
            
            # Hiển thị bảng mã Huffman (chỉ 10 ký tự đầu tiên nếu nhiều hơn 10)
            self.output_text.insert(tk.END, "Bảng mã Huffman (một phần):\n")
            sorted_codes = sorted(self.huffman.codes.items(), key=lambda x: frequency[x[0]], reverse=True)
            for char, code in sorted_codes[:10]:
                if frequency[char] > 0:  # Chỉ hiển thị ký tự có tần suất lớn hơn 0
                    self.output_text.insert(tk.END, f"Ký tự: '{char}', Tần suất: {frequency[char]}, Mã Huffman: {code}\n")
            if len([char for char, freq in frequency.items() if freq > 0]) > 10:
                self.output_text.insert(tk.END, "...\n")
            
            self.update_statistics(frequency)  # Cập nhật tab Thống kê
            self.history.append({
                'action': 'compress',
                'original_size': self.huffman.original_size,
                'compressed_size': self.huffman.compressed_size,
                'time': self.huffman.encoding_time
            })
            logging.info("Text compressed successfully")  # Ghi log
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể nén văn bản: {str(e)}")
            logging.error(f"Text compression failed: {str(e)}")
    
    # Hàm giải nén văn bản: Giải nén dữ liệu và hiển thị kết quả
    def decompress_text(self):
        # Kiểm tra xem đã nén dữ liệu trước đó chưa
        if not hasattr(self.huffman, 'reverse_codes') or not self.huffman.reverse_codes:
            messagebox.showwarning("Cảnh báo", "Bạn cần nén văn bản trước khi giải nén!")
            logging.warning("Attempted to decompress without prior compression")
            return
        
        if self.huffman.compressed_data is None:
            messagebox.showwarning("Cảnh báo", "Bạn cần nén văn bản trước khi giải nén!")
            logging.warning("Attempted to decompress without compressed data")
            return
        
        try:
            # Lặp lại giải nén nhiều lần để đo thời gian chính xác hơn
            num_iterations = 1000
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                decompressed_data = self.huffman.decompress(
                    self.huffman.compressed_data,
                    self.huffman.reverse_codes,
                    self.huffman.char_to_bytes
                )
            total_time = time.perf_counter() - start_time
            self.huffman.decoding_time = total_time / num_iterations  # Tính thời gian trung bình
            
            # Hiển thị kết quả giải nén
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Văn bản đã được giải nén thành công!\n\n")
            self.output_text.insert(tk.END, f"Nội dung giải nén: \n{decompressed_data.decode('utf-8')}\n\n")
            self.output_text.insert(tk.END, f"Thời gian giải nén (trung bình): {self.huffman.decoding_time:.6f} giây\n")
            
            self.update_statistics(self.frequency)  # Cập nhật tab Thống kê
            self.history.append({
                'action': 'decompress',
                'time': self.huffman.decoding_time
            })
            logging.info("Text decompressed successfully")  # Ghi log
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể giải nén văn bản: {str(e)}")
            logging.error(f"Text decompression failed: {str(e)}")
    
    # Hàm lưu kết quả: Lưu nội dung ô kết quả thành file .txt
    def save_text_result(self):
        result = self.output_text.get("1.0", "end-1c")  # Lấy nội dung ô kết quả
        if not result:
            messagebox.showwarning("Cảnh báo", "Không có kết quả để lưu!")
            return
        
        output_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="Lưu kết quả")
        if not output_path:
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            messagebox.showinfo("Thành công", f"Kết quả đã được lưu tại: {output_path}")
            logging.info(f"Saved text result to: {output_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu kết quả: {str(e)}")
            logging.error(f"Failed to save text result: {str(e)}")
    
    # Hàm xóa văn bản: Xóa nội dung ô nhập liệu và ô kết quả
    def clear_text(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.huffman = HuffmanCoding()  # Đặt lại đối tượng HuffmanCoding
        self.frequency = None
        self.original_text = None
        logging.info("Cleared text fields")
    
    # Hàm nén file: Nén file được chọn và lưu kết quả
    def compress_file(self):
        filepath = self.file_path_var.get()  # Lấy đường dẫn file
        if not filepath or not os.path.exists(filepath):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file hợp lệ để nén!")
            logging.warning("Attempted to compress invalid or non-existent file")
            return
        
        try:
            # Đọc file dưới dạng văn bản nếu là file .txt, nếu không thì đọc dưới dạng nhị phân
            if filepath.lower().endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = file.read()
            else:
                with open(filepath, 'rb') as file:
                    binary_data = file.read()
                    data = binary_data.decode('utf-8', errors='replace')
            
            self.progress_var.set(0)  # Đặt lại thanh tiến trình
            compressed_data, frequency = self.huffman.compress(data, self.update_progress)  # Nén dữ liệu
            self.frequency = frequency
            
            output_path = filepath + ".bin"  # File nén
            with open(output_path, 'wb') as output:
                output.write(bytes(compressed_data))
            
            codes_path = filepath + ".codes"  # File lưu bảng mã
            with open(codes_path, 'wb') as codes_file:
                pickle.dump({
                    'reverse_codes': self.huffman.reverse_codes,
                    'char_to_bytes': self.huffman.char_to_bytes
                }, codes_file)
            
            # Hiển thị thông tin file nén
            self.file_info_text.delete("1.0", tk.END)
            self.file_info_text.insert(tk.END, f"File đã được nén thành công!\n\n")
            self.file_info_text.insert(tk.END, f"File gốc: {filepath}\nKích thước gốc: {self.huffman.original_size} bytes\n\n")
            self.file_info_text.insert(tk.END, f"File nén: {output_path}\nKích thước sau nén: {self.huffman.compressed_size} bytes\n\n")
            compression_ratio = (1 - self.huffman.compressed_size / self.huffman.original_size) * 100
            self.file_info_text.insert(tk.END, f"Tỷ lệ nén: {compression_ratio:.2f}%\nThời gian nén: {self.huffman.encoding_time:.6f} giây\n")
            
            self.compressed_file, self.codes_file = output_path, codes_path
            self.update_statistics(frequency)  # Cập nhật tab Thống kê
            messagebox.showinfo("Thành công", f"File đã được nén thành công!\nKết quả lưu tại: {output_path}")
            logging.info(f"File compressed successfully: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể nén file: {str(e)}")
            logging.error(f"File compression failed: {str(e)}")
    
    # Hàm giải nén file: Giải nén file và lưu kết quả
    def decompress_file(self):
        filepath = self.file_path_var.get()  # Lấy đường dẫn file
        if not filepath or not os.path.exists(filepath) or not filepath.endswith('.bin'):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file nén (.bin) để giải nén!")
            logging.warning("Attempted to decompress invalid or non-.bin file")
            return
        
        codes_path = filepath[:-4] + '.codes'  # Đường dẫn file bảng mã
        if not os.path.exists(codes_path):
            messagebox.showwarning("Cảnh báo", f"Không tìm thấy file mã ({codes_path})!")
            logging.warning(f"Missing codes file: {codes_path}")
            return
        
        try:
            with open(filepath, 'rb') as file:
                compressed_data = file.read()  # Đọc file nén
            with open(codes_path, 'rb') as codes_file:
                data = pickle.load(codes_file)
                reverse_codes = data['reverse_codes']
                char_to_bytes = data['char_to_bytes']
            
            self.progress_var.set(0)  # Đặt lại thanh tiến trình
            decompressed_data = self.huffman.decompress(compressed_data, reverse_codes, char_to_bytes, self.update_progress)  # Giải nén
            
            output_path = filepath[:-4]  # Đường dẫn file giải nén
            if os.path.exists(output_path):
                output_path += "_decompressed"  # Thêm hậu tố nếu file đã tồn tại
            with open(output_path, 'wb') as output:
                output.write(decompressed_data)
            
            # Hiển thị thông tin file giải nén
            self.file_info_text.delete("1.0", tk.END)
            self.file_info_text.insert(tk.END, f"File đã được giải nén thành công!\n\n")
            self.file_info_text.insert(tk.END, f"File nén: {filepath}\nKích thước file nén: {os.path.getsize(filepath)} bytes\n\n")
            self.file_info_text.insert(tk.END, f"File giải nén: {output_path}\nKích thước sau giải nén: {os.path.getsize(output_path)} bytes\n\n")
            self.file_info_text.insert(tk.END, f"Thời gian giải nén: {self.huffman.decoding_time:.6f} giây\n")
            
            self.update_statistics(self.frequency)  # Cập nhật tab Thống kê
            messagebox.showinfo("Thành công", f"File đã được giải nén thành công!\nKết quả lưu tại: {output_path}")
            logging.info(f"File decompressed successfully: {output_path}")
            
            # Nếu file giải nén là ảnh, hiển thị ảnh
            if output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.display_image(output_path)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể giải nén file: {str(e)}")
            logging.error(f"File decompression failed: {str(e)}")
    
    # Hàm hiển thị ảnh: Hiển thị ảnh giải nén trong một cửa sổ mới
    def display_image(self, image_path):
        try:
            img = Image.open(image_path).resize((300, 300), Image.LANCZOS)  # Mở và thay đổi kích thước ảnh
            photo = ImageTk.PhotoImage(img)
            img_window = tk.Toplevel(self)  # Tạo cửa sổ mới
            img_window.title("Ảnh Giải Nén")
            img_label = tk.Label(img_window, image=photo)
            img_label.image = photo
            img_label.pack(padx=10, pady=10)
            logging.info(f"Displayed decompressed image: {image_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh: {str(e)}")
            logging.error(f"Failed to display image: {str(e)}")
    
    # Hàm xuất báo cáo thống kê: Lưu thông tin thống kê vào file .txt
    def export_stats(self):
        if not hasattr(self.huffman, 'original_size') or not self.frequency:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu thống kê để xuất!")
            logging.warning("Attempted to export stats without data")
            return
        
        output_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="Lưu báo cáo thống kê")
        if not output_path:
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("BÁO CÁO THỐNG KÊ NÉN DỮ LIỆU HUFFMAN\n\n")
                f.write(f"Kích thước dữ liệu gốc: {self.huffman.original_size} bytes\n")
                f.write(f"Kích thước sau khi nén: {self.huffman.compressed_size} bytes\n")
                compression_ratio = (1 - self.huffman.compressed_size / self.huffman.original_size) * 100
                f.write(f"Tỷ lệ nén: {compression_ratio:.2f}%\n")
                f.write(f"Thời gian nén: {self.huffman.encoding_time:.6f} giây\n")
                f.write(f"Thời gian giải nén: {self.huffman.decoding_time:.6f} giây\n\n")
                f.write("Bảng tần suất ký tự:\n")
                for char, freq in sorted(self.frequency.items()):
                    if freq > 0:
                        f.write(f"'{char}': {freq}\n")
            messagebox.showinfo("Thành công", f"Báo cáo đã được lưu tại: {output_path}")
            logging.info(f"Exported statistics to: {output_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất báo cáo: {str(e)}")
            logging.error(f"Failed to export statistics: {str(e)}")
    
    # Hàm xóa dữ liệu thống kê: Xóa biểu đồ và thông tin trong tab Thống kê
    def clear_stats(self):
        self.ax1.clear()  # Xóa biểu đồ 1
        self.ax2.clear()  # Xóa biểu đồ 2
        self.ax3.clear()  # Xóa biểu đồ 3
        self.stats_info_text.delete("1.0", tk.END)  # Xóa thông tin
        self.canvas.draw()  # Cập nhật giao diện
        self.history_text.delete("1.0", tk.END)
        logging.info("Cleared statistics data")
    
    # Hàm cập nhật thống kê: Vẽ biểu đồ và hiển thị thông tin trong tab Thống kê

    def export_stats_pdf(self):
        if not hasattr(self.huffman, 'original_size') or not self.frequency:
            messagebox.showwarning("Cảnh báo", _("no_stats"))
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")], title="Lưu báo cáo PDF")
        if not output_path:
            return

        try:
            c = canvas.Canvas(output_path, pagesize=A4)
            c.setFont("Helvetica", 12)
            c.drawString(100, 800, "BÁO CÁO THỐNG KÊ NÉN DỮ LIỆU HUFFMAN")
            c.drawString(100, 780, f"Kích thước gốc: {self.huffman.original_size} bytes")
            c.drawString(100, 760, f"Kích thước sau nén: {self.huffman.compressed_size} bytes")
            ratio = (1 - self.huffman.compressed_size / self.huffman.original_size) * 100
            c.drawString(100, 740, f"Tỷ lệ nén: {ratio:.2f}%")
            c.drawString(100, 720, f"Thời gian nén: {self.huffman.encoding_time:.6f} giây")
            c.drawString(100, 700, f"Thời gian giải nén: {self.huffman.decoding_time:.6f} giây")

            c.drawString(100, 670, "Bảng tần suất ký tự (Top 10):")
            sorted_freq = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            y = 650
            for char, freq in sorted_freq:
                c.drawString(120, y, f"'{char}': {freq}")
                y -= 20

            c.save()
            messagebox.showinfo("Thành công", _("success_pdf").format(output_path))
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất báo cáo PDF: {str(e)}")

    def update_statistics(self, frequency):
        self.clear_stats()  # Xóa dữ liệu cũ
        
        # Biểu đồ 1: So sánh kích thước trước và sau khi nén
        sizes = ['Kích thước gốc', 'Kích thước nén']
        values = [self.huffman.original_size, self.huffman.compressed_size]
        self.ax1.bar(sizes, values, color=['#3498db', '#2ecc71'])
        self.ax1.set_title('So sánh kích thước')
        self.ax1.set_ylabel('Bytes')
        
        # Biểu đồ 2: Tần suất 10 ký tự phổ biến nhất
        sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        chars, freqs = [], []
        for char, freq in sorted_freq:
            if freq > 0:
                chars.append(char)
                freqs.append(freq)
        self.ax2.bar(chars, freqs, color='#9b59b6')
        self.ax2.set_title('10 ký tự phổ biến nhất')
        self.ax2.set_ylabel('Tần suất')
        self.ax2.tick_params(axis='x', rotation=45)
        
        # Biểu đồ 3: Phân bố tần suất ký tự (biểu đồ tròn)
        total_freq = sum(freqs)
        if total_freq > 0:
            percentages = [freq / total_freq * 100 for freq in freqs]
            self.ax3.pie(percentages, labels=chars, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(np.arange(len(chars))))
            self.ax3.set_title('Phân bố tần suất ký tự')
        
        self.fig.tight_layout()  # Điều chỉnh bố cục biểu đồ
        self.canvas.draw()  # Vẽ lại giao diện
        
        # Hiển thị thông tin chi tiết
        self.stats_info_text.delete("1.0", tk.END)
        self.stats_info_text.insert(tk.END, "THÔNG TIN CHI TIẾT VỀ QUÁ TRÌNH NÉN\n\n")
        self.stats_info_text.insert(tk.END, f"Kích thước dữ liệu gốc: {self.huffman.original_size} bytes\n")
        self.stats_info_text.insert(tk.END, f"Kích thước sau khi nén: {self.huffman.compressed_size} bytes\n")
        compression_ratio = (1 - self.huffman.compressed_size / self.huffman.original_size) * 100
        self.stats_info_text.insert(tk.END, f"Tỷ lệ nén: {compression_ratio:.2f}%\n")
        self.stats_info_text.insert(tk.END, f"Thời gian nén: {self.huffman.encoding_time:.6f} giây\n")
        self.stats_info_text.insert(tk.END, f"Thời gian giải nén: {self.huffman.decoding_time:.6f} giây\n")
        
        if hasattr(self, 'history_text'):
            self.history_text.delete("1.0", tk.END)
            for i, record in enumerate(self.history[-5:], 1):
                if record['action'] == 'compress':
                    self.history_text.insert(tk.END, f"[{i}] Nén: {record['original_size']}B → {record['compressed_size']}B, {record['time']:.6f} giây\n")
                elif record['action'] == 'decompress':
                    self.history_text.insert(tk.END, f"[{i}] Giải nén: {record['time']:.6f} giây\n")

        logging.info("Updated statistics tab")

# Chạy ứng dụng

    def create_language_menu(self):
        menubar = tk.Menu(self)
        language_menu = tk.Menu(menubar, tearoff=0)
        language_menu.add_command(label="Tiếng Việt", command=lambda: self.set_language('vi'))
        language_menu.add_command(label="English", command=lambda: self.set_language('en'))
        menubar.add_cascade(label="Ngôn ngữ / Language", menu=language_menu)
        self.config(menu=menubar)

    def set_language(self, lang_code):
        global current_lang
        current_lang = lang_code
        messagebox.showinfo(
            "Thông báo",
            f"Đã chuyển sang ngôn ngữ: {'Tiếng Việt' if lang_code == 'vi' else 'English'}.\nVui lòng khởi động lại ứng dụng để áp dụng."
        )

# Chạy ứng dụng
if __name__ == "__main__":
    app = HuffmanGUI()
    app.mainloop()