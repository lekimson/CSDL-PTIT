import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
import heapq
from collections import defaultdict
import time
import sys
import json
import os
from math import log2

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoding:
    def __init__(self):
        self.root = None
        self.codes = {}

    def calculate_frequency(self, data):
        frequency = defaultdict(int)
        for char in data:
            frequency[char] += 1
        return frequency

    def build_huffman_tree(self, frequency):
        heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
        heapq.heapify(heap)

        if len(heap) == 1:
            node = heapq.heappop(heap)
            new_node = HuffmanNode(None, node.freq)
            new_node.left = node
            heapq.heappush(heap, new_node)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)

        self.root = heap[0]
        return self.root

    def generate_codes(self, node, current_code="", codes=None):
        if codes is None:
            codes = {}
        if node is None:
            return codes
        if node.char is not None:
            codes[node.char] = current_code if current_code else "0"
        codes = self.generate_codes(node.left, current_code + "0", codes)
        codes = self.generate_codes(node.right, current_code + "1", codes)
        return codes

    def encode(self, data):
        frequency = self.calculate_frequency(data)
        self.build_huffman_tree(frequency)
        self.codes = self.generate_codes(self.root)
        return "".join(self.codes[char] for char in data), self.codes

    def decode(self, encoded_data):
        decoded = ""
        current = self.root
        for bit in encoded_data:
            current = current.left if bit == "0" else current.right
            if current.left is None and current.right is None:
                decoded += current.char
                current = self.root
        return decoded

    def compression_ratio(self, original, encoded):
        if len(original) == 0:
            return 0
        return 100 * (1 - len(encoded) / (len(original) * 8))

    def encrypt(self, text, key='X'):
        return ''.join(chr(ord(c) ^ ord(key)) for c in text)

    def decrypt(self, text, key='X'):
        return ''.join(chr(ord(c) ^ ord(key)) for c in text)

    def encode_file(self, file_path):
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        data_str = ''.join(format(byte, '08b') for byte in binary_data)
        encoded_data, codes = self.encode(data_str)
        return encoded_data, codes, len(data_str), binary_data

    def decode_file(self, encoded_data, codes, original_len):
        reverse_codes = {v: k for v, k in codes.items()}
        current = ""
        decoded_bits = ""
        for bit in encoded_data:
            current += bit
            if current in reverse_codes:
                decoded_bits += reverse_codes[current]
                current = ""
        decoded_bits = decoded_bits[:original_len]
        byte_data = bytearray()
        for i in range(0, len(decoded_bits), 8):
            byte = decoded_bits[i:i+8]
            byte_data.append(int(byte, 2))
        return bytes(byte_data)

class HuffmanApp:
    def __init__(self, root):
        self.huff = HuffmanCoding()
        self.root = root
        self.root.title("Huffman Coding GUI Full")
        self.root.geometry("950x850")

        self.input_text = scrolledtext.ScrolledText(root, height=5, width=80)
        self.input_text.pack(pady=10)

        self.encode_btn = ttk.Button(root, text="Mã hóa văn bản", command=self.encode)
        self.encode_btn.pack()

        self.encrypt_btn = ttk.Button(root, text="Mã hóa + Bảo mật XOR", command=self.encode_with_encryption)
        self.encrypt_btn.pack()

        self.encoded_label = tk.Label(root, text="Dữ liệu mã hóa:")
        self.encoded_label.pack()
        self.encoded_text = scrolledtext.ScrolledText(root, height=4, width=80)
        self.encoded_text.pack()

        self.decode_btn = ttk.Button(root, text="Giải mã văn bản", command=self.decode)
        self.decode_btn.pack(pady=5)

        self.decrypt_btn = ttk.Button(root, text="Giải mã từ XOR", command=self.decode_with_decryption)
        self.decrypt_btn.pack()

        self.file_btn = ttk.Button(root, text="Nén file bất kỳ", command=self.compress_file)
        self.file_btn.pack(pady=5)

        self.decompress_btn = ttk.Button(root, text="Giải nén file .huff", command=self.decompress_file)
        self.decompress_btn.pack(pady=5)

        self.decoded_label = tk.Label(root, text="Dữ liệu sau giải mã:")
        self.decoded_label.pack()
        self.decoded_text = scrolledtext.ScrolledText(root, height=4, width=80)
        self.decoded_text.pack()

        self.tree_canvas = tk.Canvas(root, width=800, height=200, bg="white")
        self.tree_canvas.pack(pady=10)

        self.stats_label = tk.Label(root, text="", font=("Arial", 10), fg="blue")
        self.stats_label.pack()

    def encode(self):
        data = self.input_text.get("1.0", tk.END).strip()
        if not data:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập dữ liệu văn bản.")
            return
        start = time.time()
        encoded_data, codes = self.huff.encode(data)
        end = time.time()
        self.encoded_text.delete("1.0", tk.END)
        self.encoded_text.insert(tk.END, encoded_data)

        self.last_input = data
        self.last_encoded = encoded_data
        self.last_codes = codes

        ratio = self.huff.compression_ratio(data, encoded_data)
        entropy = self.calculate_entropy(data)
        avg_code_len = sum(len(self.last_codes[char]) * freq for char, freq in self.huff.calculate_frequency(data).items()) / len(data)
        self.stats_label.config(text=f"Thời gian mã hóa: {end - start:.6f} giây | Tỷ lệ nén: {ratio:.2f}% | Entropy: {entropy:.4f} | Độ dài mã TB: {avg_code_len:.4f}")
        self.show_tree()

    def encode_with_encryption(self):
        data = self.input_text.get("1.0", tk.END).strip()
        if not data:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập dữ liệu văn bản.")
            return
        key = simpledialog.askstring("Nhập khóa", "Nhập ký tự khóa XOR:", parent=self.root)
        if not key:
            key = 'X'
        encoded_data, codes = self.huff.encode(data)
        encrypted_data = self.huff.encrypt(encoded_data, key=key)
        self.encoded_text.delete("1.0", tk.END)
        self.encoded_text.insert(tk.END, encrypted_data)
        self.last_input = data
        self.last_encoded = encoded_data
        self.last_encrypted = encrypted_data
        self.last_codes = codes
        self.last_key = key
        ratio = self.huff.compression_ratio(data, encoded_data)
        self.stats_label.config(text=f"Tỷ lệ nén + mã hóa: {ratio:.2f}%")

    def decode(self):
        encoded_data = self.encoded_text.get("1.0", tk.END).strip()
        if not encoded_data:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu mã hóa để giải mã.")
            return
        decoded = self.huff.decode(encoded_data)
        self.decoded_text.delete("1.0", tk.END)
        self.decoded_text.insert(tk.END, decoded)

    def decode_with_decryption(self):
        encrypted_data = self.encoded_text.get("1.0", tk.END).strip()
        if not encrypted_data:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu mã hóa để giải mã.")
            return
        key = simpledialog.askstring("Nhập khóa", "Nhập ký tự khóa XOR:", parent=self.root)
        if not key:
            key = 'X'
        decrypted = self.huff.decrypt(encrypted_data, key=key)
        decoded = self.huff.decode(decrypted)
        self.decoded_text.delete("1.0", tk.END)
        self.decoded_text.insert(tk.END, decoded)

    def compress_file(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        encoded_data, codes, bit_length, original_data = self.huff.encode_file(file_path)
        save_path = filedialog.asksaveasfilename(defaultextension=".huff")
        if not save_path:
            return
        with open(save_path, 'w') as f:
            json.dump({"encoded": encoded_data, "codes": codes, "original_len": bit_length}, f)
        self.log_action("Nén", file_path, bit_length, len(encoded_data))
        messagebox.showinfo("Nén file", f"Đã nén file thành công. Dữ liệu ban đầu: {bit_length} bit, sau nén: {len(encoded_data)} bit.")

    def decompress_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Huffman Compressed Files", "*.huff")])
        if not file_path:
            return
        with open(file_path, 'r') as f:
            data = json.load(f)
        encoded_data = data["encoded"]
        codes = data["codes"]
        original_len = data["original_len"]
        output_data = self.huff.decode_file(encoded_data, codes, original_len)

        # Nếu là text, hiển thị thẳng ra
        try:
            text_output = output_data.decode('utf-8')
            self.decoded_text.delete("1.0", tk.END)
            self.decoded_text.insert(tk.END, text_output)
            messagebox.showinfo("Giải nén", "File đã được giải nén và hiển thị! (dữ liệu văn bản)")
        except:
            # Nếu không phải text, lưu ra file nhị phân
            save_path = filedialog.asksaveasfilename(defaultextension=".bin")
            if not save_path:
                return
            with open(save_path, 'wb') as f:
                f.write(output_data)
            messagebox.showinfo("Giải nén", "File đã được giải nén thành công (file nhị phân)!")

    def calculate_entropy(self, data):
        freq = self.huff.calculate_frequency(data)
        total = len(data)
        return -sum((count / total) * log2(count / total) for count in freq.values())

    def show_tree(self):
        self.tree_canvas.delete("all")
        if not self.huff.root:
            return
        self._draw_node(self.huff.root, 400, 20, 200)

    def _draw_node(self, node, x, y, offset):
        if node is None:
            return
        label = node.char if node.char else "*"
        self.tree_canvas.create_oval(x-15, y-15, x+15, y+15, fill="lightblue")
        self.tree_canvas.create_text(x, y, text=label)
        if node.left:
            self.tree_canvas.create_line(x, y+15, x-offset, y+65)
            self._draw_node(node.left, x-offset, y+80, offset//2)
        if node.right:
            self.tree_canvas.create_line(x, y+15, x+offset, y+65)
            self._draw_node(node.right, x+offset, y+80, offset//2)

    def log_action(self, action, filename, original_bits, compressed_bits):
        with open("log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{action}: {filename} | Gốc: {original_bits} bit | Sau nén: {compressed_bits} bit\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = HuffmanApp(root)
    root.mainloop()
