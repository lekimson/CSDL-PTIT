import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
import heapq
from collections import defaultdict
import time
import sys
import json
import os
from math import log2

class HuffmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Huffman Coding GUI")
        self.root.geometry("1100x950")
        self.root.configure(bg="#eaf4fb")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, background="#3498db", foreground="white")
        style.configure("TLabel", background="#eaf4fb", font=("Segoe UI", 11))

        title = tk.Label(root, text="✨ Mã hóa Huffman - Giao diện trực quan ✨", font=("Segoe UI", 20, "bold"), bg="#eaf4fb", fg="#2c3e50")
        title.pack(pady=15)

        frame = ttk.Frame(root, padding=10)
        frame.pack(pady=10)

        ttk.Label(frame, text="📥 Nhập dữ liệu:").grid(row=0, column=0, sticky="w")
        self.input_text = scrolledtext.ScrolledText(frame, height=5, width=100, font=("Consolas", 10))
        self.input_text.grid(row=1, column=0, columnspan=3, pady=5)

        self.encode_btn = ttk.Button(frame, text="🔐 Mã hóa văn bản", command=self.encode)
        self.encode_btn.grid(row=2, column=0, padx=5, pady=5)

        self.encrypt_btn = ttk.Button(frame, text="🔒 Mã hóa + Bảo mật XOR", command=self.encode_with_encryption)
        self.encrypt_btn.grid(row=2, column=1, padx=5, pady=5)

        self.decode_btn = ttk.Button(frame, text="🔓 Giải mã văn bản", command=self.decode)
        self.decode_btn.grid(row=3, column=0, padx=5, pady=5)

        self.decrypt_btn = ttk.Button(frame, text="🔑 Giải mã từ XOR", command=self.decode_with_decryption)
        self.decrypt_btn.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(frame, text="📦 Dữ liệu mã hóa:").grid(row=4, column=0, sticky="w")
        self.encoded_text = scrolledtext.ScrolledText(frame, height=5, width=100, font=("Consolas", 10))
        self.encoded_text.grid(row=5, column=0, columnspan=3, pady=5)

        ttk.Label(frame, text="📤 Dữ liệu sau giải mã:").grid(row=6, column=0, sticky="w")
        self.decoded_text = scrolledtext.ScrolledText(frame, height=5, width=100, font=("Consolas", 10))
        self.decoded_text.grid(row=7, column=0, columnspan=3, pady=5)

        self.file_btn = ttk.Button(frame, text="🗂️ Nén file bất kỳ", command=self.compress_file)
        self.file_btn.grid(row=8, column=0, padx=5, pady=5)

        self.decompress_btn = ttk.Button(frame, text="📁 Giải nén file .huff", command=self.decompress_file)
        self.decompress_btn.grid(row=8, column=1, padx=5, pady=5)

        self.compare_btn = ttk.Button(frame, text="📊 So sánh Huffman với RLE", command=self.compare_with_rle)
        self.compare_btn.grid(row=9, column=0, padx=5, pady=5)

        self.test_btn = ttk.Button(frame, text="🧪 Kiểm thử tự động (mẫu tự sinh)", command=self.auto_test_samples)
        self.test_btn.grid(row=9, column=1, padx=5, pady=5)

        self.stats_label = ttk.Label(root, text="", font=("Segoe UI", 10, "italic"), foreground="#2980b9")
        self.stats_label.pack(pady=10)

        ttk.Label(root, text="🌳 Cây mã Huffman trực quan:", font=("Segoe UI", 12, "bold")).pack(pady=(10, 5))
        self.tree_canvas = tk.Canvas(root, width=1000, height=260, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.tree_canvas.pack(pady=10)

        self.language = 'vi'
        self.labels = {
            'vi': {
                'title': "✨ Mã hóa Huffman - Giao diện trực quan ✨",
                'lang_btn': "🇬🇧 English",
                'encode': "🔐 Mã hóa văn bản",
                'encrypt': "🔒 Mã hóa + Bảo mật XOR",
                'decode': "🔓 Giải mã văn bản",
                'decrypt': "🔑 Giải mã từ XOR",
                'compress': "🗂️ Nén file bất kỳ",
                'decompress': "📁 Giải nén file .huff",
                'compare': "📊 So sánh Huffman với RLE",
                'test': "🧪 Kiểm thử tự động (mẫu tự sinh)",
                'footer': "© Đề tài đồ án Python - Nhóm Huffman"
            },
            'en': {
                'title': "✨ Huffman Encoding - Visual Interface ✨",
                'lang_btn': "🇻🇳 Tiếng Việt",
                'encode': "🔐 Encode Text",
                'encrypt': "🔒 Encode + XOR Encryption",
                'decode': "🔓 Decode Text",
                'decrypt': "🔑 Decrypt from XOR",
                'compress': "🗂️ Compress Any File",
                'decompress': "📁 Decompress .huff File",
                'compare': "📊 Compare Huffman vs RLE",
                'test': "🧪 Auto Test Samples",
                'footer': "© Python Project - Huffman Team"
            }
        }

        self.lang_button = ttk.Button(root, text=self.labels['vi']['lang_btn'], command=self.toggle_language)
        self.lang_button.pack(pady=(0, 10))

        self.footer_label = ttk.Label(root, text=self.labels['vi']['footer'], font=("Segoe UI", 9), foreground="#7f8c8d")
        self.footer_label.pack(pady=10)

        def toggle_language(self):
            self.language = 'en' if self.language == 'vi' else 'vi'
            lang = self.labels[self.language]
            self.title_label.config(text=lang['title'])
            self.lang_button.config(text=lang['lang_btn'])
            self.encode_btn.config(text=lang['encode'])
            self.encrypt_btn.config(text=lang['encrypt'])
            self.decode_btn.config(text=lang['decode'])
            self.decrypt_btn.config(text=lang['decrypt'])
            self.file_btn.config(text=lang['compress'])
            self.decompress_btn.config(text=lang['decompress'])
            self.compare_btn.config(text=lang['compare'])
            self.test_btn.config(text=lang['test'])
            self.footer_label.config(text=lang['footer'])

        ttk.Label(root, text="© Đề tài đồ án Python - Nhóm Huffman", font=("Segoe UI", 9), foreground="#7f8c8d").pack(pady=10)
