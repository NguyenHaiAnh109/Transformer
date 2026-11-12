import tiktoken

# 1. Khởi tạo tokenizer
enc = tiktoken.get_encoding("r50k_base")

# 2. Văn bản đầu vào
text = "Once upon a time"

# 3. Chuyển text thành token IDs
token_ids = enc.encode(text)

print("Original text:", text)
print("Token IDs:", token_ids)

# 4. Lấy ID của token đặc biệt
eot_id = enc.eot_token

print("End of text ID:", eot_id)

# 5. Thêm token kết thúc tài liệu
token_ids.append(eot_id)

print("Final tokens:", token_ids)

# 6. Decode để kiểm tra
decoded = enc.decode(token_ids)

print("Decoded:", repr(decoded))