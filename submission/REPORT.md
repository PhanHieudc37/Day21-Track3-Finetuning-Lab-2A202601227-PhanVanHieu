# Lab 21 — Evaluation Report

**Họ tên**: Phan Văn Hiếu

**MSSV**: 2A202601227

**Ngày**: 21/08/2026

**Tier**: `T4`

**Base model**: `unsloth/Qwen3.5-4B`

**GPU thực tế**: Tesla T4, 14.6 GB VRAM

Mọi điểm số, thời gian train, peak VRAM và cấu hình run trong báo cáo này được chép từ các artefact trong `results/`. Phần nhận xét định tính cũng chỉ sử dụng các mẫu có trong `results/qualitative.json` và nhãn vàng tương ứng của tập eval đã đóng băng.

---

## 1. Setup

| Hạng mục | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage bốn trường |
| Train / validation | 225 / 25, split với seed 42 |
| `max_length` thực tế | 1024 theo profile T4 |
| Thống kê độ dài | p95 = 98 token; giá trị gợi ý theo phép đo = 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / `max_steps` | 2 / 30 |
| Precision | fp16 |
| Effective batch | 16 |

`max_length=1024` lớn hơn mức 256 được gợi ý từ p95. Tôi giữ nguyên profile T4 dùng chung của lab để không thay đổi cấu hình giữa bốn run; dữ liệu đo được có tối đa 101 token nên không có mẫu nào bị cắt. Đổi về 256 có thể giảm padding, nhưng muốn kết luận về hiệu năng hoặc VRAM sau thay đổi đó thì cần chạy lại đồng nhất cả bốn cấu hình.

**Template có giữ khối `<think>` không?** Có. `template_check.json` ghi `ok=true`, `open_tag_present=true`, `body_present=true` và kết luận “reasoning preserved — safe to train on traces”. Corpus hiện tại chỉ chứa JSON trần, vì vậy chỉ số trace ở phần phán quyết không phải thí nghiệm reasoning-trace collapse.

---

## 2. Mask proof (NB1)

| Kiểm tra | Kết quả |
|---|---:|
| `n_supervised` / `n_total` | 39 / 94 |
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |

Ba dòng đầu của đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn supervise chứa phần kết thúc khối suy luận rỗng của chat template, JSON đáp án và token kết thúc lượt assistant. Nó không chứa ticket “Alo shop, mình đặt balo laptop…”. Vì thế model nhìn thấy câu hỏi để conditioning nhưng không bị tối ưu để chép lại câu hỏi; phần chịu cross-entropy chính là câu trả lời có cấu trúc.

---

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3131.5 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1017.3 |
| (c) LoRA fine-tune | 0.9700 | 0.5889 | 1.0000 | 1443.3 |

Baseline (b) thực sự mạnh hơn (a): target tăng từ 0.0000 lên 0.7650, format tăng từ 0.0000 lên 1.0000, trong khi regression giữ nguyên 0.7578. Nó cũng nhanh hơn (a), nên đây là một mốc khó và hợp lệ chứ không phải prompt bị làm yếu để fine-tune dễ thắng. Hai baseline được đóng băng trên đủ 50 mẫu target và 15 mẫu regression, với `smoke_mode=false` và `eval_limit=null`.

Tôi không sửa `OPTIMIZED_PROMPT` sau NB2. SHA được đóng băng là `719e74d3b6232053`, khớp prompt trong mã nguồn. Vì vậy chênh lệch của fine-tune ở NB5 được so với đúng mốc đã biết trước khi train.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí | r | Trainable params | LR | Train loss | Target NB5 | Train (s) | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6269 | 0.9700 | 934.1 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5373 | 0.9700 | 820.4 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.0000 | 981.4 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.9400 | 1040.8 | 7.09 |

Cả bốn run dùng đúng 30 optimizer step. Mỗi đối chứng chỉ đổi một trục so với `correct`: `attn_only` đổi vị trí và dùng rank đã khớp ngân sách; `wrong_lr` chỉ giảm learning rate; `qlora` chỉ đổi base sang 4-bit. `attn_only` lệch 8,192 tham số, tương đương khoảng 0.025% so với `correct`, nhỏ hơn nhiều so với ngưỡng công bằng 5%.

### 4.1 — Vị trí so với rank

`attn_only` hòa `correct` trên target ở 0.9700 dù chỉ gắn vào q,v và nâng rank lên 283 để giữ cùng ngân sách. Theo train loss, `attn_only` lại đứng trên `correct` (0.5373 so với 0.6269), nhưng lợi thế proxy đó không tạo ra lợi thế target. Vì vậy thứ tự theo loss không hoàn toàn giống thứ tự theo năng lực tác vụ: loss nói `attn_only` thắng, còn target nói hòa. Kết quả trên miền triage hẹp này không chứng minh text-linear luôn hơn attention-only; nó chứng minh rằng tăng rank có thể bù đủ cho vị trí hẹp ở tác vụ này, đồng thời nhấn mạnh rằng phải khóa ngân sách và chấm bằng target trước khi kết luận vị trí nào tốt hơn.

### 4.2 — Learning rate sai

`wrong_lr` chỉ đổi LR từ 1e-4 thành 1e-5 nhưng final loss tăng từ 0.6269 lên 1.5704, chênh 0.9435. Trên thước đo tác vụ, nó rơi từ target 0.9700 và format 1.0000 xuống cả target lẫn format bằng 0.0000. Nếu không biết LR và chỉ nhìn một đường loss giảm dần, tôi có thể kết luận sai rằng LoRA thiếu rank, dữ liệu không học được hoặc cần thêm epoch. Đối chứng này cho thấy LR ở thang full fine-tune làm adapter gần như không dịch chuyển đủ trong ngân sách 30 step; chỉnh rank trước khi sửa LR sẽ điều trị sai nguyên nhân.

### 4.3 — QLoRA

QLoRA giảm peak VRAM từ 12.01 xuống 7.09 GB, tiết kiệm 4.92 GB, tương đương khoảng 41%. Đổi lại, target giảm từ 0.9700 xuống 0.9400, train loss tăng từ 0.6269 lên 0.7058, latency tăng từ 1443.3 lên 1821.2 ms/mẫu và thời gian train tăng từ 934.1 lên 1040.8 giây. Số đo ủng hộ khuyến nghị không dùng QLoRA làm mặc định cho model này khi chất lượng là ưu tiên: tiết kiệm bộ nhớ là thật nhưng có giá target và tốc độ. Tuy vậy, nếu giới hạn phần cứng khiến bản 16-bit không chạy được, target 0.9400 vẫn cho thấy QLoRA là một phương án kỹ thuật có thể cân nhắc thay vì kết luận nó vô dụng.

Xếp hạng đúng theo target là: `correct = attn_only` (0.9700), sau đó `qlora` (0.9400), cuối cùng `wrong_lr` (0.0000). Tôi không dùng final loss để thay thế thứ tự này.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy: FAILED**

`target Δ = +0.2050` · `regression Δ = -0.1689` · `valid_trace_rate = 0.0000`

Fine-tune thắng rõ baseline prompt tốt trên tác vụ đích: target tăng từ 0.7650 lên 0.9700, tức thêm 0.2050, và format vẫn giữ 1.0000. Tuy nhiên regression giảm từ 0.7578 xuống 0.5889, mức giảm 0.1689 lớn hơn rất nhiều tolerance 0.0200. Cổng yêu cầu đồng thời thắng target và không làm hỏng năng lực chung quá ngưỡng, nên verdict FAILED là đúng dù target rất cao. Đây không phải thất bại của phép đo; ngược lại, nó phát hiện chính xác trade-off mà chỉ nhìn train loss hoặc target sẽ bỏ sót. Model hiện tại không nên được triển khai như một trợ lý dùng chung vì có dấu hiệu quên năng lực ngoài miền. `valid_trace_rate=0.0000` không được diễn giải là reasoning collapse trong thí nghiệm này: dữ liệu huấn luyện là JSON trần và đánh giá target cũng yêu cầu JSON, không có reasoning trace thật để bảo toàn. Hướng tiếp theo là bổ sung replay dữ liệu phổ thông, giữ nguyên eval đã đóng băng và chạy lại toàn bộ phép so sánh.

---

## 6. Định tính — có cả ca thắng và ca thua

Notebook NB2 chỉ lưu điểm tổng hợp của baseline (b), không lưu prediction theo từng mẫu. Vì vậy cột (b) dưới đây ghi rõ giới hạn artefact thay vì dựng lại output không có bằng chứng. Ca thắng/thua của fine-tune được xác định trực tiếp bằng `ft_score` đã chấm với nhãn vàng: 1.00 là đúng cả bốn trường, 0.75 là sai một trường.

| # | Ticket rút gọn | Nhãn đúng | (b) optimized prompt | (c) fine-tune | Nhận xét |
|---:|---|---|---|---|---|
| 1 | Bình giữ nhiệt VN804124, chưa thấy tiền, khi nào tiện | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | Không lưu output từng mẫu; aggregate target 0.7650 | Dự đoán `urgency=trung_binh`; 3/4 trường đúng | ❌ FT thua nhãn vàng ở urgency |
| 2 | Nồi chiên DH249548, thiếu phụ kiện, khi nào tiện | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | Không lưu output từng mẫu; aggregate target 0.7650 | Dự đoán `urgency=trung_binh`; 3/4 trường đúng | ❌ FT thua nhãn vàng ở urgency |
| 3 | Áo khoác VN613097, bị lỗi, khi nào tiện | `san_pham_loi / thap / áo khoác gió / tich_cuc` | Không lưu output từng mẫu; aggregate target 0.7650 | Dự đoán `urgency=trung_binh`; 3/4 trường đúng | ❌ FT thua nhãn vàng ở urgency |
| 4 | Ốp lưng DH936478, shipper không gọi, hỏi cho biết thôi | `van_chuyen / thap / ốp lưng điện thoại / tich_cuc` | Không lưu output từng mẫu; aggregate target 0.7650 | `ft_score=1.00`, đúng cả bốn trường | ✅ FT thắng nhãn vàng |
| 5 | Ốp lưng DH734695, hỏi giá, mong phản hồi | `hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh` | Không lưu output từng mẫu; aggregate target 0.7650 | `ft_score=1.00`, đúng cả bốn trường | ✅ FT thắng nhãn vàng |

Mẫu chung của các ca fine-tune sai là urgency thấp. Sáu mẫu tệ nhất trong `qualitative.json` đều có `ft_score=0.75`; ba mẫu kiểm tra ở trên đều chứa dấu hiệu “khi nào tiện” nhưng model dự đoán `trung_binh` thay vì `thap`. Model đã học rất tốt intent, product và format, nhưng ranh giới giữa urgency thấp và trung bình vẫn bị lệch. Đây là lỗi có cấu trúc, không phải JSON ngẫu nhiên, nên hướng cải thiện hợp lý là tăng dữ liệu biên cho các cách diễn đạt urgency thấp và kiểm tra cân bằng nhãn, thay vì tăng rank một cách chung chung.

---

## 7. Kết luận và điều tôi học được

**Kết luận.** Tôi chưa nên deploy adapter này như một model thay thế tổng quát cho base. Về tác vụ ticket, fine-tune rất thành công: target đạt 0.9700, cao hơn baseline đã prompt tử tế 0.7650, format đạt 1.0000 và các lỗi còn lại tập trung chủ yếu ở urgency thấp. Tuy nhiên, mục tiêu triển khai không chỉ là thắng một bảng target. Regression giảm 0.1689, vượt xa tolerance 0.0200, nên cổng FAILED đã ngăn một quyết định deploy dễ bị đánh lừa bởi target cao. Nếu endpoint được cô lập hoàn toàn cho triage, không nhận câu hỏi phổ thông, adapter có thể đáng để thử nghiệm thêm trong môi trường kiểm soát; còn với trợ lý đa nhiệm, tôi sẽ chưa phát hành. Đòn bẩy lớn nhất quan sát được là cấu hình học và dữ liệu, không phải rank đơn lẻ: LR sai làm target rơi về 0.0000; mask đúng tạo nền tảng để model học JSON; replay thiếu khiến năng lực chung suy giảm; dữ liệu biên urgency thấp vẫn chưa đủ chắc. `attn_only` rank 283 không thắng `correct` trên target dù có loss thấp hơn, cho thấy tăng capacity không tự động mang lại năng lực tốt hơn. QLoRA tiết kiệm khoảng 41% VRAM nhưng giảm target và tăng latency, nên chỉ phù hợp khi giới hạn phần cứng thực sự chi phối. Bước tiếp theo phải giữ nguyên tập eval và prompt đóng băng, bổ sung replay cùng ví dụ urgency thấp, rồi chạy lại đúng bốn đối chứng với một ngân sách step chung.

**Ba điều tôi học được:**

1. Tôi phải giải mã token được supervise trước khi train. Hai boolean và preview trong `mask_proof.json` có giá trị hơn việc tin một cờ `assistant_only_loss` của thư viện.
2. Train loss có thể đảo thứ tự kết luận: `attn_only` có loss 0.5373 thấp hơn `correct` 0.6269 nhưng chỉ hòa target 0.9700. Thang đo tác vụ mới quyết định adapter nào thực sự tốt.
3. Fine-tune thắng prompt không đồng nghĩa có thể deploy. Target tăng 0.2050 nhưng regression giảm 0.1689, và cổng bốn nhóm là thứ đã phát hiện rủi ro đó.

**Nếu có thêm thời gian, tôi sẽ thử:** bổ sung replay dữ liệu phổ thông và tăng các mẫu biên có cụm “khi nào tiện”/“không vội”, giữ nguyên prompt cùng eval đã đóng băng, rồi chạy lại `correct` và toàn bộ ba đối chứng với đúng 30 step để xác nhận regression phục hồi mà target không giảm.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
