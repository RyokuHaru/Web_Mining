# Music Flow

Ứng dụng khai thác dữ liệu âm nhạc và gợi ý bài hát theo nội dung lời nhạc, cảm xúc cộng đồng, cấu trúc đồ thị và đặc trưng âm thanh.

Repo được chia thành ba flow. Flow 1 (DUY) xử lý lyrics và tìm kiếm ngữ nghĩa; Flow 2 (HOA) thu thập bình luận YouTube và tính cảm xúc cộng đồng; Flow 3 (TÙNG) xây dựng đồ thị bài hát.

## Cấu trúc repo

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── config.py
├── data/
│   ├── raw/                         # Input cục bộ; không commit dữ liệu riêng tư
│   └── processed/                   # Kết quả pipeline; không commit artifacts
├── src/
│   ├── data_ingestion/
│   │   ├── spotify_crawler.py       # Lấy metadata/audio features (khung cho flow sau)
│   │   ├── genius_crawler.py        # Lấy lyrics từ Genius (Flow 1)
│   │   └── youtube_crawler.py       # Lấy metadata video và bình luận
│   ├── flow1_content_mining/
│   │   ├── ner_extractor.py         # Trích xuất artist, writer, genre
│   │   ├── tfidf_baseline.py        # Baseline TF-IDF và tìm kiếm cosine
│   │   ├── sbert_encoder.py         # SBERT embeddings, semantic search, pairwise similarity
│   │   ├── search.py                 # Lọc metadata và semantic search cho ứng dụng
│   │   └── pipeline.py              # Chạy Flow 1 và ghi artifacts
│   ├── flow2_social_mining/
│   │   ├── sentiment_analyzer.py    # RoBERTa emotion và valence
│   │   └── pipeline.py              # Phân tích comments và lưu kết quả
│   ├── flow3_structure_mining/      # Dành cho Flow 3 (TÙNG)
│   └── utils/
├── app/
│   └── components/                  # Giao diện Streamlit sẽ được bổ sung sau
└── notebooks/                       # Thử nghiệm và báo cáo
```

Flow 3, giao diện, Spotify crawler và notebooks vẫn là khung theo kiến trúc. Flow 1 và Flow 2 đã có pipeline riêng.

## Thiết lập

Yêu cầu Python 3.10 trở lên.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Tạo `.env` từ `.env.example` và điền thông tin API khi cần chạy crawler. Không đưa `.env`, API keys hoặc dữ liệu lyrics có bản quyền lên GitHub.

## Chạy Flow 2 (HOA)

Flow 2 cần ghép mỗi video YouTube với `song_id` từ `data/raw/songs.json`. Tạo `data/raw/youtube_video_map.csv`:

```csv
song_id,youtube_url
2476188,https://www.youtube.com/watch?v=VIDEO_ID
```

Thay `2476188` bằng ID thật trong file Flow 1 và điền URL video đúng bài đó. Một bài có thể có nhiều dòng video. Collector lấy metadata, comment và giữ `song_id` trên từng comment:

```bash
python -m src.data_ingestion.youtube_crawler \
  --urls-file data/raw/youtube_video_map.csv \
  --songs-file data/raw/songs.json \
  --output-dir data/raw \
  --max-comments 500 \
  --sort-by recent
```

Mặc định collector lấy tối đa 500 bình luận mới nhất mỗi video. Dùng `--sort-by popular` để ưu tiên bình luận phổ biến, hoặc `--max-comments 1000` để tăng giới hạn. Video lỗi được ghi vào `youtube_crawl_errors.json`; video khác vẫn tiếp tục được xử lý.

Phân tích cảm xúc và tổng hợp theo bài hát:

```bash
python -m src.flow2_social_mining.pipeline \
  --input data/raw/youtube_comments.jsonl \
  --songs-input data/raw/songs.json \
  --videos-input data/raw/youtube_videos.json \
  --output-dir data/processed
```

Flow 2 tạo `youtube_comments_scored.jsonl` (nhãn, điểm từng cảm xúc và valence mỗi comment), `youtube_video_valence.json` (tổng hợp từng video) và `social_valence.json` (tổng hợp từng bài hát, đúng định dạng để app dùng). Valence comment được tính bằng tổng xác suất cảm xúc tích cực trừ tổng xác suất tiêu cực; điểm bài hát là trung bình valence comment trong khoảng `[-1, 1]`. Bài không có comment được ghi với `social_valence_score: null`, không giả định là trung tính.

Model mặc định `BaoNhan/xlm-roberta-base-UITVSMEC` được huấn luyện trên văn bản mạng xã hội tiếng Việt. Nếu comment chủ yếu bằng tiếng Anh hoặc ngôn ngữ khác, hãy đổi `HF_SENTIMENT_MODEL` trong `.env` sang model phù hợp và đánh giá lại trên comment của project trước khi dùng điểm để xếp hạng.

## Chạy Flow 1

Đặt dữ liệu đầu vào ở `data/raw/songs.json` hoặc CSV tương đương. JSON mẫu:

```json
[
  {
    "song_id": "song-001",
    "title": "Example Song",
    "artist": "Example Artist",
    "writers": ["Writer One"],
    "genres": ["indie pop"],
    "lyrics": "Lyrics text goes here"
  }
]
```

Chạy pipeline:

```bash
python -m src.flow1_content_mining.pipeline \
  --input data/raw/songs.json \
  --output data/processed
```

Pipeline ghi ra:

- `song_features.csv`: ID, tiêu đề, artist, writers, genres và thực thể nhận diện từ lyrics.
- `tfidf_matrix.npz` cùng `tfidf_song_ids.json`: sparse vectors của baseline TF-IDF.
- `semantic_embeddings.npy` cùng `semantic_song_ids.json`: dense vectors SBERT theo đúng thứ tự ID.
- `semantic_vectors.jsonl`: một dòng mỗi bài, gồm metadata và `semantic_vector` để Flow 3 đọc trực tiếp.
- `semantic_similarity.csv`: ma trận cosine similarity giữa các bài hát; Flow 3 có thể dùng ma trận này để tạo cạnh (ví dụ ngưỡng `> 0.8`).

Mỗi dòng `semantic_vectors.jsonl` có `song_id`, `title`, `artist`, `writers`, `genres`, `vector_model`, `vector_dimension` và `semantic_vector`. Metadata nguồn như album, ngày phát hành, Genius URL, nguồn lyrics và điểm khớp LRCLIB được giữ lại nếu có. Lyrics không được chép vào file vector.

Mặc định SBERT dùng `sentence-transformers/all-MiniLM-L6-v2`. Có thể đổi bằng biến môi trường `SBERT_MODEL`. Pipeline yêu cầu trường `song_id` và `lyrics`; metadata còn thiếu được ghi thành giá trị rỗng. `search.py` cung cấp tìm kiếm theo ngữ nghĩa và bộ lọc artist/writer/genre. TF-IDF và SBERT tìm kiếm bằng cosine similarity, sắp xếp kết quả giảm dần.

### Lấy mẫu bài từ Genius và chạy Flow 1

Đặt `GENIUS_ACCESS_TOKEN` trong `.env` để tìm metadata qua Genius Developer API:

```bash
python -m src.flow1_content_mining.run_random --count 100
```

Lệnh này lấy metadata bài hát qua Genius Developer API, tìm lyrics khớp trên LRCLIB, lưu dữ liệu thô vào `data/raw/songs.json`, rồi tạo artifact ở `data/processed/`. Đây là mẫu từ kết quả tìm kiếm, không phải mẫu ngẫu nhiên trên toàn bộ Genius. LRCLIB có thể không có lời cho một số bài nên số lượng đầu ra có thể thấp hơn yêu cầu. Dữ liệu thô ghi `metadata_source`, `lyrics_source`, ID và URL của hai nguồn. Có thể truyền `--seed 42` để tái lập thứ tự mẫu trong cùng tập tìm kiếm. Không commit dữ liệu lyrics lên GitHub.

## Thành viên và phân công flow

| Flow | Phần việc | Người phụ trách |
|---|---|---|
| 1 | Lyrics, NER, TF-IDF, SBERT và semantic search | DUY |
| 2 | Phân tích cảm xúc bình luận YouTube | HOA |
| 3 | Dựng đồ thị và phát hiện cộng đồng Louvain | TÙNG |

Flow 1 xuất vector JSONL có định danh bài hát để Flow 3 dựng đồ thị; ma trận cosine CSV cũng được cung cấp để dùng trực tiếp nếu cần.

## Ghi chú dữ liệu và cấu hình

- `data/raw/` và `data/processed/` được giữ trong repo bằng `.gitkeep`; dữ liệu và artifacts bị Git bỏ qua.
- API keys được đọc từ biến môi trường, không lưu trực tiếp trong mã nguồn.
- NER dùng spaCy để nhận diện thực thể ứng viên trong lyrics; các trường artist/writers/genres có sẵn từ metadata được giữ làm nhãn đáng tin cậy. Genre được khớp từ bộ thuật ngữ phổ biến trong pipeline và có thể truyền thêm qua `NERExtractor` khi gọi từ Python.
- Lyrics cần được thu thập và sử dụng theo điều khoản của nguồn cung cấp.
