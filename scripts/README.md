# Скрипти

## 1. `validate_corpus.py`

Валидација на целиот корпус. Проверува структура на `metadata.json`, постоење на audio files, квалитет на текст, конзистентност помеѓу metadata и JSONL.

```bash
python scripts/validate_corpus.py
```

Output: `reports/validation_report.json`

Проверки:

- Структура на `metadata.json` - header со `govor` + chunk entries со задолжителни полиња
- Постоење и големина на WAV files (детектира missing и tiny audio < 5KB)
- Квалитет на текст - short text, few words, high `>>` marker ratio, music markers
- Конзистентност - `metadata.json` chunk count == `training_data.jsonl` line count
- Monotonic timestamps - `start_ms` секогаш расте

## 2. `clean_corpus.py`

Чистење на корпусот - брише chunks со music markers, празен текст, или премал audio, и отстранува `>>` speaker markers од текстот.

```bash
# Преглед на промени (dry-run)
python scripts/clean_corpus.py

# Запишување на промени
python scripts/clean_corpus.py --apply
```

Output: `reports/clean_report_dryrun.json` или `reports/clean_report_apply.json`

Правила за бришење:

- Chunks со `[музика]`, `[music]`, `♪`, `♫`, `[аплауз]` - music/applause markers
- Chunks со текст покус од 10 карактери
- Chunks со WAV помал од 5KB (тишина или корупција)

## 3. `fix_chunk_alignment.py`

Корекција на text-audio alignment за целиот корпус. YouTube auto-generated captions имаат timing кој е ~1-3 зборови пред вистинскиот говор, што значи текстот во `metadata.json` содржи зборови кои реално се слушаат во следниот chunk.

Скриптата работи во две фази:

### Фаза 1: `--analyze`

Ја пушта `faster-whisper` `small` моделот на секој WAV chunk со `word_timestamps=True`. Резултатите се кешираат per-video во `reports/alignment_cache/`. Resume-safe - прескокнува веќе обработени видеа.

```bash
# NVIDIA DLLs мора да бидат на PATH (ако се инсталирани преку pip)
export PATH=".venv/Lib/site-packages/nvidia/cublas/bin:.venv/Lib/site-packages/nvidia/cudnn/bin:.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin:$PATH"

# Анализа на целиот корпус
python scripts/fix_chunk_alignment.py --analyze --compute-type int8_float32

# Анализа на едно видео (за тестирање)
python scripts/fix_chunk_alignment.py --analyze --compute-type int8_float32 --video "1. Собранието"
```

### Фаза 2: `--apply`

Ги користи кешираните Whisper резултати за корекција на текстот:

1. Го реконструира целосниот транскрипт per-video (де-дупликација на overlapping текст)
2. Со fuzzy matching ги наоѓа точните граници на текстот за секој chunk
3. Ги запишува коригираните `metadata.json` и `training_data.jsonl`
4. Додава `speech_start_ms` и `speech_end_ms` полиња (пауза/дишење metadata)

```bash
# Преглед на промени
python scripts/fix_chunk_alignment.py --apply --dry-run

# Запишување на промени
python scripts/fix_chunk_alignment.py --apply
```

Output: `reports/alignment_report_dryrun.json` или `reports/alignment_report_applied.json`

### Dependencies

```bash
pip install faster-whisper tqdm
# За CUDA support (ако нема CUDA Toolkit инсталирано):
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## 4. `ocr_captions_mk.ipynb`

Colab notebook за екстракција на македонски текст од YouTube Shorts со embedded captions (титли вградени во видеото како overlay, не SRT). Користи EasyOCR со `rs_cyrillic` (српска кирилица) како примарен јазик, бидејќи македонски не е директно поддржан.

Pipeline: Download видео преку `yt-dlp` -> extract frames на секоја N секунди -> crop до caption region -> optional sharpening -> OCR -> deduplication преку `rapidfuzz` -> output timestamped captions.

```python
captions = extract_captions(
    youtube_url="https://youtube.com/shorts/...",
    top=0.75,        # горна граница на crop region (0-1)
    bottom=0.95,     # долна граница
    left=0.05,       # лева граница
    right=0.95,      # десна граница
    sharpen=True,    # grayscale + binary threshold за подобар OCR
    frame_every_seconds=0.5,       # колку често да вади frames
    similarity_threshold=85        # rapidfuzz threshold за дедупликација
)
```

Параметри за crop region:

- `top`/`bottom`/`left`/`right` - дефинираат правоаголник каде се наоѓаат титлите (вредности 0-1, процент од висина/ширина)
- Различни канали имаат различен positioning - потребно е подесување per-channel

Квалитет:

- Најдобро работи со голем жолт/бел текст на темна позадина
- Послабо со мал фонт, транспарентен background, или стилизирани титли
- Тестирано на: podcast shorts, Канал5 „На кавга со Иван", @trn_mk

### Dependencies

```bash
pip install easyocr rapidfuzz yt-dlp
```

## 5. `whisper_evaluation.ipynb`

Colab notebook за евалуација на Whisper fine-tuning. Pipeline: Mount Drive -> discover data -> stratified sample ~250 chunks (~2h) -> preprocess -> train `whisper-small` 500 steps -> evaluate WER per dialect.
