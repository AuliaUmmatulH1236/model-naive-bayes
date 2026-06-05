import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

# ======================
# STEMMER
# ======================
# Inisialisasi Stemmer Sastrawi sekali saja di luar fungsi agar cepat
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# ======================
# PREPROCESSING
# ======================
df_norm = pd.read_excel("normalisasi.xlsx")

norm_dict = dict(zip(
    df_norm['kata awal'],
    df_norm['kata perbaikan']
))

STOPWORDS_ID = {
    "yang","dan","di","ke","dari","untuk","dengan",
    "adalah","pada","ini","itu","atau","ada",
    "saya","kami","anda","dia","mereka",
    "akan","telah","lebih","semua","setiap","pun",
    "ya","yaa","nih","sih","siih","dong","deh","dehh",
    "aja","lah","lahh","kah","mah","eh","oh","ooh","ohh",
    "kok","tuh","doang","kan","begitu","gitu",
    "k","d","n","an","la",
    "hehe","hehehe","xixi","xixixi",
    "wkwk","wkwkwkw",
    "btw",
    "and","or",
    "sat", "pas",
    "tisp","weh","woi","bro","sis","cuk","anjir","anjay",
    "lagi","masih"
}

PROTECTED_WORDS = {
    "wifi", "ok",
    "gacoan", "wizzmie", "ramen","sushi","gelato"
}

# ❗ JANGAN HAPUS NEGASI
NEGATION_WORDS = {"tidak", "bukan", "kurang", "belum", "jangan", "nggak", "ga", "gak", "gk", "tdk", "ngga", "kagak", "gx"}

try:
    STOPWORDS_EN = set(stopwords.words("english"))
except:
    STOPWORDS_EN = set()

ALL_STOPS = STOPWORDS_ID | STOPWORDS_EN

factory = StemmerFactory()
stemmer = factory.create_stemmer()

SKIP_NEGATION_WORDS = {
    "terlalu",
    "cukup",
    "lumayan",
    "agak"
}

def handle_negation(words):
    result = []
    negating = False
    window = 0

    for w in words:
        if w in NEGATION_WORDS:
            negating = True
            window = 2
            result.append(w)
            continue

        if negating and window > 0:
            if w in SKIP_NEGATION_WORDS:
                result.append(w)
            else:
                result.append("NOT_" + w)
            window -= 1
        else:
            result.append(w)

        if window == 0:
            negating = False

    return result

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    for k, v in norm_dict.items():
        text = re.sub( rf'\b{re.escape(k)}\b', v, text )

    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()

    filtered_words = []
    for w in words:
        if w in ALL_STOPS and w not in NEGATION_WORDS:
            continue
        filtered_words.append(w)

    filtered_words = [w for w in filtered_words if len(w) > 2]
    words = handle_negation(filtered_words)

    clean_words = []
    for w in words:
        if w not in PROTECTED_WORDS and not w.startswith("NOT_"):
            w = stemmer.stem(w)
        clean_words.append(w)

    return " ".join(clean_words)


# ==================================================================
# ASPEK & SENTIMENT KEYWORDS (Kombinasi Spesifik Aspek)
# ==================================================================
ASPEK_KEYWORDS = {
    "makanan": [
        r"\b(rasa|enak|lezat|nikmat|hambar|gurih|manis|pedas|asam|asin|sambal|pahit)\b",
        r"\b(tasteless|delicious|tasty|yummy|flavou?|minum\w*|jus|kopi|teh|es)\b",
        r"\b(mantap|mantul|sedap|nampol|nagih|ketagihan)\b",
        r"\b(bumbu|rempah|cita\s*rasa|bland)\b",
        r"\b(makanan\w*|menu\w*)\b\s*(enak|lezat|mantap|sedap|juara|top|pedas|kurang|biasa|variatif|banyak)"
    ],
    "layanan": [
        r"\b(pelayan\w*|service|staff|staf|karyawan|waiter|waitress|mbak|mas)\b",
        r"\b(pelayan|kasir|staf|karyawan|service)\b\s*(ramah|cepat|lambat|lama|jutek)",
        r"\b(ramah|jutek|lambat|lama|cepat|cepet|responsif|respon)\b",
        r"\b(antri|antre|antrian|tunggu|nunggu|lelet)\b",
        r"\b(order\w*|pesan\w*)\b.{0,15}\b(lama|cepat|lambat)\b"
    ],
    "harga": [
        r"\b(harga|mahal|murah|affordable|worth|terjangkau|murmer|miring)\b",
        r"\b(price|budget|value|pricy|pricey|overprice|kemahalan)\b",
        r"\b(rupiah|ribu|rp\s?\d+|duit|uang|bayar|biaya|cost)\b",
        r"\b(worth\s*it|pas\s*di\s*kantong|ramah\s*kantong|kantong\s*(pelajar|mahasiswa))\b"
    ],
    "lingkungan": [
        r"\b(fasilitas\w*|tempat\w*|lokasi\w*|suasana\w*|atmosphere|ambience)\b",
        r"\b(bersih|kotor|jorok|nyaman|sempit|luas|adem|panas|dingin|pengap|gerah)\b",
        r"\b(parkir|toilet|ac|wifi|dekor\w*|interior|outdoor)\b",
        r"\b(meja|kursi|duduk|view|pemandangan)\b",
        r"\b(strategis|gampang\s*dijangkau)\b"
    ]
}

# Kamus kata sifat dasar untuk pemetaan silang kepuasan per aspek
LEXICON_SENTIMEN = {
    "positif": [
        r"(?<!NOT_)\b(enak|lezat|mantap|mantul|sedap|nikmat|juara|top)\b",
        r"(?<!NOT_)\b(murah|terjangkau|worth|affordable|ramah\s*kantong)\b",
        r"(?<!NOT_)\b(ramah|cepat|responsif|baik|sopan)\b",
        r"(?<!NOT_)\b(nyaman|bersih|sejuk|adem|luas|enjoy|suka|love|recommended|indah|estetik|aesthetic)\b"
    ],
    "negatif": [
        r"\bNOT_(enak|lezat|mantap|ramah|nyaman|bagus)\b",
        r"\b(tidak|tdk|nggak|gak|ga|ngga|kagak|bukan|jangan)\b\s*(enak|ramah|bagus)?",
        r"\b(kecewa|mengecewakan|buruk|jelek|parah|worst|rugi|menyesal|nyesel|kapok|zonk|scam)\b",
        r"\b(hambar|tasteless|bland|basi|busuk|benyek|keras|alot)\b",
        r"\b(lambat|lelet|lama\s*banget|kelamaan)\b",
        r"\b(mahal|mahal\s*banget|kemahalan|overprice|terlalu\s*mahal)\b", 
        r"\b(kotor|jorok|bau|pengap|gerah|sumpek|sempit|pelit|sedikit)\b",
        r"\b(jutek|galak|sinis|cuek|tidak\s*ramah|gak\s*sopan)\b",
        r"\b(tidak\s*recommended|not\s*recommended|avoid|jangan\s*kesini)\b"
    ],
    "netral": [
        r"\b(biasa\s*aja|lumayan|so\s*so|standar|cukup|not\s*bad|ya\s*begitu|gitu\s*aja|oke\s*lah)\b"
    ]
}

# 🛠️ UBAHAN UTAMA: Fungsi leksikon baru yang mengikat Aspek + Sentimen bersamaan
def keyword_features_smart(text):
    text = text.lower()
    words = text.split()
    
    # 1. Identifikasi aspek dasar yang muncul
    aspek_terdeteksi = []
    for asp, patterns in ASPEK_KEYWORDS.items():
        for p in patterns:
            if re.search(p, text):
                aspek_terdeteksi.append(asp)
                break
                
    # 2. Windowing dan modifikasi kata langsung (Asimilasi Token)
    for asp in aspek_terdeteksi:
        patterns_aspek = ASPEK_KEYWORDS[asp]
        
        # Cari indeks kata aspek di dalam list kata
        for i, word in enumerate(words):
            is_aspek_word = False
            for p in patterns_aspek:
                if re.search(p, word):
                    is_aspek_word = True
                    break
            
            if is_aspek_word:
                # Cek radius 4 kata di sekitarnya untuk mencari sentimen
                start = max(0, i - 4)
                end = min(len(words), i + 5)
                window_text = " ".join(words[start:end])
                
                # Cek polaritas sentimen dalam window tersebut
                for pol, patterns_sent in LEXICON_SENTIMEN.items():
                    for p in patterns_sent:
                        if re.search(p, window_text):
                            # 🔥 UBAH KATA ASPEK MENJADI TOKEN UTUH: contoh "tempat" -> "tempat_negatif"
                            words[i] = f"{word}_{pol}"
                            break
    
    # Kembalikan teks asli yang kata aspeknya sudah ditempeli status sentimen
    processed_text = " ".join(words)
    
    # Tambahkan tag bayangan di akhir untuk memperkuat Stage 1 & Stage 2
    tag_aspek = " ".join([f"ASPEK_{a}" for a in aspek_terdeteksi])
    
    return processed_text, tag_aspek