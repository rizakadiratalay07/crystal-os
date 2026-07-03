import requests
import json
import os
import re
import time
import html as html_lib

CIKTI_DOSYASI = "turkce_veri.json"
HEDEF_BOYUT_KB = 100
USER_AGENT = "TurkceVeriBot/1.0"
API_URL = "https://tr.wikipedia.org/w/api.php"

KATEGORI_SAYFALAR = {
    "Türkçe": [
        "Türkçe",
        "Türk alfabesi",
        "Türk Dil Kurumu",
        "Dil Devrimi",
        "Türkçe söz varlığı",
        "Türkçe ekler",
        "Çekim eki",
        "İyelik eki"
    ],
    "biyoloji": [
        "Biyoloji",
        "Memeliler",
        "Primat",
        "İnsan"
    ],
    "python": [
        "Python",
        "Pygame",
        "NumPy",
        "Django (yazılım)"
    ],
    "html": [
        "HTML",
        "CSS",
        "HTML5",
        "JavaScript"
    ],
    "programlama": [
        "Algoritma",
        "Veri yapısı",
        "Nesne yönelimli programlama",
        "Fonksiyonel programlama"
    ],
    "teknoloji": [
        "Yapay zeka",
        "Makine öğrenimi",
        "Derin öğrenme",
        "Doğal dil işleme",
        "Pekiştirmeli öğrenme"
    ],
    "veri_bilimi": [
        "Veri bilimi",
        "Büyük veri",
        "Veri madenciliği",
        "İstatistik"
    ],
    "ülke": [
        "Türkiye",
        "Azerbaycan",
        "Kuzey Kıbrıs Türk Cumhuriyeti"
    ],
    "dil": [
        "Türkçe",
        "Türk alfabesi",
        "Türk Dil Kurumu"
    ],
    "eğitim": [
        "Eğitim",
        "Türkçe eğitimi",
        "Matematik eğitimi"
    ]
}

DISLANACAK_BASLIKLAR = (
    "kaynak",
    "kaynakça",
    "kaynaklar",
    "dış bağlantı",
    "dış kaynak",
    "ayrıca bakınız",
    "notlar",
    "dipnot",
    "ek okuma",
    "daha fazla bilgi",
    "bibliyografya",
    "okuma önerileri",
)

BASLIK_DESENI = re.compile(r'^=+\s*(.+?)\s*=+$')

REFERANS_SATIR_DESENLERI = (
    re.compile(r'^\s*[↑^]'),
    re.compile(r'^\s*\d+\.\s*\^'),
    re.compile(r'Erişim tarihi', re.IGNORECASE),
    re.compile(r'\bISBN\b', re.IGNORECASE),
    re.compile(r'\bdoi\s*:', re.IGNORECASE),
    re.compile(r'^\s*http[s]?://'),
)


def baslik_dislanmali_mi(baslik):
    if not baslik:
        return False
    b = baslik.strip().lower()
    return any(b.startswith(k) for k in DISLANACAK_BASLIKLAR)


def referans_satiri_mi(satir):
    return any(desen.search(satir) for desen in REFERANS_SATIR_DESENLERI)


def tam_metni_kaynakcadan_kirp(metin):
    if not metin:
        return metin
    satirlar = metin.split('\n')
    sonuc = []
    for satir in satirlar:
        eslesme = BASLIK_DESENI.match(satir.strip())
        if eslesme:
            baslik = eslesme.group(1)
            if baslik_dislanmali_mi(baslik):
                break
        sonuc.append(satir)
    return '\n'.join(sonuc).strip()


def temizle(metin):
    metin = html_lib.unescape(metin)
    metin = re.sub(r'<[^>]+>', ' ', metin)
    metin = re.sub(r'\[\s*\d+\s*\]', '', metin)
    metin = re.sub(r'\[[^\]]*değiştir[^\]]*\]', '', metin)
    metin = re.sub(r'\[[^\]]*kaynağı değiştir[^\]]*\]', '', metin)
    metin = re.sub(r'\[\s*\]', '', metin)
    metin = re.sub(r'\b(Ana madde|Ayrıca bakınız|Bu madde|Başlığın diğer anlamları):?\s*[^.]*\.', '', metin, flags=re.IGNORECASE)
    metin = re.sub(r'\s*•\s*', ' ', metin)
    metin = re.sub(r'\s*·\s*', ' ', metin)
    metin = re.sub(r'[{};@]', ' ', metin)
    metin = re.sub(r'\s+', ' ', metin)
    return metin.strip()


def api_istek_yap(params, max_retries=5):
    headers = {"User-Agent": USER_AGENT}
    for deneme in range(max_retries):
        try:
            resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                bekleme_suresi = 30 * (deneme + 1)
                print(f"  Rate limit, {bekleme_suresi} saniye bekleniyor...")
                time.sleep(bekleme_suresi)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if deneme == max_retries - 1:
                print(f"  API hatası: {e}")
                return None
            time.sleep(5)
    return None


def sayfa_icerigi_al(sayfa_adi):
    params = {
        "action": "query",
        "format": "json",
        "titles": sayfa_adi,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "wiki",
        "redirects": 1,
    }
    data = api_istek_yap(params)
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return None
        metin = html_lib.unescape(page_data.get("extract", ""))
        metin = tam_metni_kaynakcadan_kirp(metin)
        return metin
    return None


def metni_bolumlere_ayir(metin):
    if not metin:
        return []
    satirlar = metin.split('\n')
    bolumler = []
    gecerli_baslik = None
    gecerli_satirlar = []

    for satir in satirlar:
        eslesme = BASLIK_DESENI.match(satir.strip())
        if eslesme:
            if gecerli_satirlar:
                bolumler.append((gecerli_baslik, '\n'.join(gecerli_satirlar)))
            gecerli_baslik = eslesme.group(1).strip()
            gecerli_satirlar = []
        else:
            gecerli_satirlar.append(satir)

    if gecerli_satirlar:
        bolumler.append((gecerli_baslik, '\n'.join(gecerli_satirlar)))

    return bolumler


def metni_parcala(metin):
    satirlar = metin.split('\n')
    parcalar = []
    for satir in satirlar:
        ham_satir = satir.strip()
        if not ham_satir:
            continue
        if referans_satiri_mi(ham_satir):
            continue
        satir_temiz = temizle(satir)
        if satir_temiz and len(satir_temiz) > 30:
            parcalar.append(satir_temiz)
    return parcalar


def sohbet_olustur(baslik, metin_parcalari, kategori):
    if not metin_parcalari:
        return None
    soru = f"{baslik} nedir?"
    sohbet = []
    for parca in metin_parcalari[:5]:
        sohbet.append({
            "misafir": soru,
            "Pusula-AI": parca
        })
    return {
        "kategori": kategori,
        "baslik": baslik,
        "sohbet": sohbet
    }


def kayit_ekle(tum_kayitlar, baslik, metin_parcalari, kategori):
    if not metin_parcalari:
        return False
    kayit = sohbet_olustur(baslik, metin_parcalari, kategori)
    if kayit:
        tum_kayitlar.append(kayit)
        print(f"  Eklendi: {baslik}")
        return True
    return False


def main():
    print("Turkce Wikipedia'dan JSON veri seti olusturuluyor...")
    print("=" * 50)

    tum_kayitlar = []
    toplam_kayit_sayisi = 0

    for kategori, sayfalar in KATEGORI_SAYFALAR.items():
        for sayfa_adi in sayfalar:
            if baslik_dislanmali_mi(sayfa_adi):
                continue
            print(f"\nİşleniyor: {sayfa_adi} (kategori: {kategori})")
            time.sleep(3)

            icerik = sayfa_icerigi_al(sayfa_adi)
            if not icerik:
                print(f"  Sayfa bulunamadı veya boş: {sayfa_adi}")
                continue

            bolumler = metni_bolumlere_ayir(icerik)
            for baslik, bolum_metni in bolumler:
                if baslik and baslik_dislanmali_mi(baslik):
                    continue
                if baslik:
                    kayit_baslik = f"{sayfa_adi}_{baslik}"
                    alt_kategori = f"{kategori}_{baslik.lower().replace(' ', '_')}"
                else:
                    kayit_baslik = sayfa_adi
                    alt_kategori = kategori

                metin_parcalari = metni_parcala(bolum_metni)
                if kayit_ekle(tum_kayitlar, kayit_baslik, metin_parcalari, alt_kategori):
                    toplam_kayit_sayisi += 1

    with open(CIKTI_DOSYASI, 'w', encoding='utf-8') as f:
        json.dump(tum_kayitlar, f, ensure_ascii=False, indent=2)

    son_boyut = os.path.getsize(CIKTI_DOSYASI)
    print(f"\n{'='*50}")
    print(f"İşlem tamamlandı!")
    print(f"Toplam kayıt sayısı: {toplam_kayit_sayisi}")
    print(f"Dosya boyutu: {son_boyut / 1024:.2f} KB")
    print(f"Dosya yolu: {CIKTI_DOSYASI}")
    if son_boyut < HEDEF_BOYUT_KB * 1024:
        print(f"UYARI: Hedeflenen boyuta ({HEDEF_BOYUT_KB} KB) ulaşılamadı.")
        print("Daha fazla sayfa ekleyin veya HEDEF_BOYUT_KB değerini düşürün.")


if __name__ == "__main__":
    main()
    print("\nProgram sonlandı.")
