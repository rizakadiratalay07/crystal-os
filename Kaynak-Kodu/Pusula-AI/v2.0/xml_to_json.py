import xml.etree.ElementTree as ET
import json
import sys
import os
import re

def parse_wikipedia_xml(xml_file, output_json):
    file_size = os.path.getsize(xml_file)
    print(f"📂 Dosya boyutu: {file_size / (1024**3):.2f} GB")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        f.write('[')
        first = True
        count = 0
        total_pages = 0
        current_page = {}
        
        for event, elem in ET.iterparse(xml_file, events=('start', 'end')):
            tag = elem.tag.split('}')[-1]
            
            if event == 'start' and tag == 'page':
                current_page = {'title': None, 'text': None}
            
            elif event == 'end':
                if tag == 'title':
                    if elem.text:
                        current_page['title'] = elem.text.strip()
                
                elif tag == 'text':
                    if elem.text:
                        text = elem.text.strip()
                        # SADECE XML etiketlerini temizle (varsa)
                        text = re.sub(r'<[^>]+>', '', text)
                        # Çoklu boşlukları tek boşluğa çevir
                        text = ' '.join(text.split())
                        current_page['text'] = text
                
                elif tag == 'page':
                    total_pages += 1
                    if total_pages % 10000 == 0:
                        print(f"📄 {total_pages} sayfa taranıyor... (kayıtlı makale: {count})", file=sys.stderr)
                    
                    title = current_page.get('title')
                    text = current_page.get('text')
                    
                    if title and text and len(text) > 100 and not title.startswith('Kategori:'):
                        if not first:
                            f.write(',')
                        json.dump({'baslik': title, 'metin': text}, f, ensure_ascii=False)
                        first = False
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"📊 {count} makale kaydedildi", file=sys.stderr)
                    
                    elem.clear()
                    current_page = {}
        
        f.write(']')
    
    print(f"\n✅ {count} makale {output_json} dosyasına kaydedildi.")
    print(f"📄 Toplam taranan sayfa: {total_pages}")

if __name__ == '__main__':
    print("🚀 XML'den JSON dönüşümü başlıyor...")
    parse_wikipedia_xml('trwiki-latest-pages-articles.xml', 'wikipedia_tr.json')
