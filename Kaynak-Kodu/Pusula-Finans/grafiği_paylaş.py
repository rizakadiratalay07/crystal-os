import sys
import os
import tempfile
import json
import subprocess
import time
from datetime import datetime
from PyQt6 import QtWidgets, QtCore, QtGui
import requests

# Programcı: Ali Emre ERYILMAZ

class PusulaFinans(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grafiği Paylaş")
        self.setGeometry(100, 100, 700, 500)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #000000;
                font-size: 12px;
            }
            QLineEdit, QTextEdit {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #4a6fa5;
            }
            QPushButton {
                background-color: #4a6fa5;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a7fb5;
            }
            QPushButton:pressed {
                background-color: #3a5f95;
            }
            QComboBox {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox:hover {
                border: 1px solid #4a6fa5;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #000000;
                margin-right: 5px;
            }
            QCheckBox {
                color: #000000;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #4a6fa5;
                border: 1px solid #4a6fa5;
                border-radius: 3px;
            }
            QGroupBox {
                color: #000000;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #f8f8f8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #f0f0f0;
            }
            QStatusBar {
                background-color: #e0e0e0;
                color: #000000;
            }
        """)

        self.data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_sunucular.json")
        self.servers = self.load_servers()
        self.init_ui()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setSpacing(10)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)
        form_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.token_input = QtWidgets.QLineEdit()
        self.token_input.setPlaceholderText("Telegram Bot Token (ör: 123456:ABC-DEF)")
        form_layout.addRow("🤖 Bot Token:", self.token_input)

        self.chat_input = QtWidgets.QLineEdit()
        self.chat_input.setPlaceholderText("Kullanıcı ID veya Kanal adı (@kanal_adi)")
        form_layout.addRow("👤 Chat ID:", self.chat_input)

        self.server_combo = QtWidgets.QComboBox()
        self.server_combo.addItem("-- Yeni sunucu --")
        self.update_server_combo()
        self.server_combo.currentIndexChanged.connect(self.on_server_selected)
        form_layout.addRow("💾 Kayıtlı Sunucular:", self.server_combo)

        self.extra_text = QtWidgets.QTextEdit()
        self.extra_text.setPlaceholderText("Ekran görüntüsü altına eklemek istediğiniz metni yazın...")
        self.extra_text.setMaximumHeight(80)
        form_layout.addRow("📝 Ekstra Metin:", self.extra_text)

        options_group = QtWidgets.QGroupBox("Seçenekler")
        options_group.setMaximumHeight(70)
        options_layout = QtWidgets.QHBoxLayout(options_group)
        options_layout.setContentsMargins(10, 5, 10, 5)
        options_layout.setSpacing(10)

        self.save_server_check = QtWidgets.QCheckBox("Bu sunucuyu kaydet")
        self.save_server_check.setChecked(False)
        options_layout.addWidget(self.save_server_check)

        self.server_name_input = QtWidgets.QLineEdit()
        self.server_name_input.setPlaceholderText("Sunucu adı (opsiyonel)")
        self.server_name_input.setEnabled(False)
        self.save_server_check.stateChanged.connect(
            lambda: self.server_name_input.setEnabled(self.save_server_check.isChecked())
        )
        options_layout.addWidget(self.server_name_input)

        options_layout.addStretch()
        form_layout.addRow(options_group)

        layout.addLayout(form_layout)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(10)

        self.capture_btn = QtWidgets.QPushButton("📸 Ekran Görüntüsü Al ve Gönder")
        self.capture_btn.clicked.connect(self.capture_and_send)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #5CBF60;
            }
        """)
        button_layout.addWidget(self.capture_btn)

        self.clear_btn = QtWidgets.QPushButton("🗑️ Temizle")
        self.clear_btn.clicked.connect(self.clear_fields)
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır")

    def load_servers(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_servers(self):
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.servers, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Hata", f"Sunucu kaydedilemedi: {e}")

    def update_server_combo(self):
        current_text = self.server_combo.currentText()
        self.server_combo.clear()
        self.server_combo.addItem("-- Yeni sunucu --")
        for name, data in self.servers.items():
            display_name = f"{name} ({data.get('chat_id', '')})"
            self.server_combo.addItem(display_name, data)

        index = self.server_combo.findText(current_text)
        if index >= 0:
            self.server_combo.setCurrentIndex(index)

    def on_server_selected(self, index):
        if index > 0:
            data = self.server_combo.itemData(index)
            if data:
                self.token_input.setText(data.get('token', ''))
                self.chat_input.setText(data.get('chat_id', ''))

    def clear_fields(self):
        self.token_input.clear()
        self.chat_input.clear()
        self.extra_text.clear()
        self.server_name_input.clear()
        self.server_combo.setCurrentIndex(0)
        self.status_bar.showMessage("Form temizlendi")

    def find_window_id_by_title(self, title):
        try:
            result = subprocess.run(
                ['xdotool', 'search', '--name', title],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                window_ids = result.stdout.strip().split()
                if window_ids:
                    return int(window_ids[0])
            return None
        except Exception:
            return None

    def capture_and_send(self):
        token = self.token_input.text().strip()
        if not token:
            QtWidgets.QMessageBox.warning(self, "Hata", "Lütfen Bot Token girin!")
            return

        chat_id = self.chat_input.text().strip()
        if not chat_id:
            QtWidgets.QMessageBox.warning(self, "Hata", "Lütfen Chat ID girin!")
            return

        extra_text = self.extra_text.toPlainText().strip()

        self.status_bar.showMessage("Ekran görüntüsü alınıyor...")
        QtWidgets.QApplication.processEvents()

        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            screen = QtWidgets.QApplication.primaryScreen()
            if screen is None:
                raise Exception("Ekran bulunamadı!")

            window_id = self.find_window_id_by_title("Pusula Finans V1.3")
            if window_id is None:
                raise Exception("Pusula Finans V1.3 penceresi bulunamadı! Lütfen xdotool kurulu olduğundan emin olun (sudo apt install xdotool)")

            # Hedef pencereyi öne getir
            subprocess.run(['xdotool', 'windowactivate', str(window_id)], capture_output=True)
            time.sleep(0.3)  # Pencere öne gelsin diye bekle

            pixmap = screen.grabWindow(window_id)
            if pixmap.isNull():
                raise Exception("Pencere görüntüsü alınamadı!")
            if not pixmap.save(temp_path, 'PNG'):
                raise Exception("Dosya kaydedilemedi!")

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise Exception("Ekran görüntüsü dosyası oluşturulamadı!")

            self.status_bar.showMessage("Ekran görüntüsü Telegram'a gönderiliyor...")
            QtWidgets.QApplication.processEvents()

            success = self.send_to_telegram(token, chat_id, temp_path, extra_text)

            try:
                os.unlink(temp_path)
            except:
                pass

            if success:
                if self.save_server_check.isChecked():
                    server_name = self.server_name_input.text().strip()
                    if not server_name:
                        server_name = f"Sunucu_{len(self.servers) + 1}"

                    self.servers[server_name] = {
                        'token': token,
                        'chat_id': chat_id,
                        'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.save_servers()
                    self.update_server_combo()

                    QtWidgets.QMessageBox.information(
                        self,
                        "Başarılı",
                        f"✅ Ekran görüntüsü başarıyla gönderildi!\n\n"
                        f"📌 Sunucu '{server_name}' olarak kaydedildi."
                    )
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Başarılı",
                        "✅ Ekran görüntüsü başarıyla gönderildi!"
                    )

                self.status_bar.showMessage("Gönderim tamamlandı")

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Hata", f"İşlem başarısız:\n{str(e)}")
            self.status_bar.showMessage(f"Hata: {str(e)}")

    def send_to_telegram(self, token, chat_id, photo_path, caption=""):
        try:
            url = f"https://api.telegram.org/bot{token}/sendPhoto"

            if os.path.getsize(photo_path) > 10 * 1024 * 1024:
                QtWidgets.QMessageBox.warning(self, "Hata", "Dosya çok büyük! (max 10MB)")
                return False

            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': chat_id,
                    'caption': caption if caption else "📸 Ekran görüntüsü",
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                return True
            else:
                error_msg = response.json().get('description', 'Bilinmeyen hata')
                QtWidgets.QMessageBox.warning(
                    self,
                    "Telegram Hatası",
                    f"Hata kodu: {response.status_code}\n{error_msg}"
                )
                return False

        except requests.exceptions.Timeout:
            QtWidgets.QMessageBox.warning(self, "Hata", "Bağlantı zaman aşımı!")
            return False
        except requests.exceptions.ConnectionError:
            QtWidgets.QMessageBox.warning(self, "Hata", "Telegram'a bağlanılamadı!")
            return False
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Hata", f"Gönderim hatası: {e}")
            return False

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setWindowIcon(QtGui.QIcon())
    window = PusulaFinans()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
