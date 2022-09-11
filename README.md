## Automatische Erkennung von sexistischem Content auf Twitter mit BERT

**Zielsetzung:**

	- Entwicklung eines Modells zur automatischen Erkennung von sexistischen Tweets
	- Einsatz eines vortrainierten BERT-Modells mit Fine-Tuning auf spezifische Aufgabe

**Input:**

	1. Korpus mit annotierten Daten (Label: sexist = True/False): "Call me sexist but"-Datenset
	2. Aufteilung des Korpus in Trainings-, Validierungs- und Testdaten

**Hauptstruktur:**

	1. Einlesen der Daten
	2. Preprocessing
	3. Training
    4. Testing

**Output:** 

	- Layer mit binärer Klassifikation der Tweets
***
### Quickstart
***
Die Anwendung wurde in Python 3.9 geschrieben.

Klone das Repository

`git clone https://github.com/ChristineSchaefer/sexism_detection_bert.git`

Installiere Requirements

`python -m pip install -r requirements`

→ Um das Programm starten zu können, muss Tensorflow installiert sein.

Beim Starten müssen folgende Argumente als String übergeben werden:
1. Pfad zum Korpus
2. Spaltenname Tweets
3. Spaltenname Label
4. Data Augmentation (True/False)

z.B. `"\Users\Christine\sexism_data.csv" "text" "sexist" "False"`

Starten des Programms über `main.py`.
***
### Code Struktur
***
```
📦sexism_detection_bert
├── 📂data_augmentation
│   └── 📜construction.py
├── 📂experiments
│   ├── 📂balanced
│   │   ├── 📂1
│   │   ├── 📂2
│   │   ├── 📂3
│   │   └── 📂4
│   └── 📂unbalanced
│       ├── 📂1
│       ├── 📂2
│       ├── 📂3
│       └── 📂4
├── 📂graphs
├── 📂logger
├── 📂model
│   ├── 📜config.py
│   ├── 📜creation.py
│   └── 📜evaluation.py
├── 📂models
│   ├── 📂balanced_model
│   │   ├── 📂assets
│   │   ├── 📂variables
│   │   └── 📜saved_model.pb
│   └── 📂unbalanced_model
│       ├── 📂assets
│       ├── 📂variables
│       └── 📜saved_model.pb
├── 📂preprocessing
│   ├── 📜normalization.py
│   ├── 📜process_data.py
│   └── 📜tokenizer.py
├── 📂utils
│   └── 📜argparser.py
├── 📜.gitignore
├── 📜main.py
├── 📜README.md
└── 📜requirements.txt
```
