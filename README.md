https://github.com/user-attachments/assets/8291c708-96b4-421e-a49c-ab254f167874
# Improving Diffusion Policy

# Anleitung für Windows

## Minicon

da Installation

Laden Sie den Miniconda Installer für Windows von der offiziellen Website herunter: "https://docs.conda.io/en/latest/miniconda.html"	
Direkter Link: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
- Führen Sie den Installer aus und folgen Sie den Anweisungen
- Wählen Sie 'Just Me' bei der Installation
- Aktivieren Sie die Option 'Add Miniconda3 to my PATH environment variable'


## Visual Studio Code Installation

- Laden Sie VS Code von "https://code.visualstudio.com/" herunter
- Führen Sie den Installer aus
- Öffnen Sie VS Code
- Klicken Sie auf das Extensions-Symbol in der linken Leiste Strg+Shift+X
- Suchen Sie nach "Python"
- Installieren Sie die offizielle Python Extension von Microsoft

Für Pybullet ist außerdem die Installation von cpp visual build tools notwendig. Nur dann kann pybullet verwendet werden.
https://visualstudio.microsoft.com/de/visual-cpp-build-tools/

## Codebase Ordner öffnen

In VS drücken Sie "Strg+k" und danach "Strg+o" und öffnen Sie einen Ordner aus der Codebase, bspw. 'Push_T - clean'. Der Ordner ist nun geöffnet und die Scripte sind links im Explorer verfügbar.

## Environment aus YAML-Datei erstellen

Überprüfen Sie bitte, ob Ihr PC CUDA unterstützt: Dafür geben Sie "nvidia-smi" in bash ein oder prüfen Sie, ob ihre GPU auf dieser Liste ist: https://en.wikipedia.org/wiki/CUDA.
### Mit CUDA

In VS drücken Sie "Strg+ö" um ein Terminal zu öffnen. Stellen Sie sicher, dass die Datei "environment_cuda.yml" links im Explorer auf der ersten Ebene unter dem Ordner angezeigt wird! Führen Sie folgenden Befehl im Terminal aus. Das klappt allerdings nur, wenn conda Teil der System-Umgebungsvariablen ist. 

 conda env create --name envname --file=environments_cuda.yml python=3.11.9

Dabei ist "environment_cuda.yml" der Pfad zu Ihrer YAML-Datei und "envname" der Umgebungsname. 

### Ohne CUDA

In VS drücken Sie "Strg+ö" um ein Terminal zu öffnen. Stellen Sie sicher, dass die Datei "environment_cpu.yml" links im Explorer auf der ersten Ebene unter dem Ordner angezeigt wird! Führen Sie folgenden Befehl im Terminal aus. Das klappt allerdings nur, wenn conda Teil der System-Umgebungsvariablen ist. 

 conda env create --name envname --file=environments_cpu.yml python=3.11.9

Dabei ist "environment_cuda.cpu" der Pfad zu Ihrer YAML-Datei und "envname" der Umgebungsname. 


## Environment in VS Code verwenden
- Drücken Sie "Strg+Shift+P" um die Command Palette zu öffnen
- Geben Sie "Python: Select Interpreter" ein
- Wählen Sie Ihr neu erstelltes Conda Environment aus der Liste, bspw. "Python 3.11.9 envname"


## Tipps und Troubleshooting


- Falls das Conda Environment nicht in der Liste erscheint, starten Sie VS Code neu
- Überprüfen Sie, ob Conda im PATH ist mit dem Befehl "conda --version" im VS Terminal
- Bei Problemen mit der Environment-Aktivierung, nutzen Sie das VS Code-Terminal und aktivieren Sie die Umgebung manuell:	

 conda activate envname
