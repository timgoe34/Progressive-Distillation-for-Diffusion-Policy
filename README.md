# Progressive Distillation for Diffusion Policy

## Miniconda Installation

Laden Sie den Miniconda Installer für Windows von der offiziellen Website herunter: "https://docs.conda.io/en/latest/miniconda.html"	
	- Führen Sie den Installer aus und folgen Sie den Anweisungen
	- Wählen Sie 'Just Me' bei der Installation
	- Aktivieren Sie die Option 'Add Miniconda3 to my PATH environment variable'


## Visual Studio Code Installation

	- Laden Sie VS Code von "https://code.visualstudio.com/" herunter
	- Führen Sie den Installer aus
	- Installieren Sie die Python Extension in VS Code:	
		- Öffnen Sie VS Code
		- Klicken Sie auf das Extensions-Symbol in der linken Leiste
		- Suchen Sie nach "Python"
		- Installieren Sie die offizielle Python Extension von Microsoft


## Codebase Ordner öffnen

In VS drücken Sie "Strg+O" und öffnen Sie einen Ordner aus der Codebase, bspw. 'Push_T - clean'. Der Ordner ist nun geöffnet und die Scripte sind links im Explorer verfügbar.

## Environment aus YAML-Datei erstellen
In VS drücken Sie "Strg+ö" um ein Terminal zu öffnen. Stellen Sie sicher, dass die Datei "environment.yml" links im Explorer angezeigt wird! Führen Sie folgenden Befehl im Terminal aus:

	conda env create --name envname --file=environments.yml python=3.11.9

Dabei ist "environment.yml" der Pfad zu Ihrer YAML-Datei und "envname" der Umgebungsname. 


## Environment in VS Code verwenden
	- Drücken Sie "Strg+Shift+P" um die Command Palette zu öffnen
	- Geben Sie "Python: Select Interpreter" ein
	- Wählen Sie Ihr neu erstelltes Conda Environment aus der Liste, bspw. "Python 3.11.9 envname"


## Tipps und Troubleshooting


	- Falls das Conda Environment nicht in der Liste erscheint, starten Sie VS Code neu
	- Überprüfen Sie, ob Conda im PATH ist mit dem Befehl \texttt{conda --version} im VS Terminal
	- Bei Problemen mit der Environment-Aktivierung, nutzen Sie das VS Code-Terminal und aktivieren Sie die Umgebung manuell:	

		conda activate envname
