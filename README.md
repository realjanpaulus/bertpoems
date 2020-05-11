# bertpoems

Term paper for the Course "Textklassifikation" of the Julius-Maximilians-University, WS 19/20.

## TODOS

- Korpus cleanen und aufbereiten
- SVM und LR auf Korpus trainieren
- gucken, wie man Domänenadaption verwendet (siehe Severin)


## Fragestellung

- Korpus: Gedichte im Zeitraum von 1870 bis 1920/30 (Moderne)
	- kurze Texte
	- recht stabile Orthographie (anders als davor)
- Ist ein fine-tuned BERT Modell so mächtig, dass es besser als traditionelle Machine Learning Verfahren (SVM, LR) und andere Neuronale Netzstrukturen (CNN) funktioniert?
	- Problem: Bert ist auf Wikipedia trainiert, welches teilweise schon sehr unterschiedlich im Gegensatz zur Sprache der Gedichte ist. SVM und LR lernen "from scratch", d.h. aus den vorgegeben Daten
	- Kommt es BERTs Klassifikationsergebnissen zugute, dass ähnlich wie bei Fasttext out-of-vocabulary tokens aus Subword Tokens zusammengerechnet werden und somit das OOV-Problem umgangen wird?
	- IDEE: verschiedene BERT Modelle ausprobieren
		- BERT Multi
		- BERT Deutsch
- Hilft eine Domänenadaption von einem fine-tuned BERT für eine bessere Klassifikation?
	- IDEEN: 
		- ausprobieren, ob Domänenadaption auf gleiches Gedichtskorpus sinnvoll
		- auch ausprobieren, ob Domänenadaption auf Prosa Korpus aus der gleichen Zeit sinnvoll?
	
### Domänenadaption
Bei einer Domänenadaption wird ein möglichst großes Korpus mit Texten aus der Zieldomäne (= Gedichte aus der Moderne oder evtl. auch Prosa Texte aus der gleichen Zeit) verwendet. BERT trainiert dann auf Masked Language Modeling weiter, womit es im Idealfall etwas über Gedichte im Allgemeinen lernt. Dann werden die Texte für eine Textklassifikation adaptiert, indem die auf das Masked Language Modeling angepassten Gewichte für die Klassifikation optimiert werden. Durch das Transfer Learning sind die Gewichte des Netwerks nicht zufällig initalisiert, sondern auf Task A optimiert, wovon Task B profitieren kann. Je näher Task A an Task B ist (durch eine Domänenadaption sollte A ja im Idealfall näher an B sein), desto mehr profitiert Task B natürlich davon.

	
