A csomag tartalma:

1. docs.pdf: az üzenetek felépítését, működési architektúráról is író doksi aktuális állapotában

2. log_output_20200902novar.log: egy valós mozgás során a szenzorok által küldött üzenetek logja

3. Filter_Simulator.exe: "Filter_Simulator 1111" hívásra a fenti log alapján működteti a filtert és ZMQ publishert nyit a megadott (1111) porton, valamint az adatokat elmenti a sim_output.log fájlba is.
(A warningok amiért az ID=12 nincs figyelembe véve, normális.) Kb 150s-es a felvett folyamat, utána a szimulátor leáll.
A mintavételi idő most épp 50ms-re van állítva, de ez még változhat, és most is besűrűsödhet, ha éppen jön adat.

4. A felvett folyamatot szemlélteti a folyamatlog.jpg is, rajta a már szűrt sebesség.

5. A ZMQ csatorna működése ellenőrizhető a LogAll.exe app-pal "LogAll 1111" hívással, ha az 1111 portra állítanátok rá.
Ez a logall_output_*.log fájlokba termeli a sim_output.log tartalmának megfelelő üzeneteket.

6. DataMsgContentNameSpace: a flatbuffer üzenet kibontására elvileg megfelelő python scriptek (a saját formátumhoz generálva)


Feladatok:

1. ZMQ Subscriber inicializálása tcp://localhost:1111 (vagy más megfelelő port) címre, elvileg ekkor meg fognak indulni az adatok.

2. Szűrni a topik megadásával, hogy csak a szűrt pozíciókat, sebességeket tartalmazó cuccok jöjjenek be, a filter alapértelmezetten mindent elérhetővé tesz.

4. A Flatbuffer-es feldolgozóval kivenni a "value" vektort és annak a megfelelő elemeit. 
