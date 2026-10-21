# Równoległe sito Eratostenesa w Pythonie i MPI

Pakiet realizuje obie części zadania: sito z przesyłaniem komunikatów oraz jego
modyfikację ze współdzieloną tablicą. Układ `comm`, `rank`, `size`, alokacja tylko
przez proces 0, `Shared_query`, widok `np.ndarray` i `Fence` nawiązują do
załączonego `test_sm.py`. Przedział wynikowy jest **domknięty: [2, n]**.

## 1. Pliki i wymagania

| Plik | Przeznaczenie |
| --- | --- |
| `sito_mpi.py` | Oba warianty sita, pomiar czasu i opcjonalna kontrola wyniku |
| `sito.slurm` | Zadanie wsadowe na jednym węźle, domyślnie 5 procesów |
| `quickstart.py` | Test rang, liczby procesów i nazw węzłów |
| `test_sito.py` | Testy matematyki bez konieczności instalowania MPI |
| `WERYFIKACJA.md` | Zakres i wyniki faktycznie wykonanych sprawdzeń |

Program używa Pythona >= 3.8 (`math.isqrt` istnieje od 3.8), NumPy i mpi4py oraz
systemowej implementacji MPI z obsługą okien współdzielonych MPI-3. Opisy API MPI
odnoszą się do dokumentacji mpi4py **4.0.3** wskazanej w zadaniu. Kod nie używa
nowych funkcji wymagających wyłącznie tej wersji. Stary moduł z Quickstartu może
zawierać inną wersję; zgodność konkretnego środowiska trzeba sprawdzić na klastrze.
Nie są potrzebne SciPy, multiprocessing ani wątki Pythona.

Poza klastrem, po zainstalowaniu zgodnej implementacji MPI, biblioteki Pythona
można zainstalować poleceniem `python -m pip install numpy mpi4py`. Na Aresie
najpierw korzystaj z modułów administratora. Interpreter, mpi4py i launcher
`mpiexec` muszą używać tej samej implementacji MPI; nie mieszaj Intel MPI,
Open MPI i MPICH w jednym uruchomieniu.

## 2. Teoria sita i uzasadnienie poprawności

Liczba pierwsza jest liczbą całkowitą większą od 1 mającą dokładnie dwa dodatnie
dzielniki. Liczby 0 i 1 nie są pierwsze. Dla każdej złożonej liczby x istnieje
pierwszy dzielnik d <= sqrt(x): gdyby oba czynniki w rozkładzie x = a*b były
większe od sqrt(x), iloczyn przekraczałby x. Rozkładając mniejszy czynnik dalej,
otrzymujemy odpowiedni dzielnik pierwszy. Zatem do sprawdzenia wszystkich
x <= n wystarczają pierwsze d <= floor(sqrt(n)).

Sekwencyjne sito tworzy tablicę flag, początkowo prawdziwych poza 0 i 1.
Dla kolejnej nadal zaznaczonej liczby d wykreśla d², d²+d, d²+2d, ... .
Dlaczego od d²? Mniejsze złożone wielokrotności d mają już mniejszy czynnik
pierwszy, a samego d nie wolno wykreślić. Po obsłużeniu dzielników do pierwiastka
pozostają dokładnie liczby pierwsze.

Kod wyznacza `limit = isqrt(n)` i dzieli dziedzinę zgodnie z załącznikiem:

- B = [2, limit]: małe liczby pierwsze, obliczane niezależnie przez każdy proces.
- C = [limit+1, n]: część dzielona między procesy.

W Pythonie wygodniej zapisywać przedziały półotwarte: C = `[limit+1, n+1)`.
Dla procesu r, liczby procesów P i L = n-limit granice wynoszą:

```text
lo_r = limit + 1 + (L*r)//P
hi_r = limit + 1 + (L*(r+1))//P
```

Sąsiednie granice są identyczne, więc nie ma luk ani nakładania przedziałów.
Pierwsza to limit+1, ostatnia n+1; długości różnią się najwyżej o 1.
Działa to również, gdy P > L: część procesów dostaje puste przedziały.
To drobne uogólnienie opisu z zadania: reszta jest rozłożona między bloki,
zamiast wymuszać jeden krótszy blok na końcu.

Dla dzielnika d pierwszy wykreślany element w `[lo, hi)` to:

```text
first = max(d*d, ((lo+d-1)//d)*d)
```

Drugi składnik jest najmniejszą wielokrotnością d nie mniejszą niż lo.
Pierwszy chroni samą liczbę pierwszą i pomija zbędne wykreślenia poniżej d².
Indeks lokalny to `first-lo`, dlatego zapis to `flags[first-lo::d] = False`.
Jeżeli first >= hi, nie ma nic do wykreślenia. Po wszystkich dzielnikach żaden
złożony element nie pozostaje zaznaczony, a żaden pierwszy nie został usunięty.

Przykład n=30, P=5: limit=5, B zawiera [2,3,5].

| Ranga | Fragment C | Pierwsze z fragmentu |
| --- | --- | --- |
| 0 | [6,11) | 7 |
| 1 | [11,16) | 11,13 |
| 2 | [16,21) | 17,19 |
| 3 | [21,26) | 23 |
| 4 | [26,31) | 29 |

Wynik łączny to [2,3,5,7,11,13,17,19,23,29]. Dla n=0 lub n=1 wynik jest pusty;
dla n=2 wynosi [2]. Ujemne n jest odrzucane.

## 3. Model MPI i wariant message

MPI uruchamia P osobnych procesów wykonujących ten sam program (SPMD).
Każdy ma własny interpreter i zwykłą pamięć prywatną. `rank` identyfikuje proces
w komunikatorze, a `size` jest liczbą jego uczestników. Ranga 0 też oblicza swój
fragment; nie jest wyłącznie koordynatorem.

`sieve_message` wykonuje cztery kroki zgodne z dokumentem zadania:

1. Każdy proces sam wykonuje `base_primes(limit)`; celowo nie rozsyłamy B.
2. Każdy wyznacza własny `[lo, hi)` i alokuje prywatne flagi.
3. `mark_segment` wykreśla wielokrotności; `flatnonzero(flags)+lo` daje liczby.
4. Wszyscy wywołują `gather`; proces 0 dokleja B i kolejne wyniki C.

`gather` zwraca elementy w kolejności rang, nie kolejności nadejścia. Każdy
fragment jest już posortowany, a bloki mają rosnące granice, więc `concatenate`
daje posortowany wynik bez dodatkowego sortowania. Pozostałe procesy zwracają
`None`. Wariant ten może działać również na wielu węzłach.

Metody mpi4py z małych liter, np. `gather`, przesyłają obiekty Pythona przez
serializację. Metody z wielkiej litery, np. `Gather`/`Gatherv`, komunikują bufory.
Wybrano czytelniejsze `gather`, które obsługuje różne długości fragmentów.
`Gather` o stałych licznościach nie byłoby bezpośrednim zamiennikiem.
W dużych zadaniach można użyć `Gatherv`, wcześniej zbierając liczności i obliczając
przesunięcia. Tutorial NYU wyjaśnia `Send`/`Recv`; tutaj kolektyw zastępuje ręczne
pary wysyłania i odbierania. Blokujące `Send` gwarantuje możliwość ponownego użycia
bufora po powrocie, nie zawsze zakończenie odbioru przez drugi proces.

## 4. Wariant shared i synchronizacja

`sieve_shared` tworzy jedną tablicę flag długości n+1. Indeks x opisuje liczbę x.
Na każdym procesie istnieje osobny obiekt Pythona `shared_array`, ale jego dane
wskazują ten sam segment pamięci. Samo `np.ones` w wielu procesach nie dałoby
tego efektu: powstałyby niezależne tablice.

`Split_type(COMM_TYPE_SHARED, key=rank)` wydziela procesy mogące współdzielić pamięć.
Kod wymaga, aby ta grupa obejmowała cały `COMM_WORLD`; w przeciwnym razie kończy
zadanie z błędem. `key=rank` zachowuje kolejność rang, więc właścicielem segmentu
jest również ranga 0 nowego komunikatora. Domyślne `--nodes=1` w SLURM odpowiada
temu wymaganiu. MPI nie tworzy jednego takiego bufora RAM między dowolnymi węzłami.

Wszyscy uczestniczą w `Allocate_shared`, lecz tylko proces 0 podaje dodatni
rozmiar `(n+1)*dtype.itemsize`; inni podają 0 bajtów. `Shared_query(0)` daje każdemu
lokalny dostęp do segmentu rangi 0. Zwrócony adres może być inny w różnych
procesach — ważne, że opisuje te same dane. `np.ndarray(buffer=..., shape=..., dtype=...)`
interpretuje ten bufor bez tworzenia drugiej tablicy flag.

Wybrano `np.bool_`, a nie C-int `'i'` z przykładu: wystarczy flaga True/False.
Rozmiar jest obliczany przez `dtype.itemsize`; NumPy bool zajmuje 1 bajt, nie 1 bit.
`disp_unit` okna jest jednostką przesunięć w bajtach, a nie długością tablicy.
Kod sprawdza ją po `Shared_query` oraz wymaga modelu `WIN_UNIFIED`.

Kolejność pracy i cztery kolektywne wywołania `win.Fence()`:

1. Pierwsze Fence wyznacza początek fazy dostępu.
2. Proces 0 ustawia część `[0,limit]` na False, a indeksy podzielników na True.
   Każdy proces ustawia swój fragment C na True. Drugie Fence kończy inicjalizację.
3. Procesy wykreślają wyłącznie we własnych fragmentach C. Trzecie Fence oddziela
   wszystkie zapisy od odczytu całej tablicy przez proces 0.
4. Proces 0 wykonuje `flatnonzero(shared_array)`, tworząc własną tablicę wyniku.
   Czwarte Fence oddziela odczyt od zwalniania zasobu. Usuwamy widoki i wywołujemy
   kolektywnie `win.Free()` oraz `node_comm.Free()`.

`Fence` synchronizuje okno i widoczność zapisów; nie jest jedynie umownym miejscem
w kodzie. Zgodnie ze standardem może zamykać poprzednią i otwierać następną epokę
RMA. Tutaj służy rozdzieleniu faz bezpośredniego dostępu do wspólnych danych,
bez `Put` ani `Get`. Używamy domyślnego `assertion=0`; nie deklarujemy optymalizacji
`MODE_NOPRECEDE`, która może osłabić synchronizację. Samo `comm.Barrier()` nie jest
ogólnym zamiennikiem synchronizacji pamięci okna.

Nie rozdzielamy pracy według dzielników, bo np. procesy wykreślające wielokrotności
2 i 3 zapisywałyby jednocześnie komórkę 6. Identyczna wartość False nie jest
uzasadnieniem poprawności współbieżnych zapisów. Rozłączne fragmenty eliminują
konflikty na poziomie elementów. Możliwe współdzielenie linii cache na granicy
bloków wpływa na wydajność, ale nie zmienia podziału własności komórek.

Wyniki nie są zbierane `gather`: proces 0 odczytuje wspólną tablicę.
Niewielkie kolektywy pomiaru czasu i statusu nadal występują.
W porównaniu z przykładem `test_sm.py` rozwiązanie nie wymaga dokładnie 5 procesów,
obsługuje resztę z dzielenia i sprawdza domenę współdzielenia.

## 5. Użyte funkcje i obiekty bibliotek

Poniższe opisy odnoszą się do wywołań w dostarczonym kodzie, a odsyłacze w sekcji
11 prowadzą do specyfikacji. `MPI` importujemy wewnątrz `main`, aby testy czystej
matematyki mogły działać bez mpi4py. Normalny import `mpi4py.MPI` automatycznie
inicjalizuje MPI; standardowo kończy je też przy wyjściu z interpretera.

### mpi4py

| API | Znaczenie w programie |
| --- | --- |
| `MPI.COMM_WORLD` | Komunikator wszystkich uruchomionych procesów |
| `comm.Get_rank()` | Numer bieżącego procesu, od 0 |
| `comm.Get_size()` | Liczba uczestników komunikatora |
| `comm.gather(obj, root=0)` | Lista obiektów w procesie 0, `None` w pozostałych |
| `comm.Barrier()` | Czeka na wejście wszystkich uczestników |
| `MPI.Wtime()` | Lokalny czas ścienny w sekundach; odejmujemy odczyty tego samego procesu |
| `comm.reduce(value, op=MPI.MAX, root=0)` | Maksimum czasów w procesie 0 |
| `comm.bcast(obj, root=0)` | Rozsyła wynik kontroli do wszystkich procesów |
| `comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)` | Tworzy komunikator domeny pamięci wspólnej |
| `MPI.Win.Allocate_shared(size, disp_unit, comm=...)` | Kolektywnie alokuje okno; size oznacza lokalne bajty |
| `win.Shared_query(0)` | Zwraca `(buffer, disp_unit)` segmentu rangi 0 |
| `win.Get_attr(MPI.WIN_MODEL)` | Odczytuje model okna, porównywany z `MPI.WIN_UNIFIED` |
| `win.Fence()` | Kolektywnie synchronizuje okno |
| `win.Free()` | Kolektywnie zwalnia okno |
| `node_comm.Free()` | Zwalnia utworzony komunikator, nie `COMM_WORLD` |
| `comm.Abort(1)` | Awaryjnie kończy zadanie po wyjątku |

Kolektywy muszą być wywoływane przez wszystkich uczestników we właściwej
kolejności. `if rank == 0` obejmuje odczyt/drukowanie, lecz nie samo `gather` ani
Fence. `Abort` jest ścieżką awaryjną zapobiegającą pozostawieniu innych procesów
w oczekiwaniu; omija zwykłe mechanizmy sprzątania Pythona.

### NumPy

| API / operacja | Znaczenie |
| --- | --- |
| `np.ones(length, dtype=np.bool_)` | Nowa prywatna tablica True |
| `np.bool_` | Typ logiczny elementów flag |
| `np.dtype(np.bool_)`, `.itemsize` | Opis reprezentacji i bajty na element |
| `np.ndarray(buffer=buf_ptr, dtype=dtype, shape=(n+1,))` | Widok danych okna o zadanym typie i kształcie |
| `np.flatnonzero(flags)` | Nowa tablica indeksów elementów niezerowych, tutaj True |
| `np.concatenate([divisors] + parts)` | Połączenie jednowymiarowych wyników |
| `array.size` | Liczba elementów, nie bajtów |
| `array.tolist()` | Kopia danych jako lista skalarów Pythona |
| `array[lo:hi]` | Wycięcie podstawowe, widok tej samej pamięci |
| `array[start::step] = False` | Przypisanie co step elementów, także w buforze okna |
| `shared_array[divisors] = True` | Przypisanie pod wskazane indeksy |

`flatnonzero(flags)+lo` dodaje dolną granicę do każdego indeksu. Tablica indeksów
używa całkowitego typu indeksowego NumPy, zwykle 64-bitowego na klastrze.
W `mark_segment` konwertujemy każdy dzielnik przez `int`, by obliczać d² i granice
w arytmetyce całkowitej Pythona. Indeksy i rozmiary tablic nadal muszą mieścić się
w możliwościach NumPy i dostępnej pamięci.

### Biblioteka standardowa Pythona

| API | Znaczenie |
| --- | --- |
| `math.isqrt(n)` | Dokładne floor(sqrt(n)) bez zaokrągleń zmiennoprzecinkowych |
| `argparse.ArgumentParser(description=...)` | Definiuje interfejs konsolowy i automatyczną pomoc |
| `parser.add_argument(...)` | Dodaje argument; `type=int` konwertuje, `choices` ogranicza wartości |
| `action='store_true'` | Flaga przełączająca wartość logiczną |
| `parser.parse_args()` | Odczytuje argumenty w obiekt z polami |
| `parser.error(message)` | Wypisuje błąd użycia i kończy kodem 2 |
| `socket.gethostname()` | Nazwa węzła w `quickstart.py`, bez otwierania połączeń |
| `unittest.TestCase`, `.assertEqual(...)`, `unittest.main()` | Organizacja i uruchamianie testu oraz porównywanie wyniku z oczekiwanym |

Wbudowane `range` generuje zakres całkowity, `all` sprawdza wszystkie warunki,
a `%` bada resztę z dzielenia w niezależnej kontroli. `//` to dzielenie całkowite
z zaokrągleniem w dół. `del` usuwa referencje do widoków; bufor MPI zwalnia dopiero
`win.Free()`. `raise SystemExit(main())` przekazuje kod zakończenia programu.

## 6. Quickstart na Aresie

Wykonuj polecenia w katalogu zawierającym pliki pakietu. Grant z załącznika
`plgar2022-cpu` jest przykładem, nie przydzielonym tutaj kontem. Wstaw własny grant.
Nazwa modułu poniżej jest **dosłownie z załączonego Quickstartu**, nie jest
potwierdzeniem jego aktualnej dostępności. Najpierw sprawdź:

```bash
module avail scipy-bundle
```

Jeżeli wskazanego modułu już nie ma, wybierz zgodny moduł Pythona/NumPy/mpi4py/MPI
według dokumentacji lub pomocy administratora Aresa i zmień `module load`.

```bash
srun --overlap -A TWOJ_GRANT-cpu -N 1 --tasks-per-node=5 -p plgrid -t 00:20:00 --pty /bin/bash
module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1
mpiexec python quickstart.py
```

Oczekuj pięciu linii z rangami 0–4, wszędzie `out of 5 processes` i nazwą tego
samego węzła. Kolejność linii jest nieokreślona. Nazwa hosta nie dowodzi przypięcia
procesów do różnych rdzeni; do tego służą ustawienia wiązania MPI/SLURM.
Zgodnie z Quickstartem **na Aresie nie podajemy `-n` ani `-np` do mpiexec**:
liczbę procesów bierze z przydziału SLURM. Nie uruchamiaj całych obliczeń na
węźle logowania.

Umieść oryginalny `test_sm.py` z załącznika w katalogu pakietu. Sprawdź go
w tej samej alokacji pięciu procesów:

```bash
mpiexec python test_sm.py
```

Oczekiwana tablica to:

```text
[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
```

Przykład wymaga dokładnie 5 procesów i wspólnej domeny pamięci. Nie zastępuj tej
liczby 4 z wcześniejszego przykładu Quickstartu. Plik `test_sm.py` pozostaje
oryginalnym załącznikiem; nie jest kopią pliku `test_sito.py` z pakietu.

Następnie, nadal w przydzielonej alokacji:

```bash
mpiexec python sito_mpi.py 100 --mode message --check --print-primes
mpiexec python sito_mpi.py 100 --mode shared --check --print-primes
exit
```

Oba wyniki powinny mieć `count=25` i `check=OK`. Użycie `python plik.py` nie wymaga
`chmod +x`; jest to potrzebne przy bezpośrednim `./plik.py`, jak w Quickstarcie.
Nie uruchamiaj skryptu wsadowego wewnątrz tego interaktywnego testu jako sposobu
wykorzystania jego alokacji: `sbatch` zamawia nowe zadanie.

## 7. Uruchomienie wsadowe i specyfikacja SLURM

Z katalogu pakietu na węźle logowania wyślij zadanie:

```bash
sbatch -A TWOJ_GRANT-cpu sito.slurm 1000000 shared
sbatch -A TWOJ_GRANT-cpu sito.slurm 1000000 message
squeue -u "$USER"
```

Po zakończeniu czytaj `sito-ID.out` i `sito-ID.err`, gdzie ID to numer zadania.
Domyślny skrypt uruchamia test Quickstartu, potem jedno sito; nie uruchamia
oryginalnego `test_sm.py` i nie wykonuje kosztownego `--check` dla dużego n.
Parametry po nazwie skryptu to kolejno n oraz tryb; domyślne to 1000000 i shared.

| Dyrektywa / ustawienie | Znaczenie |
| --- | --- |
| `--job-name=sito-mpi` | Nazwa widoczna w kolejce |
| `--partition=plgrid` | Partycja z Quickstartu, do sprawdzenia dla własnego konta |
| `--nodes=1` | Jeden węzeł dla współdzielonego bufora |
| `--ntasks-per-node=5` | Pięć zadań MPI na węźle |
| `--cpus-per-task=1` | Jeden CPU na proces; nie tworzy wątków |
| `--time=00:10:00` | Limit 10 minut czasu ściennego |
| `--mem=2G` | Pamięć na węzeł, nie na każdy proces |
| `--output`, `--error` | Oddzielne pliki wyjścia i błędów; `%j` to ID zadania |
| `sbatch -A TWOJ_GRANT-cpu` | Konto rozliczeniowe, podane jawnie przy wysyłaniu |

Dyrektywy `#SBATCH` muszą poprzedzać pierwsze polecenie. SLURM interpretuje je
samodzielnie i nie rozwija w nich zmiennych powłoki. Opcje podane do `sbatch`
pozwalają zmienić zasoby, np.:

```bash
sbatch -A TWOJ_GRANT-cpu --ntasks-per-node=1 sito.slurm 1000000 shared
sbatch -A TWOJ_GRANT-cpu --ntasks-per-node=4 sito.slurm 1000000 shared
sbatch -A TWOJ_GRANT-cpu --nodes=2 --ntasks-per-node=4 sito.slurm 1000000 message
```

Ostatni przykład dotyczy wyłącznie trybu message i wymaga uprawnień do takich
zasobów. Tryb shared zgłosi błąd po rozdzieleniu procesów na różne domeny pamięci.

`#!/bin/bash -l` zapewnia powłokę logowania, w której klaster zwykle udostępnia
`module`. `set -euo pipefail` zatrzymuje skrypt przy błędzie polecenia, niezdefiniowanej
zmiennej lub błędzie potoku. `cd "$SLURM_SUBMIT_DIR"` ustawia katalog wysłania
zadania: wszystkie pliki muszą być dostępne tam na węźle obliczeniowym.
Zmienna `MPI_PYTHON_MODULE` pozwala zastąpić historyczną nazwę bez edycji skryptu:

```bash
export MPI_PYTHON_MODULE='NAZWA_MODULU_DOSTEPNEGO_NA_ARESIE'
sbatch -A TWOJ_GRANT-cpu sito.slurm 1000000 shared
```

`SLURM_OVERLAP=1` zachowuje ustawienie z Quickstartu; samo w sobie nie przydziela
zasobów. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` i `MKL_NUM_THREADS` ustawiamy na 1,
aby biblioteki nie tworzyły dodatkowych wątków ponad żądane procesy.
`PYTHONUNBUFFERED=1` ułatwia bieżący odczyt logów. Nie łącz launcherów w postaci
`srun mpiexec ...`; alternatywny bezpośredni start przez `srun` wymaga konfiguracji
MPI właściwej dla klastra, dlatego pakiet pozostaje przy schemacie z załącznika.

## 8. Testy lokalne i interpretacja wyjścia

Poza SLURM, na komputerze z działającym MPI, liczbę procesów podajemy jawnie:

```bash
python -m unittest -v
mpiexec -n 1 python sito_mpi.py 0 --mode shared --check
mpiexec -n 5 python sito_mpi.py 2 --mode shared --check
mpiexec -n 5 python sito_mpi.py 25 --mode message --check --print-primes
mpiexec -n 5 python sito_mpi.py 25 --mode shared --check --print-primes
mpiexec -n 4 python sito_mpi.py 1000000 --mode message
```

Dla n=25 lista to `[2,3,5,7,11,13,17,19,23]`, a `count=9`.
Znane liczności do kontroli: pi(10)=4, pi(100)=25, pi(1000)=168,
pi(1000000)=78498. `--check` porównuje **pełną listę**, nie tylko liczność,
z niezależną definicją pierwszości przez dzielenie próbne. Ograniczono je do
n <= 100000, aby kontrola nie dominowała czasu dużych zadań.

`test_sito.py` porównuje wyniki z dzieleniem próbnym dla n=0..80 oraz 121, 997,
1000, przy 1, 2, 5 i 13 fragmentach. Sprawdza też pełne pokrycie dziedziny.
To test matematyki, nie test synchronizacji MPI. Osobne uruchomienia pod
`mpiexec` są niezbędne do sprawdzenia kolektywów i okien.

Kod zakończenia 0 oznacza poprawne wykonanie, 1 nieudaną kontrolę lub awarię,
a 2 błędne argumenty. Launcher MPI może przekształcić kod awarii na własny.
Pomoc: `python sito_mpi.py --help`.

## 9. Czas, pamięć i skalowanie

Pomiar rozpoczyna się po wspólnej barierze i obejmuje obliczenie B, przygotowanie
flag, wykreślanie, zebranie/odczyt wyniku oraz końcową barierę; w shared także
utworzenie i zwolnienie okna i komunikatora. `reduce(MAX)` wybiera największy
lokalny czas. Nie odejmujemy zegarów różnych procesów, więc ich globalna
synchronizacja nie jest wymagana. Samo końcowe `reduce`, drukowanie i `--check`
są poza pomiarem. Koszt startu interpretera i MPI też jest poza nim.

Klasyczne sito ma koszt O(n log log n) i pamięć O(n). Tutaj każdy proces
powtarza małe sito: koszt około O(sqrt(n) log log sqrt(n)) na proces.
Dla C główna praca wykreślania dzieli się między procesy, ale dochodzą przegląd
pi(sqrt(n)) dzielników w każdym procesie, alokacja, synchronizacja i zbieranie
wyników. Całkowita praca nie maleje P-krotnie. Dla małego n narzut MPI może
przewyższyć zysk z równoległości.

| Cecha | message | shared |
| --- | --- | --- |
| Flagi C | Łącznie około n-limit bajtów, rozdzielone prywatnie | Jedna tablica n+1 bajtów na domenę |
| Dzielniki B | Replikowane w każdym procesie | Także replikowane |
| Wynik | Lokalne tablice, serializacja, lista fragmentów i wynik w procesie 0 | Proces 0 skanuje globalne flagi i tworzy indeksy |
| Zasięg | Jeden lub wiele węzłów | Jedna domena wspólnej pamięci |
| Koszt końcowy | Komunikacja i łączenie | Skanowanie O(n) i synchronizacja |

W message nie ma P pełnych kopii dużej tablicy: są tylko rozłączne lokalne
fragmenty. Shared usuwa przesyłanie list pierwszych i umożliwia bezpośredni odczyt,
ale nie gwarantuje mniejszego zużycia RAM ani szybszego wykonania w każdym
przypadku. Wynik zajmuje dodatkowo około `pi(n)*sizeof(np.intp)` bajtów;
`gather` i `concatenate` chwilowo utrzymują dodatkowe kopie. Interpreter i biblioteki
również zajmują pamięć w każdym procesie. Limit `--mem=2G` jest przykładowy,
nie gwarantuje obsługi dowolnego n. Dla bardzo dużych n potrzebne byłoby sito
segmentowane blokami, zbieranie samych liczności lub strumieniowy zapis wyniku.

Porównując czasy, używaj tego samego n, modułu i zasobów, bez `--check` i bez
`--print-primes`. Zrób kilka powtórzeń dla P=1,2,4,5 i porównuj mediany.
Przyspieszenie S(P)=T(1)/T(P), efektywność E(P)=S(P)/P. Podaj, czy punktem
odniesienia jest ten sam wariant przy P=1. Skanowanie wyniku przez proces 0,
przepustowość pamięci i bariery ograniczają skalowanie.

## 10. Typowe problemy

- `ModuleNotFoundError`: wybrany Python nie zawiera NumPy/mpi4py; sprawdź moduł.
- Błąd ładowania `libmpi`: niezgodne środowisko lub brak biblioteki MPI.
- Każda linia Quickstartu ma `out of 1 processes`: launcher nie uruchamia właściwej
  grupy MPI; nie traktuj takiego testu jako poprawnego.
- Błąd jednej domeny w shared: użyj jednego węzła lub przejdź na message.
- Odrzucenie `sbatch`: sprawdź grant z końcówką `-cpu`, partycję i limity konta.
- `OUT_OF_MEMORY` lub przekroczenie czasu: zmniejsz n albo dostosuj zasoby.
- Brak natychmiastowego wyniku: zadanie może czekać w kolejce; sprawdź `squeue`.

## 11. Źródła i granice weryfikacji środowiska

Podstawą wymagań są trzy załączniki użytkownika: `Sito Erastotenesa równolegle.docx`,
`Quickstart do zadania.docx` oraz `test_sm.py`. Nazwy modułów i sposób uruchamiania
na Aresie pochodzą z Quickstartu. Dokumentacja techniczna sprawdzona przy
przygotowaniu pakietu 20 października 2026:

- [mpi4py 4.0.3 — Comm](https://mpi4py.readthedocs.io/en/4.0.3/reference/mpi4py.MPI.Comm.html)
- [mpi4py 4.0.3 — Win](https://mpi4py.readthedocs.io/en/4.0.3/reference/mpi4py.MPI.Win.html)
- [mpi4py 4.0.3 — moduł MPI i Wtime](https://mpi4py.readthedocs.io/en/4.0.3/reference/mpi4py.MPI.html)
- [mpi4py — tutorial, obiekty i bufory](https://mpi4py.readthedocs.io/en/4.0.3/tutorial.html)
- [MPI 4.1 — Fence i widoczność zapisów](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node328.htm)
- [MPI 4.1 — model pamięci](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node326.htm)
- [Open MPI — Allocate_shared](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Win_allocate_shared.3.html)
- [NumPy — ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
- [NumPy — ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html)
- [NumPy — dtype i itemsize](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html)
- [NumPy — flatnonzero](https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html)
- [NumPy — concatenate](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
- [NumPy — tolist](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html)
- [NumPy — indeksowanie i widoki](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Python — math.isqrt](https://docs.python.org/3/library/math.html#math.isqrt)
- [Python 3.9 — argparse](https://docs.python.org/3.9/library/argparse.html)
- [Python 3.9 — socket.gethostname](https://docs.python.org/3.9/library/socket.html#socket.gethostname)
- [Python — unittest](https://docs.python.org/3/library/unittest.html)
- [SchedMD — sbatch i dyrektywy](https://slurm.schedmd.com/sbatch.html)
- [Tutorial NYU — przesyłanie komunikatów](https://nyu-cds.github.io/python-mpi/02-messagepassing/)
- [Cyfronet — podstawy SLURM, link z zadania](https://kdm.cyfronet.pl/portal/Podstawy:SLURM)

Strona Cyfronetu zwracała błąd HTTP 502 podczas sprawdzania. Nie potwierdzono
aktualnych modułów, grantów ani polityki launcherów Aresa. W ich opisie zachowano
schemat z załącznika i wskazano miejsca wymagające dostosowania. Nie wykonano
logowania ani wysłania zadania na Aresa. Faktyczny zakres lokalnych testów
opisano w `WERYFIKACJA.md`; instrukcje do wykonania nie są deklaracją ich wykonania.
