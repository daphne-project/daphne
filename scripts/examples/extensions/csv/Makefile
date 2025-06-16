libCsvIO.so: CsvIO.o
	g++ -shared CsvIO.o -L../../../../build/src/runtime/local/datastructures/libDataStructures.a -L../../../../build/src/util/libUtil.a -o libCsvIO.so

CsvIO.o: CSVPlugIn.cpp
	g++ -c -fPIC CSVPlugIn.cpp -I../../../../src/ -std=c++17 -O3 -mavx2 -o CsvIO.o
    
clean:
	rm -rf CsvIO.o libCsvIO.so