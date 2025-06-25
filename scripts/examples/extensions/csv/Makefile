libCsvIO.so: CsvIO.o
	g++ -shared CsvIO.o -o libCsvIO.so -L../../../../build/src/runtime/local/datastructures -lDataStructures -L../../../../thirdparty/installed/lib -lLLVMSupport -lfmt

CsvIO.o: CSVPlugIn.cpp
	g++ -c -fPIC CSVPlugIn.cpp -I../../../../src/ -std=c++17 -O3 -mavx2 -o CsvIO.o
    
clean:
	rm -rf CsvIO.o libCsvIO.so