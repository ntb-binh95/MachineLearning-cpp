main: main.cpp gemm.cpp
	g++ -o main main.cpp gemm.cpp

logicR: LogisticRegression.cpp gemm.cpp
	g++ -o logicR LogisticRegression.cpp gemm.cpp