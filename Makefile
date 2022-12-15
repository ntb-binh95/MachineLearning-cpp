main: main.cpp gemm.cpp
	g++ -o main main.cpp gemm.cpp

logistic: LogisticRegression.cpp gemm.cpp
	g++ -o logistic LogisticRegression.cpp gemm.cpp

linear: LinearRegression.cpp gemm.cpp
	g++ -o linear LinearRegression.cpp gemm.cpp

clean:
	rm main linear logistic