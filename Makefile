build:
	g++ main.cpp -o app `pkg-config --cflags --libs opencv` -fopenmp -O3 -g -Wall

exec:
	./app
