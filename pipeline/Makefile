CC     = gcc
CFLAGS = -O3 -march=native -fno-math-errno -ffinite-math-only \
         -fno-signed-zeros -fno-trapping-math -fopenmp -std=c99
TARGET = hmf_sigma.so
SOURCE = hmf_sigma.c

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -shared -fPIC -o $(TARGET) $(SOURCE) -lm

clean:
	rm -f $(TARGET)