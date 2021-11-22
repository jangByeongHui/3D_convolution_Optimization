
# If you just type "make", it will default to "make all".
make:
	nvcc conv3d.cu -o conv3d

	g++ -mavx2 -mfma -o conv3d_avx conv3d_avx.cpp

	g++ -mavx2 -mfma -o conv3d_avx_mul conv3d_avx_mul.cpp -lpthread


cuda:
	nvcc conv3d.cu -o conv3d
avx:
	g++ -mavx2 -mfma -o conv3d_avx conv3d_avx.cpp
avx_mul:
	g++ -mavx2 -mfma -o conv3d_avx_mul conv3d_avx_mul.cpp -lpthread

test1:
	./conv3d -a
	./conv3d -b
	./conv3d -c
	./conv3d -d
	./conv3d -e

test2:
	./conv3d_avx -a
	./conv3d_avx -b
	./conv3d_avx -c
	./conv3d_avx -d
	./conv3d_avx -e
	
test3:
	./conv3d_avx_mul -a
	./conv3d_avx_mul -b
	./conv3d_avx_mul -c
	./conv3d_avx_mul -d
	./conv3d_avx_mul -e


all:
	./conv3d -a
	./conv3d -b
	./conv3d -c
	./conv3d -d
	./conv3d -e

clean:
	rm -rf conv3d conv3d_avx conv3d_avx_mul