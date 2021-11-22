
# If you just type "make", it will default to "make all".
make:
	nvcc conv3d.cu -o conv3d

	g++ -mavx2 -mfma -o avx_conv3d avx_conv3d.cpp
cuda:
	nvcc conv3d.cu -o conv3d
avx:
	g++ -mavx2 -mfma -o conv3d_avx conv3d_avx.cpp

test:
	./conv3d_avx -a


test1:
	./conv3d -a
test2:
	./conv3d -b
test3:
	./conv3d -c
test4:
	./conv3d -d
test5:
	./conv3d -e

atest1:
	./conv3d_avx -a
atest2:
	./conv3d_avx -b
atest3:
	./conv3d_avx -c
atest4:
	./conv3d_avx -d
atest5:
	./conv3d_avx -e

all2:
	./conv3d_avx -a
	./conv3d_avx -b
	./conv3d_avx -c
	./conv3d_avx -d
	./conv3d_avx -e
	

all:
	./conv3d -a
	./conv3d -b
	./conv3d -c
	./conv3d -d
	./conv3d -e

clean:
	rm -rf conv3d conv3d_avx