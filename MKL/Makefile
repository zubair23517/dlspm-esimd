all : onemkl_test.hpp onemkl_test.cpp
	icpx -fsycl -qmkl onemkl_test.cpp -o onemkl_test.out

clean :
	rm -f *.out
