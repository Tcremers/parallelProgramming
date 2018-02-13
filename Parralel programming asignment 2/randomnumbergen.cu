#include <iostream>
#include <random>

int main()
{
	std::random_device rd; // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator
	std::uniform_int_distribution<> distr(-10, 10); // define the range

	for(int n=0; n<40; ++n)
		std::cout << distr(eng) << ' '; // generate numbers
}