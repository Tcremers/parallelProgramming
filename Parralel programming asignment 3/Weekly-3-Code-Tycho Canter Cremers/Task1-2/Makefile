CXX        = nvcc

SOURCES_CPP =MatTranspose.cu
HELPERS     =MatTranspose.cu.h MatTranspose.h
EXECUTABLE  =matrixMult


default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)
$(EXECUTABLE): 
	$(CXX) -o $(EXECUTABLE)_1 $(SOURCES_CPP) -DVERSION=1
	$(CXX) -o $(EXECUTABLE)_2 $(SOURCES_CPP) -DVERSION=2
	$(CXX) -o $(EXECUTABLE)_3 $(SOURCES_CPP) -DVERSION=3


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)_1
	rm -f $(EXECUTABLE)_2
	rm -f $(EXECUTABLE)_3
	
