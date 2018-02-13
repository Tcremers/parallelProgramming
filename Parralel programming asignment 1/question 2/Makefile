CXX        = futhark-opencl 

SOURCES_CPP =lssp.fut
EXECUTABLE  =lssp


default: compile


compile: $(EXECUTABLE)
$(EXECUTABLE): 
	$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)


run: $(EXECUTABLE)
	futhark-dataset --i32-bounds=-20:20 -g [8000000]i32 | ./$(EXECUTABLE) -r 10 -t /dev/stderr


clean:
	rm -f $(EXECUTABLE)

