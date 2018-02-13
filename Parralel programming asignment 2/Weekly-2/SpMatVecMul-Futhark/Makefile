CFLAT        = futhark-opencl
CSEQ		 = futhark-c

SOURCES_FLAT = spMVmult-flat.fut
SOURCES_SEQ = spMVmult-seq.fut
EXECUTABLE_FLAT  = spMVmult-flat
EXECUTABLE_SEQ = spMVmult-seq


default: compile


compile:
	$(CFLAT) -o $(EXECUTABLE_FLAT) $(SOURCES_FLAT)
	$(CSEQ) -o $(EXECUTABLE_SEQ) $(SOURCES_SEQ)


run-flat:
	futhark-dataset --i32-bounds=0:9999 -g [1000000]i32 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i32-bounds=100:100 -g [10000]i32 --f32-bounds=-10.0:10.0 -g [10000]f32 | ./$(EXECUTABLE_FLAT) -t /dev/stderr

run-seq:
	futhark-dataset --i32-bounds=0:9999 -g [1000000]i32 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i32-bounds=100:100 -g [10000]i32 --f32-bounds=-10.0:10.0 -g [10000]f32 | ./$(EXECUTABLE_SEQ) -t /dev/stderr

clean:
	rm -f $(EXECUTABLE_FLAT)
	rm -f $(EXECUTABLE_SEQ)

