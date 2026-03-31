NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -gencode arch=compute_86,code=sm_86 -I../../

TARGETS = cc_profile sssp_profile pagerank_profile

all: $(TARGETS)

cc_profile: cc_profile.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

sssp_profile: sssp_profile.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

pagerank_profile: pagerank_profile.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
