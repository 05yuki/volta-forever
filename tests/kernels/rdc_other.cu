__device__ int add_from_other_object(int value) {
    return value + 37;
}

__device__ void cluster_in_other_object() {
    asm volatile("barrier.cluster.arrive;");
}
