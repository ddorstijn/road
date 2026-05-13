// Traffic sort: exclusive prefix sum scan

struct PushConstants {
    uint num_workgroups;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> histograms;

[shader("compute")]
[numthreads(1, 1, 1)]
void traffic_sort_scan_main(uint3 gid : SV_DispatchThreadID) {
    uint total = 256 * pc.num_workgroups;

    uint sum = 0;
    for (uint i = 0; i < total; i++) {
        uint val = histograms[i];
        histograms[i] = sum;
        sum += val;
    }
}
