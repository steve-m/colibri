// Vulkan compute backend for colibri's quantized matmul, targeting the
// Strix Halo iGPU (RADV gfx1151). Mirrors backend_cuda.c's contract but
// exploits unified memory: weight "uploads" write into HOST_VISIBLE|
// DEVICE_LOCAL memory — the same physical RAM the iGPU reads — so there is
// no PCIe copy. That is what makes offloading *streamed experts* profitable
// here, which the discrete-CUDA path deliberately avoids.
//
// M2 scope: correctness + a standalone GPU-vs-CPU test harness. Synchronous
// submit/wait per call; async queues and zero-copy import come in M4.
#include "backend_vulkan.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VKCHECK(x, what) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[VK] %s failed: %d\n", what, _r); return 0; } } while (0)

struct ColiVkTensor {
    VkBuffer wbuf, sbuf;
    VkDeviceMemory wmem, smem;
    size_t wbytes;
    int fmt, I, O, rowWords;
};

typedef struct {
    VkBuffer buf; VkDeviceMemory mem; void *ptr; size_t cap;
} Scratch;

static struct {
    int ready;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    uint32_t memtype;            // HOST_VISIBLE|HOST_COHERENT (prefer DEVICE_LOCAL)
    VkDescriptorSetLayout dsl;
    VkPipelineLayout plyt;
    VkPipeline pipe;
    VkShaderModule shader;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;
    /* fused dual gate+up+silu pipeline (6 bindings): x, Wg, gscale, Wu, uscale, hidden */
    VkShaderModule shader_gu; VkDescriptorSetLayout dsl_gu; VkPipelineLayout plyt_gu;
    VkPipeline pipe_gu; VkDescriptorPool dpool_gu; VkDescriptorSet dset_gu;
    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkFence fence;
    Scratch x, y, h;   /* h = fused gate+up hidden output */
    /* full expert-group scratch: activations/hidden/output for K experts + per-expert
     * descriptor sets (gate_up: dsl_gu, down: dsl), so gate_up->down runs on-device in
     * one submit with hidden never leaving the GPU. */
    Scratch eg_x, eg_h, eg_y;
    VkDescriptorPool eg_pool; VkDescriptorSet eg_gu[64], eg_dn[64]; int eg_nsets;
    /* resubmit cache: skip vkUpdateDescriptorSets + command re-record when the bound
     * tensor / shape / scratch buffers are unchanged from the previous call (the hot-
     * expert-called-repeatedly pattern). The synchronous fence wait each call means no
     * submission is ever in flight, so rebinding/re-recording only when something
     * actually changed is safe. */
    ColiVkTensor *bound_tensor; int bound_S, bound_I, bound_O, cmd_ready;
    VkBuffer bound_xbuf, bound_ybuf;
    size_t used_bytes, tensor_count;
} G;

struct PC { int fmt, S, I, O, rowWords; };

static int pick_memtype(void) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(G.phys, &m);
    int best = -1;
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return (int)i; // ideal on APU
            if (best < 0) best = (int)i;
        }
    }
    return best;
}

static int alloc_hostvis(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = G.memtype};
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, *mem, 0), "vkBindBufferMemory");
    if (ptr) VKCHECK(vkMapMemory(G.dev, *mem, 0, bytes, 0, ptr), "vkMapMemory");
    return 1;
}

static int scratch_reserve(Scratch *s, size_t bytes) {
    if (s->cap >= bytes) return 1;
    if (s->buf) { vkDestroyBuffer(G.dev, s->buf, NULL); vkFreeMemory(G.dev, s->mem, NULL); }
    s->buf = VK_NULL_HANDLE; s->cap = 0; s->ptr = NULL;
    if (!alloc_hostvis(bytes, &s->buf, &s->mem, &s->ptr)) return 0;
    s->cap = bytes;
    return 1;
}

static int rowwords(int fmt, int I) {
    size_t rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2; // bytes/row on CPU side
    return (int)((rb + 3) / 4);                              // padded to uint32
}

static VkShaderModule load_spv(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[VK] cannot open %s\n", path); return VK_NULL_HANDLE; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) {   // SPIR-V is a stream of uint32; empty/non-seekable/odd size is invalid
        fprintf(stderr, "[VK] bad SPIR-V size %ld in %s\n", n, path); fclose(f); return VK_NULL_HANDLE; }
    uint32_t *code = malloc((size_t)n);
    if (!code) { fclose(f); return VK_NULL_HANDLE; }
    if (fread(code, 1, n, f) != (size_t)n) { fclose(f); free(code); return VK_NULL_HANDLE; }
    fclose(f);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = n, .pCode = code};
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(G.dev, &si, NULL, &m);
    free(code);
    return r == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

/* Build a compute pipeline + descriptor pool/set for nbind storage buffers sharing the
 * struct PC push constant. Used for both the 4-binding matmul and 6-binding gate_up. */
static int build_pipeline(int nbind, VkShaderModule shader, VkDescriptorSetLayout *dsl,
                          VkPipelineLayout *plyt, VkPipeline *pipe, VkDescriptorPool *dpool,
                          VkDescriptorSet *dset) {
    VkDescriptorSetLayoutBinding b[8];
    for (int i = 0; i < nbind; i++) b[i] = (VkDescriptorSetLayoutBinding){
        .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)nbind, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(G.dev, &dsli, NULL, dsl), "descSetLayout");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(struct PC)};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(G.dev, &pli, NULL, plyt), "pipelineLayout");
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = *plyt};
    VKCHECK(vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe), "pipeline");
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)nbind};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, dpool), "descPool");
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = *dpool, .descriptorSetCount = 1, .pSetLayouts = dsl};
    VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, dset), "allocDescSet");
    return 1;
}

/* "…/qmatmul.spv" -> "…/qmatmul_gate_up.spv" (sibling of the main shader). */
static void derive_gu_path(const char *spv, char *out, size_t n) {
    const char *dot = strstr(spv, ".spv");
    if (dot && (size_t)(dot - spv) + 13 < n) {
        size_t pre = (size_t)(dot - spv);
        memcpy(out, spv, pre); strcpy(out + pre, "_gate_up.spv");
    } else snprintf(out, n, "%s", spv);
}

int coli_vk_init(const char *spv_path) {
    if (G.ready) return 1;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app};
    VKCHECK(vkCreateInstance(&ici, NULL, &G.inst), "vkCreateInstance");

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, NULL);
    if (!nd) { fprintf(stderr, "[VK] no devices\n"); return 0; }
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    vkEnumeratePhysicalDevices(G.inst, &nd, devs);
    // Prefer a real GPU over a CPU/software device (llvmpipe) on multi-adapter hosts:
    // discrete > integrated > virtual > other/cpu. Falls back to devs[0] if all equal.
    G.phys = devs[0];
    int bestrank = -1;
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
        int rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 4 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER          ? 1 : 0; // CPU last
        if (rank > bestrank) { bestrank = rank; G.phys = devs[i]; }
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, qf);
    G.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { G.qfam = i; break; }
    if (G.qfam == UINT32_MAX) { fprintf(stderr, "[VK] no compute queue\n"); return 0; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G.qfam, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
    VKCHECK(vkCreateDevice(G.phys, &di, NULL, &G.dev), "vkCreateDevice");
    vkGetDeviceQueue(G.dev, G.qfam, 0, &G.queue);

    int mt = pick_memtype();
    if (mt < 0) { fprintf(stderr, "[VK] no host-visible memory\n"); return 0; }
    G.memtype = (uint32_t)mt;

    G.shader = load_spv(spv_path);
    if (!G.shader) return 0;
    if (!build_pipeline(4, G.shader, &G.dsl, &G.plyt, &G.pipe, &G.dpool, &G.dset)) return 0;

    /* Optional fused gate+up pipeline: skip gracefully if its shader isn't present
     * (single-matmul path keeps working). */
    char gu_path[512]; derive_gu_path(spv_path, gu_path, sizeof(gu_path));
    G.shader_gu = load_spv(gu_path);
    if (G.shader_gu && !build_pipeline(6, G.shader_gu, &G.dsl_gu, &G.plyt_gu, &G.pipe_gu, &G.dpool_gu, &G.dset_gu))
        return 0;

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G.qfam};
    VKCHECK(vkCreateCommandPool(G.dev, &cpci, NULL, &G.cpool), "cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.cmd), "cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence), "fence");

    G.ready = 1;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(G.phys, &p);
    fprintf(stderr, "[VK] ready: %s, compute qfam %u, memtype %u%s\n", p.deviceName, G.qfam, G.memtype,
            G.shader_gu ? ", fused gate+up" : "");
    return 1;
}

int coli_vk_available(void) { return G.ready; }

void coli_vk_mem_info(size_t *used, size_t *count) {
    if (used) *used = G.used_bytes;
    if (count) *count = G.tensor_count;
}

static int upload_tensor(ColiVkTensor **out, const void *weights, const float *scales,
                         int fmt, int I, int O) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I);
    size_t stride = (size_t)t->rowWords * 4;         // padded row bytes
    size_t cpu_rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
    t->wbytes = stride * (size_t)O;
    void *wptr;
    if (!alloc_hostvis(t->wbytes, &t->wbuf, &t->wmem, &wptr)) { free(t); return 0; }
    memset(wptr, 0, t->wbytes);
    for (int o = 0; o < O; o++)                        // copy row-by-row into padded layout
        memcpy((uint8_t *)wptr + (size_t)o * stride,
               (const uint8_t *)weights + (size_t)o * cpu_rb, cpu_rb);
    void *sptr;
    if (!alloc_hostvis((size_t)O * sizeof(float), &t->sbuf, &t->smem, &sptr)) {
        vkDestroyBuffer(G.dev, t->wbuf, NULL); vkFreeMemory(G.dev, t->wmem, NULL); free(t); return 0;
    }
    memcpy(sptr, scales, (size_t)O * sizeof(float));
    // Counters are touched concurrently: frees run from expert_load under
    // `#pragma omp parallel`, so RMW them atomically (torn counts otherwise).
    __atomic_add_fetch(&G.used_bytes, t->wbytes + (size_t)O * sizeof(float), __ATOMIC_RELAXED);
    __atomic_add_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    *out = t;
    return 1;
}

int coli_vk_matmul(ColiVkTensor **tensor, float *y, const float *x,
                   const void *weights, const float *scales,
                   int fmt, int S, int I, int O) {
    if (!G.ready || S < 1 || !upload_tensor(tensor, weights, scales, fmt, I, O)) return 0;
    ColiVkTensor *t = *tensor;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    VkBuffer old_x = G.x.buf, old_y = G.y.buf;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve(&G.y, yb)) return 0;
    memcpy(G.x.ptr, x, xb);

    /* Rebind descriptors only when the tensor or a scratch buffer changed (a realloc
     * makes the old VkBuffer handle stale); otherwise the previous binding is still valid. */
    int rebind = G.bound_tensor != t || G.x.buf != old_x || G.y.buf != old_y
              || G.bound_xbuf != G.x.buf || G.bound_ybuf != G.y.buf;
    if (rebind) {
        VkDescriptorBufferInfo bi[4] = {
            {.buffer = G.x.buf, .range = VK_WHOLE_SIZE},
            {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
            {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},
            {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[4];
        for (int i = 0; i < 4; i++) w[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset,
            .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
        vkUpdateDescriptorSets(G.dev, 4, w, 0, NULL);
        G.bound_tensor = t; G.bound_xbuf = G.x.buf; G.bound_ybuf = G.y.buf;
    }

    /* Re-record the command buffer only when the binding or the dispatch shape changed.
     * Recorded WITHOUT one-time-submit so the same buffer can be resubmitted verbatim —
     * for repeated calls to the same expert this drops setup to a bare submit+wait. */
    if (rebind || !G.cmd_ready || G.bound_S != S || G.bound_I != I || G.bound_O != O) {
        VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
        vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
        struct PC pc = {fmt, S, I, O, t->rowWords};
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        /* Grid-stride shader: one subgroup per output row (~8 rows/workgroup at wave32).
         * Launch ~O/8 workgroups for occupancy; the shader loops to cover any O / wave width. */
        vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), (uint32_t)S, 1);
        VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
        G.cmd_ready = 1; G.bound_S = S; G.bound_I = I; G.bound_O = O;
    }

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    // Bounded wait: a GPU hang/TDR must never wedge the process. 10s is orders of
    // magnitude over a single-GEMV dispatch; on timeout/device-loss disable VK for
    // the rest of the run and fall back to CPU (the caller degrades on our 0 return).
    VkResult wr = vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    if (wr != VK_SUCCESS) {
        fprintf(stderr, "[VK] fence wait failed: %d — disabling GPU offload, staying on CPU\n", wr);
        G.ready = 0;
        return 0;
    }
    memcpy(y, G.y.ptr, yb);
    return 1;
}

/* Fused first half of the expert MLP: hidden = silu(gate(x)) * up(x), computed in ONE
 * dispatch that reads x once for both projections. gate/up are resident (uploaded on
 * first call). D = input (hidden) dim, I = moe_inter. Returns 0 -> caller falls back. */
int coli_vk_gate_up(ColiVkTensor **gate, ColiVkTensor **up, float *hidden, const float *x,
                    const void *gw, const float *gs, const void *uw, const float *us,
                    int fmt, int S, int D, int I) {
    if (!G.ready || !G.shader_gu || S < 1) return 0;
    if (!upload_tensor(gate, gw, gs, fmt, D, I) || !upload_tensor(up, uw, us, fmt, D, I)) return 0;
    ColiVkTensor *tg = *gate, *tu = *up;
    size_t xb = (size_t)S * D * sizeof(float), hb = (size_t)S * I * sizeof(float);
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve(&G.h, hb)) return 0;
    memcpy(G.x.ptr, x, xb);

    VkDescriptorBufferInfo bi[6] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tg->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tg->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = tu->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tu->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.h.buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset_gu,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, 6, w, 0, NULL);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.dset_gu, 0, NULL);
    struct PC pc = {fmt, S, D, I, tg->rowWords};   // PC.I = input D, PC.O = moe_inter I
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)((I + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL) != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(hidden, G.h.ptr, hb);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

static void wr_desc(VkDescriptorSet set, int n, const VkDescriptorBufferInfo *bi) {
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < n; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = (uint32_t)i,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, (uint32_t)n, w, 0, NULL);
}

/* Full batched expert MLP for `count` experts in ONE submit, hidden staying on-device:
 * for each c, hidden_c = silu(gate_c(x_c))*up_c(x_c) (fused), then y_c = down_c(hidden_c).
 * x/y are packed [sum(rows)*D]; experts are resident VkTensors (gate/up: D->I, down: I->D).
 * Mirrors coli_cuda_expert_group. Returns 0 -> caller falls back to CPU. */
int coli_vk_expert_group(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                         ColiVkTensor *const *downs, const int *rows, int count,
                         float *y, const float *x) {
    if (!G.ready || !G.shader_gu || count < 1 || count > 64) return 0;
    ColiVkTensor *g0 = gates[0]; if (!g0) return 0;
    int D = g0->I, I = g0->O, fmt = g0->fmt, total = 0, off[64];
    for (int c = 0; c < count; c++) {
        off[c] = total; total += rows[c];
        if (rows[c] < 1 || gates[c]->I != D || gates[c]->O != I || gates[c]->fmt != fmt ||
            ups[c]->I != D || ups[c]->O != I || downs[c]->I != I || downs[c]->O != D) return 0;
    }
    size_t xb = (size_t)total*D*4, hb = (size_t)total*I*4, yb = (size_t)total*D*4;
    if (!scratch_reserve(&G.eg_x, xb) || !scratch_reserve(&G.eg_h, hb) || !scratch_reserve(&G.eg_y, yb)) return 0;
    memcpy(G.eg_x.ptr, x, xb);

    if (!G.eg_pool) {   /* one-time: 64 gate_up (6-binding) + 64 down (4-binding) sets */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 64*6 + 64*4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 128, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.eg_pool), "eg descPool");
        VkDescriptorSetLayout lg[64], ld[64];
        for (int c = 0; c < 64; c++) { lg[c] = G.dsl_gu; ld[c] = G.dsl; }
        VkDescriptorSetAllocateInfo ag = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = lg};
        VkDescriptorSetAllocateInfo ad = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = ld};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ag, G.eg_gu), "eg gu sets");
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ad, G.eg_dn), "eg dn sets");
        G.eg_nsets = 64;
    }
    for (int c = 0; c < count; c++) {
        VkDeviceSize xo = (VkDeviceSize)off[c]*D*4, ho = (VkDeviceSize)off[c]*I*4, yo = (VkDeviceSize)off[c]*D*4;
        VkDescriptorBufferInfo gi[6] = {
            {G.eg_x.buf, xo, (VkDeviceSize)rows[c]*D*4}, {gates[c]->wbuf, 0, VK_WHOLE_SIZE},
            {gates[c]->sbuf, 0, VK_WHOLE_SIZE}, {ups[c]->wbuf, 0, VK_WHOLE_SIZE},
            {ups[c]->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}};
        wr_desc(G.eg_gu[c], 6, gi);
        VkDescriptorBufferInfo di[4] = {
            {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}, {downs[c]->wbuf, 0, VK_WHOLE_SIZE},
            {downs[c]->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_y.buf, yo, (VkDeviceSize)rows[c]*D*4}};
        wr_desc(G.eg_dn[c], 4, di);
    }

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    /* phase 1: fused gate+up+silu -> hidden (per expert, bound to its x/hidden slices) */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    for (int c = 0; c < count; c++) {
        struct PC pc = {fmt, rows[c], D, I, gates[c]->rowWords};
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.eg_gu[c], 0, NULL);
        vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.cmd, (uint32_t)((I + 7) / 8), (uint32_t)rows[c], 1);
    }
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* phase 2: down projection hidden -> y */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    for (int c = 0; c < count; c++) {
        struct PC pc = {fmt, rows[c], I, D, downs[c]->rowWords};
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.eg_dn[c], 0, NULL);
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.cmd, (uint32_t)((D + 7) / 8), (uint32_t)rows[c], 1);
    }
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL) != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(y, G.eg_y.ptr, yb);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

void coli_vk_tensor_free(ColiVkTensor *t) {
    if (!t) return;
    if (G.bound_tensor == t) { G.bound_tensor = NULL; G.cmd_ready = 0; }  /* drop stale cache */
    // If the device was lost/disabled (the fence-timeout path sets G.ready=0), a submission
    // may still reference these buffers — do NOT vkDestroy into a dead device (GPU-side UAF).
    // Leak the GPU handles (we're degrading to CPU for the rest of the run) and reclaim the
    // host struct + counters only.
    if (G.ready) {
        if (t->wbuf) { vkDestroyBuffer(G.dev, t->wbuf, NULL); vkFreeMemory(G.dev, t->wmem, NULL); }
        if (t->sbuf) { vkDestroyBuffer(G.dev, t->sbuf, NULL); vkFreeMemory(G.dev, t->smem, NULL); }
    }
    // Mirror upload_tensor exactly (weights + scales), atomically — otherwise
    // used_bytes leaks the O*float scales buffer on every free and drifts upward.
    __atomic_sub_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&G.used_bytes, t->wbytes + (size_t)t->O * sizeof(float), __ATOMIC_RELAXED);
    free(t);
}

size_t coli_vk_tensor_bytes(const ColiVkTensor *t) { return t ? t->wbytes : 0; }

void coli_vk_shutdown(void) {
    if (!G.ready) return;
    vkDeviceWaitIdle(G.dev);
    if (G.x.buf) { vkDestroyBuffer(G.dev, G.x.buf, NULL); vkFreeMemory(G.dev, G.x.mem, NULL); }
    if (G.y.buf) { vkDestroyBuffer(G.dev, G.y.buf, NULL); vkFreeMemory(G.dev, G.y.mem, NULL); }
    if (G.h.buf) { vkDestroyBuffer(G.dev, G.h.buf, NULL); vkFreeMemory(G.dev, G.h.mem, NULL); }
    if (G.eg_x.buf) { vkDestroyBuffer(G.dev, G.eg_x.buf, NULL); vkFreeMemory(G.dev, G.eg_x.mem, NULL); }
    if (G.eg_h.buf) { vkDestroyBuffer(G.dev, G.eg_h.buf, NULL); vkFreeMemory(G.dev, G.eg_h.mem, NULL); }
    if (G.eg_y.buf) { vkDestroyBuffer(G.dev, G.eg_y.buf, NULL); vkFreeMemory(G.dev, G.eg_y.mem, NULL); }
    if (G.eg_pool) vkDestroyDescriptorPool(G.dev, G.eg_pool, NULL);
    vkDestroyFence(G.dev, G.fence, NULL);
    vkDestroyCommandPool(G.dev, G.cpool, NULL);
    vkDestroyDescriptorPool(G.dev, G.dpool, NULL);
    vkDestroyPipeline(G.dev, G.pipe, NULL);
    vkDestroyPipelineLayout(G.dev, G.plyt, NULL);
    vkDestroyDescriptorSetLayout(G.dev, G.dsl, NULL);
    vkDestroyShaderModule(G.dev, G.shader, NULL);
    if (G.shader_gu) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gu, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gu, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gu, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gu, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gu, NULL);
    }
    vkDestroyDevice(G.dev, NULL);
    vkDestroyInstance(G.inst, NULL);
    memset(&G, 0, sizeof(G));
}

#ifdef VK_TEST
// ---- standalone GPU-vs-CPU validation + microbench --------------------------
#include <math.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9; }

static float deq(const uint8_t *row, int fmt, int i) {
    if (fmt == 1) { int b = ((const int8_t *)row)[i]; return (float)b; }
    uint8_t v = row[i >> 1]; int nib = (i & 1) ? (v >> 4) : (v & 15); return (float)(nib - 8);
}

static void cpu_ref(float *y, const float *x, const uint8_t *w, const float *sc,
                    int fmt, int S, int I, int O) {
    size_t rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double sum = 0; const uint8_t *row = w + (size_t)o * rb;
        for (int i = 0; i < I; i++) sum += x[s * I + i] * deq(row, fmt, i);
        y[s * O + o] = (float)(sum * sc[o]);
    }
}

static int run_case(int fmt, int S, int I, int O, int iters) {
    size_t rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
    float *x = malloc((size_t)S * I * sizeof(float));
    uint8_t *w = malloc(rb * O);
    float *sc = malloc((size_t)O * sizeof(float));
    float *yg = malloc((size_t)S * O * sizeof(float));
    float *yc = malloc((size_t)S * O * sizeof(float));
    for (int i = 0; i < S * I; i++) x[i] = (float)((rand() % 200 - 100) / 100.0);
    for (size_t i = 0; i < rb * O; i++) w[i] = rand() & 0xff;
    for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;

    ColiVkTensor *t = NULL;
    if (!coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O)) { printf("matmul failed\n"); return 1; }
    cpu_ref(yc, x, w, sc, fmt, S, I, O);
    double maxerr = 0, maxrel = 0;
    for (int i = 0; i < S * O; i++) {
        double e = fabs(yg[i] - yc[i]); if (e > maxerr) maxerr = e;
        if (fabs(yc[i]) > 1e-2) { double r = e / fabs(yc[i]); if (r > maxrel) maxrel = r; }
    }
    // microbench (GPU)
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O);
    double gpu_ms = (now() - t0) * 1000 / iters;
    // microbench (CPU ref, 1 iter — it's slow)
    double c0 = now(); cpu_ref(yc, x, w, sc, fmt, S, I, O); double cpu_ms = (now() - c0) * 1000;
    printf("fmt=%d S=%d I=%d O=%d | maxerr=%.4g maxrel=%.4g | gpu=%.3f ms  cpu_ref=%.3f ms\n",
           fmt, S, I, O, maxerr, maxrel, gpu_ms, cpu_ms);
    coli_vk_tensor_free(t);
    free(x); free(w); free(sc); free(yg); free(yc);
    return maxrel > 1e-3 ? 1 : 0;
}

/* Batched throughput: record N dispatches in ONE command buffer, one submit + one
 * fence wait — the amortized per-matmul cost with the submit roundtrip spread across
 * the batch (how the real expert tier would drive it). Reuses the descriptor binding
 * left by a prior coli_vk_matmul call for this tensor/shape. */
static double bench_batched(ColiVkTensor *t, const float *x, int fmt, int S, int I, int O, int N) {
    size_t xb = (size_t)S * I * sizeof(float);
    memcpy(G.x.ptr, x, xb);
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC pc = {fmt, S, I, O, t->rowWords};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    for (int i = 0; i < N; i++) {
        vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), (uint32_t)S, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int warm = 0; warm < 2; warm++) {
        vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
        vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) {
        vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
        vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    }
    G.cmd_ready = 0;   /* we clobbered the cached command buffer */
    return (now() - t0) * 1000.0 / iters / N;   /* ms per matmul, roundtrip amortized */
}

/* Fused gate+up correctness vs CPU ref: hidden = silu(gate*x)*(up*x). */
static int run_gate_up(int fmt, int S, int D, int I) {
    size_t rb = fmt == 1 ? (size_t)D : (size_t)(D + 1) / 2;
    float *x = malloc((size_t)S*D*4); uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
    float *gs = malloc((size_t)I*4), *us = malloc((size_t)I*4);
    float *hg = malloc((size_t)S*I*4), *hc = malloc((size_t)S*I*4);
    for (int i = 0; i < S*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
    for (int o = 0; o < I; o++) { gs[o] = 0.01f+(rand()%100)/10000.0f; us[o] = 0.01f+(rand()%100)/10000.0f; }
    ColiVkTensor *tg = NULL, *tu = NULL;
    if (!coli_vk_gate_up(&tg, &tu, hg, x, gw, gs, uw, us, fmt, S, D, I)) { printf("gate_up failed\n"); return 1; }
    for (int s = 0; s < S; s++) for (int o = 0; o < I; o++) {
        double g = 0, u = 0; const uint8_t *gr = gw+(size_t)o*rb, *ur = uw+(size_t)o*rb;
        for (int i = 0; i < D; i++) { g += x[s*D+i]*deq(gr,fmt,i); u += x[s*D+i]*deq(ur,fmt,i); }
        float gt = (float)(g*gs[o]), ut = (float)(u*us[o]);
        hc[s*I+o] = (gt/(1.0f+expf(-gt)))*ut;
    }
    double maxrel = 0;
    for (int i = 0; i < S*I; i++) { double e = fabs(hg[i]-hc[i]); if (fabs(hc[i])>1e-2) { double r = e/fabs(hc[i]); if (r>maxrel) maxrel = r; } }
    printf("gate_up(fused) fmt=%d S=%d D=%d I=%d | maxrel=%.4g\n", fmt, S, D, I, maxrel);
    coli_vk_tensor_free(tg); coli_vk_tensor_free(tu);
    free(x); free(gw); free(uw); free(gs); free(us); free(hg); free(hc);
    return maxrel > 1e-3 ? 1 : 0;
}

/* Batched throughput of the fused gate_up (N dispatches / one submit). */
static double bench_gu_batched(ColiVkTensor *tg, const float *x, int fmt, int S, int D, int I, int N) {
    memcpy(G.x.ptr, x, (size_t)S*D*sizeof(float));
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.dset_gu, 0, NULL);
    struct PC pc = {fmt, S, D, I, tg->rowWords};
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    for (int i = 0; i < N; i++) {
        vkCmdDispatch(G.cmd, (uint32_t)((I+7)/8), (uint32_t)S, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    return (now()-t0)*1000.0/iters/N;
}

/* FAIR fused-gate_up throughput: cycle K DISTINCT experts (own descriptor set each) so
 * weights come from VRAM, not L2 — matching ROCm's expert_group reading distinct experts.
 * Returns ms per gate_up (one expert). */
static double bench_experts_fair(int fmt, int D, int I, int K, int Npass) {
    if (K > 32) K = 32;
    size_t rb = fmt == 1 ? (size_t)D : (size_t)(D + 1) / 2;
    ColiVkTensor *tg[32] = {0}, *tu[32] = {0};
    float *h = malloc((size_t)I*4), *x = malloc((size_t)D*4);
    for (int i = 0; i < D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I); float *gs = malloc((size_t)I*4), *us = malloc((size_t)I*4);
        for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
        for (int o = 0; o < I; o++) { gs[o] = 0.01f; us[o] = 0.01f; }
        coli_vk_gate_up(&tg[c], &tu[c], h, x, gw, gs, uw, us, fmt, 1, D, I);   /* uploads distinct experts */
        free(gw); free(uw); free(gs); free(us);
    }
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)(6*K)};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = (uint32_t)K, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool pool; vkCreateDescriptorPool(G.dev, &dpi, NULL, &pool);
    VkDescriptorSetLayout lays[32]; VkDescriptorSet sets[32]; for (int c = 0; c < K; c++) lays[c] = G.dsl_gu;
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = (uint32_t)K, .pSetLayouts = lays};
    vkAllocateDescriptorSets(G.dev, &dsa, sets);
    memcpy(G.x.ptr, x, (size_t)D*4);
    for (int c = 0; c < K; c++) {
        VkDescriptorBufferInfo bi[6] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=tg[c]->wbuf,.range=VK_WHOLE_SIZE},{.buffer=tg[c]->sbuf,.range=VK_WHOLE_SIZE},{.buffer=tu[c]->wbuf,.range=VK_WHOLE_SIZE},{.buffer=tu[c]->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.h.buf,.range=VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[6]; for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[c],.dstBinding=(uint32_t)i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
        vkUpdateDescriptorSets(G.dev, 6, w, 0, NULL);
    }
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    struct PC pc = {fmt, 1, D, I, tg[0]->rowWords};
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    for (int pass = 0; pass < Npass; pass++) for (int c = 0; c < K; c++) {
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &sets[c], 0, NULL);
        vkCmdDispatch(G.cmd, (uint32_t)((I+7)/8), 1, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    double ms = (now()-t0)*1000.0/iters/((double)Npass*K);
    vkDestroyDescriptorPool(G.dev, pool, NULL);
    for (int c = 0; c < K; c++) { coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); }
    free(h); free(x); G.cmd_ready = 0; G.bound_tensor = NULL;
    return ms;
}

/* Full expert-group correctness (vs CPU ref) + fair throughput: K distinct experts,
 * one submit, hidden on-device. The real comparison to ROCm's coli_cuda_expert_group. */
static int run_expert_group(int fmt, int D, int I, int K) {
    if (K > 64) K = 64;
    size_t gu_rb = fmt == 1 ? (size_t)D : (size_t)(D + 1) / 2;
    size_t d_rb  = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
    ColiVkTensor *tg[64] = {0}, *tu[64] = {0}, *td[64] = {0};
    uint8_t *hgw[64], *huw[64], *hdw[64]; float *hgs[64], *hus[64], *hds[64];
    float *x = malloc((size_t)K*D*4), *yg = malloc((size_t)K*D*4), *yc = malloc((size_t)K*D*4);
    float *tmp = malloc((size_t)(D > I ? D : I) * 4);
    for (int i = 0; i < K*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        hgw[c] = malloc(gu_rb*I); huw[c] = malloc(gu_rb*I); hdw[c] = malloc(d_rb*D);
        for (size_t i = 0; i < gu_rb*I; i++) { hgw[c][i] = rand()&0xff; huw[c][i] = rand()&0xff; }
        for (size_t i = 0; i < d_rb*D; i++) hdw[c][i] = rand()&0xff;
        hgs[c] = malloc((size_t)I*4); hus[c] = malloc((size_t)I*4); hds[c] = malloc((size_t)D*4);
        for (int o = 0; o < I; o++) { hgs[c][o] = 0.01f+(rand()%100)/10000.0f; hus[c][o] = 0.01f+(rand()%100)/10000.0f; }
        for (int o = 0; o < D; o++) hds[c][o] = 0.01f+(rand()%100)/10000.0f;
        coli_vk_matmul(&tg[c], tmp, x, hgw[c], hgs[c], fmt, 1, D, I);   /* upload gate  (D->I) */
        coli_vk_matmul(&tu[c], tmp, x, huw[c], hus[c], fmt, 1, D, I);   /* upload up    (D->I) */
        coli_vk_matmul(&td[c], tmp, x, hdw[c], hds[c], fmt, 1, I, D);   /* upload down  (I->D) */
    }
    int rows[64]; for (int c = 0; c < K; c++) rows[c] = 1;
    if (!coli_vk_expert_group(tg, tu, td, rows, K, yg, x)) { printf("expert_group failed\n"); return 1; }
    float *hid = malloc((size_t)I*4);
    for (int c = 0; c < K; c++) {
        float *xc = x + (size_t)c*D;
        for (int o = 0; o < I; o++) {
            double g = 0, u = 0; const uint8_t *gr = hgw[c]+(size_t)o*gu_rb, *ur = huw[c]+(size_t)o*gu_rb;
            for (int i = 0; i < D; i++) { g += xc[i]*deq(gr,fmt,i); u += xc[i]*deq(ur,fmt,i); }
            float gt = (float)(g*hgs[c][o]), ut = (float)(u*hus[c][o]);
            hid[o] = (gt/(1.0f+expf(-gt)))*ut;
        }
        for (int d = 0; d < D; d++) {
            double sdot = 0; const uint8_t *dr = hdw[c]+(size_t)d*d_rb;
            for (int o = 0; o < I; o++) sdot += hid[o]*deq(dr,fmt,o);
            yc[c*D+d] = (float)(sdot*hds[c][d]);
        }
    }
    double maxrel = 0;
    for (int i = 0; i < K*D; i++) { double e = fabs(yg[i]-yc[i]); if (fabs(yc[i])>1e-2) { double r = e/fabs(yc[i]); if (r>maxrel) maxrel = r; } }
    coli_vk_expert_group(tg, tu, td, rows, K, yg, x);   /* warm + leaves G.cmd recorded */
    int iters = 20; double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_expert_group(tg, tu, td, rows, K, yg, x);
    double ms = (now()-t0)*1000.0/iters/K;
    /* GPU-only: re-submit the already-recorded command buffer (skips per-call host setup:
     * descriptor updates + recording), isolating raw GPU throughput. */
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    double g0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    double gpums = (now()-g0)*1000.0/iters/K;
    printf("FULL VK expert_group fmt=%d %2d experts | maxrel=%.4g | per-call %.4f  GPU-only %.4f ms/expert (ROCm 0.179)\n",
           fmt, K, maxrel, ms, gpums);
    free(hid); free(x); free(yg); free(yc); free(tmp);
    for (int c = 0; c < K; c++) {
        coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); coli_vk_tensor_free(td[c]);
        free(hgw[c]); free(huw[c]); free(hdw[c]); free(hgs[c]); free(hus[c]); free(hds[c]);
    }
    return maxrel > 1e-3 ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) { printf("vk init failed\n"); return 1; }
    srand(1234);
    int bad = 0;
    bad |= run_case(1, 1, 6144, 1536, 50);   // int8 expert gate/up shape (S=1 decode)
    bad |= run_case(2, 1, 6144, 1536, 50);   // int4 expert
    bad |= run_case(1, 1, 1536, 6144, 50);   // down proj shape
    bad |= run_case(2, 8, 6144, 1536, 20);   // prefill/MTP batch
    bad |= run_case(1, 1, 512, 512, 100);    // small
    /* Batched (amortized) throughput on the int4 expert shapes — the real expert-tier pattern. */
    {
        int I = 6144, O = 2048;   /* our gate/up dims */
        float *x = malloc((size_t)I * 4); uint8_t *w = malloc((size_t)(I + 1) / 2 * O); float *sc = malloc((size_t)O * 4);
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        for (size_t i = 0; i < (size_t)(I + 1) / 2 * O; i++) w[i] = rand() & 0xff;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
        float *y = malloc((size_t)O * 4);
        ColiVkTensor *t = NULL; coli_vk_matmul(&t, y, x, w, sc, 2, 1, I, O);   /* bind */
        printf("BATCHED int4 S=1 6144->2048 (our gate/up): %.4f ms/matmul (N=64, one submit)\n",
               bench_batched(t, x, 2, 1, I, O, 64));
        coli_vk_tensor_free(t); free(x); free(w); free(sc); free(y);
        I = 2048; O = 6144;   /* our down dims */
        x = malloc((size_t)I * 4); w = malloc((size_t)(I + 1) / 2 * O); sc = malloc((size_t)O * 4);
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        for (size_t i = 0; i < (size_t)(I + 1) / 2 * O; i++) w[i] = rand() & 0xff;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
        y = malloc((size_t)O * 4);
        t = NULL; coli_vk_matmul(&t, y, x, w, sc, 2, 1, I, O);
        printf("BATCHED int4 S=1 2048->6144 (our down):    %.4f ms/matmul (N=64, one submit)\n",
               bench_batched(t, x, 2, 1, I, O, 64));
        coli_vk_tensor_free(t); free(x); free(w); free(sc); free(y);
    }
    /* FUSED gate+up: correctness + batched throughput (vs 2x separate gate/up). */
    {
        int D = 6144, I = 2048;
        bad |= run_gate_up(2, 1, D, I);
        size_t rb = (size_t)(D + 1) / 2;
        float *x = malloc((size_t)D*4); uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
        float *gs = malloc((size_t)I*4), *us = malloc((size_t)I*4), *h = malloc((size_t)I*4);
        for (int i = 0; i < D; i++) x[i] = (rand()%200-100)/100.0f;
        for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
        for (int o = 0; o < I; o++) { gs[o] = 0.01f; us[o] = 0.01f; }
        ColiVkTensor *tg = NULL, *tu = NULL; coli_vk_gate_up(&tg, &tu, h, x, gw, gs, uw, us, 2, 1, D, I);
        printf("BATCHED fused gate_up int4 6144->2048:     %.4f ms (N=64, SAME expert = L2-cached)\n",
               bench_gu_batched(tg, x, 2, 1, D, I, 64));
        coli_vk_tensor_free(tg); coli_vk_tensor_free(tu); free(x); free(gw); free(uw); free(gs); free(us); free(h);
    }
    /* FAIR: cycle 8 distinct experts (VRAM reads, not L2) — matches ROCm expert_group. */
    printf("FAIR fused gate_up int4 6144->2048 (8 distinct experts): %.4f ms/expert\n",
           bench_experts_fair(2, 6144, 2048, 8, 8));
    /* FULL expert_group: the real primitive. Sweep K to see if per-expert cost is fixed
     * per-call overhead (drops with K) or per-dispatch (constant). */
    bad |= run_expert_group(2, 6144, 2048, 1);
    bad |= run_expert_group(2, 6144, 2048, 8);
    bad |= run_expert_group(2, 6144, 2048, 32);
    printf(bad ? "FAIL\n" : "PASS\n");
    coli_vk_shutdown();
    return bad;
}
#endif
