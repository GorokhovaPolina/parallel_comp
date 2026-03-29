#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cpuid.h> 

// макросы для удобного чтения битов и полей
#define BIT(n) (1U << (n))
#define FIELD(val, low, high) (((val) >> (low)) & ((1U << ((high)-(low)+1))-1))

// структура для хранения информации о кэше
typedef struct {
    unsigned int type;         // 1=data, 2=instr, 3=unified
    unsigned int level;        // 1,2,3,...
    unsigned int threads_share;// сколько логических процессоров делят кэш
    unsigned int line_size;    // байт
    unsigned int partitions;   // число линий с одним тегом
    unsigned int associativity; // степень ассоциативности
    unsigned int sets;         // число наборов
    unsigned int inclusive;    // 1=инклюзивный, 0=эксклюзивный
    unsigned int size;         // вычисленный размер в КБ
} cache_info_t;

// Функция вывода флагов из регистра
void print_flags(const char *title, unsigned int reg, const char *flags[], int n) {
    printf("%s:\n", title);
    for (int i = 0; i < n; i++) {
        if (reg & BIT(i) && flags[i] != NULL)
            printf("  %s\n", flags[i]);
    }
}

int main() {
    unsigned int eax, ebx, ecx, edx;
    unsigned int max_basic, max_ext;

    // 1 - получаем максимальный базовый leaf и строку производителя
    __cpuid_count(0, 0, eax, ebx, ecx, edx);
    max_basic = eax;
    char vendor[13] = {0};
    memcpy(vendor,      &ebx, 4);
    memcpy(vendor + 4,  &edx, 4);
    memcpy(vendor + 8,  &ecx, 4);
    printf("Производитель: %s\n", vendor);
    printf("Макс. базовый leaf: 0x%x\n\n", max_basic);

    // 2 - leaf 1 - версия процессора и основные флаги
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    printf("--- Информация о процессоре (leaf 1) ---\n");
    unsigned int stepping = FIELD(eax, 0, 3);
    unsigned int model    = FIELD(eax, 4, 7);
    unsigned int family   = FIELD(eax, 8, 11);
    unsigned int ext_model = FIELD(eax, 16, 19);
    unsigned int ext_family = FIELD(eax, 20, 27);
    unsigned int proc_type = FIELD(eax, 12, 13);
    printf("Stepping: %u, Model: %u, Family: %u, Extended model: %u, Extended family: %u, Processor type: %u\n",
           stepping, model, family, ext_model, ext_family, proc_type);
    unsigned int max_logical = FIELD(ebx, 16, 23);
    unsigned int apic_id = ebx & 0xFF;
    printf("Max logical processors per core: %u, Local APIC ID: %u\n", max_logical, apic_id);

    // флаги из EDX (старые)
    const char *edx_flags[] = {
        [0]  = "FPU", [1]  = "VME", [2]  = "DE", [3]  = "PSE",
        [4]  = "TSC", [5]  = "MSR", [6]  = "PAE", [7]  = "MCE",
        [8]  = "CX8", [9]  = "APIC", [10] = NULL, [11] = "SEP",
        [12] = "MTRR", [13] = "PGE", [14] = "MCA", [15] = "CMOV",
        [16] = "PAT", [17] = "PSE36", [18] = "PSN", [19] = "CLFSH",
        [20] = NULL, [21] = "DS", [22] = "ACPI", [23] = "MMX",
        [24] = "FXSR", [25] = "SSE", [26] = "SSE2", [27] = "SS",
        [28] = "HTT", [29] = "TM", [30] = "IA64", [31] = "PBE"
    };
    print_flags("Поддерживаемые технологии (EDX)", edx, edx_flags, 32);

    // флаги из ECX (более новые)
    const char *ecx_flags[] = {
        [0]  = "SSE3", [1]  = "PCLMULQDQ", [2]  = "DTES64", [3]  = "MONITOR",
        [4]  = "DS-CPL", [5]  = "VMX", [6]  = "SMX", [7]  = "EST",
        [8]  = "TM2", [9]  = "SSSE3", [10] = "CNXT-ID", [11] = "SDBG",
        [12] = "FMA3", [13] = "CX16", [14] = "xTPR", [15] = "PDCM",
        [16] = NULL, [17] = "PCID", [18] = "DCA", [19] = "SSE4.1",
        [20] = "SSE4.2", [21] = "x2APIC", [22] = "MOVBE", [23] = "POPCNT",
        [24] = "TSC-Deadline", [25] = "AESNI", [26] = "XSAVE", [27] = "OSXSAVE",
        [28] = "AVX", [29] = "F16C", [30] = "RDRAND", [31] = "HYPERVISOR"
    };
    print_flags("Поддерживаемые технологии (ECX)", ecx, ecx_flags, 32);
    printf("\n");

    // 3 - информация о кэше (leaf 4)
    printf("--- Информация о кэш-памяти (leaf 4) ---\n");
    for (int sub = 0; ; sub++) {
        __cpuid_count(4, sub, eax, ebx, ecx, edx);
        unsigned int cache_type = eax & 0x1F;
        if (cache_type == 0) break;   // конец списка
        cache_info_t cache = {0};
        cache.type = cache_type;
        cache.level = (eax >> 5) & 0x7;
        cache.threads_share = ((eax >> 14) & 0xFFF) + 1;
        cache.line_size = (ebx & 0xFFF) + 1;
        cache.partitions = ((ebx >> 12) & 0x3FF) + 1;
        cache.associativity = ((ebx >> 22) & 0x3FF) + 1;
        cache.sets = ecx + 1;
        cache.inclusive = edx & 1;
        cache.size = cache.line_size * cache.partitions * cache.associativity * cache.sets / 1024;
        printf("Кэш %d: %s, уровень %u, разделяется %u потоков, размер %u КБ, %s\n",
               sub,
               cache.type == 1 ? "Data" : (cache.type == 2 ? "Instruction" : "Unified"),
               cache.level, cache.threads_share, cache.size,
               cache.inclusive ? "инклюзивный" : "эксклюзивный");
        printf("  line size = %u, associativity = %u, sets = %u\n",
               cache.line_size, cache.associativity, cache.sets);
    }
    printf("\n");

    // 4 - leaf 7, ECX = 0 - расширенные возможности
    if (max_basic >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        unsigned int max_subleaf = eax;
        printf("--- Расширенные возможности (leaf 7, max subleaf = %u) ---\n", max_subleaf);
        const char *ebx7_flags[] = {
            [0]  = "FSGSBASE", [1]  = "IA32_TSC_ADJUST", [2]  = "SGX", [3]  = "BMI1",
            [4]  = "HLE", [5]  = "AVX2", [6]  = "FDP_EXCPTN_ONLY", [7]  = "SMEP",
            [8]  = "BMI2", [9]  = "ERMS", [10] = "INVPCID", [11] = "RTM",
            [12] = "PQM", [13] = "ZERO_FCS_FDS", [14] = "MPX", [15] = "PQE",
            [16] = "AVX512F", [17] = "AVX512DQ", [18] = "RDSEED", [19] = "ADX",
            [20] = "SMAP", [21] = "AVX512IFMA", [22] = "PCOMMIT", [23] = "CLFLUSHOPT",
            [24] = "CLWB", [25] = "INTEL_PT", [26] = "AVX512PF", [27] = "AVX512ER",
            [28] = "AVX512CD", [29] = "SHA", [30] = "AVX512BW", [31] = "AVX512VL"
        };
        print_flags("Флаги EBX (leaf 7, ECX=0)", ebx, ebx7_flags, 32);
    }
    printf("\n");

    // 5 - leaf 0x16
    if (max_basic >= 0x16) {
        __cpuid_count(0x16, 0, eax, ebx, ecx, edx);
        printf("--- Тактовые частоты (leaf 0x16) ---\n");
        printf("Базовая частота: %u МГц\n", eax & 0xFFFF);
        printf("Макс. частота (boost): %u МГц\n", ebx & 0xFFFF);
        printf("Частота шины: %u МГц\n", ecx & 0xFFFF);
        printf("\n");
    }

    // 6 - расширенные функции (начиная с 0x80000000)
    __cpuid_count(0x80000000, 0, eax, ebx, ecx, edx);
    max_ext = eax;
    printf("--- Расширенные функции (макс. leaf = 0x%x) ---\n", max_ext);

    if (max_ext >= 0x80000001) {
        __cpuid_count(0x80000001, 0, eax, ebx, ecx, edx);
        // AMD-специфичные флаги
        const char *ecx_ext_flags[] = {
            [5] = "LZCNT", [6] = "SSE4a", [7] = "MISALIGNSSE", [8] = "PREFETCHW",
            [11] = "XOP", [12] = "FMA4", [16] = "TBM"
        };
        print_flags("Флаги ECX (leaf 0x80000001)", ecx, ecx_ext_flags, 32);
        const char *edx_ext_flags[] = {
            [31] = "3DNow!", [30] = "3DNow! Ext"
        };
        print_flags("Флаги EDX (leaf 0x80000001)", edx, edx_ext_flags, 32);
    }

    // 7 - строка с названием процессора (brand string) через листья 0x80000002-0x80000004
    if (max_ext >= 0x80000004) {
        char brand[49] = {0};
        __cpuid_count(0x80000002, 0, eax, ebx, ecx, edx);
        memcpy(brand,      &eax, 4);
        memcpy(brand + 4,  &ebx, 4);
        memcpy(brand + 8,  &ecx, 4);
        memcpy(brand + 12, &edx, 4);
        __cpuid_count(0x80000003, 0, eax, ebx, ecx, edx);
        memcpy(brand + 16, &eax, 4);
        memcpy(brand + 20, &ebx, 4);
        memcpy(brand + 24, &ecx, 4);
        memcpy(brand + 28, &edx, 4);
        __cpuid_count(0x80000004, 0, eax, ebx, ecx, edx);
        memcpy(brand + 32, &eax, 4);
        memcpy(brand + 36, &ebx, 4);
        memcpy(brand + 40, &ecx, 4);
        memcpy(brand + 44, &edx, 4);
        printf("Название процессора: %s\n", brand);
    }

    return 0;
}