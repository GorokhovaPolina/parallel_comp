#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

// чтение и запись изображений в формате pgm (P5, 8 бит)
bool readPGM(const std::string &filename, std::vector<uint8_t> &data, size_t &width, size_t &height)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    std::cerr << "Cannot open file: " << filename << std::endl;
    return false;
  }

  std::string magic;
  file >> magic;
  if (magic != "P5")
  {
    std::cerr << "onle P5 PGM format" << std::endl;
    return false;
  }

  char c;
  file.get(c);
  while (c == '#')
  {
    std::string comment;
    std::getline(file, comment);
    file.get(c);
  }
  file.unget();

  file >> width >> height;
  int maxVal;
  file >> maxVal;
  file.get();

  if (maxVal != 255)
  {
    std::cerr << "only 8-bit PGM" << std::endl;
    return false;
  }

  data.resize(width * height);
  file.read(reinterpret_cast<char *>(data.data()), width * height);
  return true;
}

bool writePGM(const std::string &filename, const std::vector<uint8_t> &data, size_t width, size_t height)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    std::cerr << "Cannot write file: " << filename << std::endl;
    return false;
  }

  file << "P5\n"
       << width << " " << height << "\n255\n";
  file.write(reinterpret_cast<const char *>(data.data()), width * height);
  return true;
}

// скалярная реализация размытия 3х3
void blur_3x3_scalar(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, size_t width, size_t height)
{
  dst.resize(width * height);
  // обработка всех пикселнй кроме крайних
  for (size_t y = 1; y < height - 1; ++y)
  {
    for (size_t x = 1; x < width - 1; ++x)
    {
      uint32_t sum = 0;
      for (int dy = -1; dy <= 1; ++dy)
      {
        for (int dx = -1; dx <= 1; ++dx)
        {
          sum += src[(y + dy) * width + (x + dx)];
        }
      }
      dst[y * width + x] = static_cast<uint8_t>((sum + 4) / 9);
    }
  }
  // копирование границ без изменений
  for (size_t y = 0; y < height; ++y)
  {
    dst[y * width] = src[y * width];
    dst[y * width + width - 1] = src[y * width + width - 1];
  }
  for (size_t x = 0; x < width; ++x)
  {
    dst[x] = src[x];
    dst[(height - 1) * width + x] = src[(height - 1) * width + x];
  }
}

// основная SIMD-функция размытия
void blur_3x3_avx2(
    const std::vector<uint8_t> &src,
    std::vector<uint8_t> &dst,
    size_t width,
    size_t height)
{
  dst.resize(width * height);

  // копирование границ
  for (size_t x = 0; x < width; ++x)
  {
    dst[x] = src[x];
    dst[(height - 1) * width + x] = src[(height - 1) * width + x];
  }

  for (size_t y = 1; y < height - 1; ++y)
  {
    dst[y * width] = src[y * width];
    dst[y * width + width - 1] = src[y * width + width - 1];
  }

  for (size_t y = 1; y < height - 1; ++y)
  {
    const uint8_t *r0 = src.data() + (y - 1) * width;
    const uint8_t *r1 = src.data() + y * width;
    const uint8_t *r2 = src.data() + (y + 1) * width;

    size_t x = 1;

    auto process_16 = [&](size_t px, uint8_t *out)
    {
      __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r0 + px - 1));
      __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r0 + px));
      __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r0 + px + 1));

      __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r1 + px - 1));
      __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r1 + px));
      __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r1 + px + 1));

      __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r2 + px - 1));
      __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r2 + px));
      __m128i i = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r2 + px + 1));

      __m256i a16 = _mm256_cvtepu8_epi16(a);
      __m256i b16 = _mm256_cvtepu8_epi16(b);
      __m256i c16 = _mm256_cvtepu8_epi16(c);
      __m256i d16 = _mm256_cvtepu8_epi16(d);
      __m256i e16 = _mm256_cvtepu8_epi16(e);
      __m256i f16 = _mm256_cvtepu8_epi16(f);
      __m256i g16 = _mm256_cvtepu8_epi16(g);
      __m256i h16 = _mm256_cvtepu8_epi16(h);
      __m256i i16 = _mm256_cvtepu8_epi16(i);

      __m256i sum = _mm256_add_epi16(
          _mm256_add_epi16(_mm256_add_epi16(a16, b16), _mm256_add_epi16(c16, d16)),
          _mm256_add_epi16(_mm256_add_epi16(e16, f16), _mm256_add_epi16(_mm256_add_epi16(g16, h16), i16)));

      alignas(32) uint16_t tmp[16];
      _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), sum);

      for (int k = 0; k < 16; ++k)
        out[k] = static_cast<uint8_t>((tmp[k] + 4) / 9);
    };

    for (; x + 31 < width - 1; x += 32)
    {
      process_16(x, dst.data() + y * width + x);
      process_16(x + 16, dst.data() + y * width + x + 16);
    }

    for (; x < width - 1; ++x)
    {
      uint32_t sum = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          sum += src[(y + dy) * width + (x + dx)];

      dst[y * width + x] = static_cast<uint8_t>((sum + 4) / 9);
    }
  }
}

// тестирование производительности
double measure_scalar(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, size_t width, size_t height, int iterations)
{
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i)
  {
    blur_3x3_scalar(src, dst, width, height);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / iterations;
}

double measure_avx(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, size_t width, size_t height, int iterations)
{
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i)
  {
    blur_3x3_avx2(src, dst, width, height);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / iterations;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " <input.pgm>" << std::endl;
    return 1;
  }

  std::string inputFilename = argv[1];
  std::vector<uint8_t> src;
  size_t width, height;
  if (!readPGM(inputFilename, src, width, height))
  {
    return 1;
  }

  std::cout << "Image loaded: " << width << "x" << height << std::endl;

  const int iterations = 50; // количество итераций для усреднения

  std::vector<uint8_t> dst_scalar, dst_avx;

  // измерение скалярной версии
  double time_scalar = measure_scalar(src, dst_scalar, width, height, iterations);
  std::cout << "Scalar average time: " << time_scalar << " ms" << std::endl;

  // измерение avx версии
  double time_avx = measure_avx(src, dst_avx, width, height, iterations);
  std::cout << "AVX average time:    " << time_avx << " ms" << std::endl;

  double speedup = time_scalar / time_avx;
  std::cout << "Speedup: " << speedup << "x" << std::endl;

  // сохранение результатов
  writePGM("output_scalar.pgm", dst_scalar, width, height);
  writePGM("output_simd.pgm", dst_avx, width, height);

  // запись лога
  std::ofstream log("benchmark.txt");
  log << "Method,Avg Time (ms),Speedup\n";
  log << "Scalar," << time_scalar << ",1.00x\n";
  log << "AVX," << time_avx << "," << speedup << "x\n";
  log.close();

  bool match = true;
  int max_diff = 0;
  for (size_t y = 1; y < height - 1; ++y)
  {
    for (size_t x = 1; x < width - 1; ++x)
    {
      size_t idx = y * width + x;
      int diff = abs(static_cast<int>(dst_scalar[idx]) - static_cast<int>(dst_avx[idx]));
      if (diff > max_diff)
        max_diff = diff;
      if (diff > 1)
      {
        std::cerr << "mismatch at (" << x << "," << y << "): scalar="
                  << (int)dst_scalar[idx] << ", avx=" << (int)dst_avx[idx] << std::endl;
        match = false;
      }
    }
  }

  if (match)
  {
    std::cout << "results match within tolerance (max difference = " << max_diff << ")" << std::endl;
  }
  else
  {
    std::cout << "results differ significantly in interior region" << std::endl;
  }

  return 0;
}