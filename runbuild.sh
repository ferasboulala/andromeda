set -x
set -e

cmake -B build
cmake --build build -j$(sysctl -n hw.ncpu)
