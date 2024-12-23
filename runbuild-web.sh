set -x
set -e

source ../emsdk/emsdk_env.sh

emcmake cmake -B build-web -DUSE_EMSCRIPTEN=1 -DPLATFORM=Web
cmake --build build-web -j$(sysctl -n hw.ncpu) --verbose