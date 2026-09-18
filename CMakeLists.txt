cmake_minimum_required(VERSION 3.15)
project(micro_actinf C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

include_directories(include)

# Static Library
add_library(micro_actinf STATIC src/micro_actinf.c)
target_compile_definitions(micro_actinf PUBLIC MICRO_ACTINF_STATIC)

# Shared Library
add_library(micro_actinf_shared SHARED src/micro_actinf.c)
set_target_properties(micro_actinf_shared PROPERTIES OUTPUT_NAME "micro_actinf")
target_compile_definitions(micro_actinf_shared PRIVATE MICRO_ACTINF_BUILD_DLL)

# Unit Tests
add_executable(test_c_core tests/test_c_core.c)
target_link_libraries(test_c_core PRIVATE micro_actinf)

add_executable(test_fast_log_accuracy tests/test_fast_log_accuracy.c)

# Game AI Example
add_executable(game_ai_bot examples/game_ai_bot.c)
target_link_libraries(game_ai_bot PRIVATE micro_actinf)

# Comparative Benchmark Example (Normal vs Professional)
add_executable(comparative_test examples/comparative_test.c)
target_link_libraries(comparative_test PRIVATE micro_actinf)

if(NOT MSVC)
    target_link_libraries(test_c_core PRIVATE m)
    target_link_libraries(test_fast_log_accuracy PRIVATE m)
    target_link_libraries(game_ai_bot PRIVATE m)
    target_link_libraries(comparative_test PRIVATE m)
    target_link_libraries(micro_actinf_shared PRIVATE m)
endif()

