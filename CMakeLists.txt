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

# Unit Tests
add_executable(test_c_core tests/test_c_core.c)
target_link_libraries(test_c_core PRIVATE micro_actinf)

# Game AI Example
add_executable(game_ai_bot examples/game_ai_bot.c)
target_link_libraries(game_ai_bot PRIVATE micro_actinf)

if(NOT MSVC)
    target_link_libraries(test_c_core PRIVATE m)
    target_link_libraries(game_ai_bot PRIVATE m)
endif()
