cmake_minimum_required(VERSION 3.22)
project(ClayMaterial LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# SDL2 будет клонирован workflow-ом в ./SDL
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SDL/CMakeLists.txt")
    message(FATAL_ERROR "SDL2 не найден в ${CMAKE_CURRENT_SOURCE_DIR}/SDL. Workflow должен был его склонировать.")
endif()

set(SDL_SHARED ON  CACHE BOOL "" FORCE)
set(SDL_STATIC OFF CACHE BOOL "" FORCE)
set(SDL_TEST   OFF CACHE BOOL "" FORCE)
add_subdirectory(SDL EXCLUDE_FROM_ALL)

add_library(main SHARED
    src/main.cpp
    src/imgui/imgui.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_demo.cpp
    src/imgui/backends/imgui_impl_sdl2.cpp
    src/imgui/backends/imgui_impl_sdlrenderer2.cpp
)

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/SDL/include
)

target_link_libraries(main SDL2)
