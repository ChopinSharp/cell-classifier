# CMakeList.txt : CMake project for cell_classifier_gui, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

set(CMAKE_PREFIX_PATH "C:\\Applications\\Qt-crap\\5.12.2\\msvc2017_64;C:\\Users\\mwb\\Documents\\VS Projects\\Library\\libtorch;C:\\Users\\mwb\\Documents\\VS Projects\\Library\\opencv\\build")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)

# Find the QtWidgets library
find_package(Qt5 CONFIG REQUIRED Widgets Charts)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Populate a CMake variable with the sources
set(cell_classifier_gui_SRCS
    mainwindow.ui
    mainwindow.cpp
    main.cpp
	CellClassifier.cpp
)
# Tell CMake to create the helloworld executable
#add_executable(cell_classifier_gui WIN32 ${cell_classifier_gui_SRCS})
add_executable(cell_classifier_gui ${cell_classifier_gui_SRCS})
# Use the Widgets module from Qt 5
target_link_libraries(cell_classifier_gui Qt5::Widgets Qt5::Charts ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET cell_classifier_gui PROPERTY CXX_STANDARD 17)

