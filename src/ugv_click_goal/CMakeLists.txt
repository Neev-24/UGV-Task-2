cmake_minimum_required(VERSION 3.8)
project(ugv_click_goal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

link_directories(${gazebo_dev_LIBRARY_DIRS})
add_library(ugv_click_goal_gui SHARED src/click_goal_gui.cpp)
target_include_directories(ugv_click_goal_gui SYSTEM PRIVATE ${gazebo_dev_INCLUDE_DIRS})
ament_target_dependencies(ugv_click_goal_gui gazebo_dev gazebo_ros geometry_msgs rclcpp)
target_link_libraries(ugv_click_goal_gui gazebo_gui ${gazebo_dev_LIBRARIES} Qt5::Widgets)

install(TARGETS ugv_click_goal_gui LIBRARY DESTINATION lib)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
