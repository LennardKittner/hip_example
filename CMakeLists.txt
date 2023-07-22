cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21
cmake_policy(VERSION 3.21.3...3.27)
project(hip_example LANGUAGES HIP)
add_executable(hip_example main.hip)