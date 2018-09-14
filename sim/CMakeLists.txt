# The main process executable, for debugging and playing with the code.
# No graphics version of process model for code profiling
add_executable(process_nogfx process_nogfx.cpp tools.cpp BezCurve.cpp HdfData.cpp HexGrid.cpp ReadCurves.cpp)

if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  # Statically link the libittnotify library to allow pausing of data collection during init.
  target_link_libraries(process_nogfx ${HDF5_C_${LIB_TYPE}_LIBRARY} /opt/intel/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a)
else()
  target_link_libraries(process_nogfx ${HDF5_C_${LIB_TYPE}_LIBRARY})
endif()
