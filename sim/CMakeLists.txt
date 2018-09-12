# The main process executable, for debugging and playing with the code.
# No graphics version of process model for code profiling
add_executable(process_nogfx process_nogfx.cpp tools.cpp BezCurve.cpp HdfData.cpp HexGrid.cpp ReadCurves.cpp)
target_link_libraries(process_nogfx ${HDF5_C_${LIB_TYPE}_LIBRARY})
