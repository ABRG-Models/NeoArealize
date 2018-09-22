# The main process executable, for debugging and playing with the code.
# No graphics version of process model for code profiling
add_executable(process_nogfx process_nogfx.cpp HexGrid.cpp)
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  # Statically link the libittnotify library to allow pausing of data collection during init.
  target_link_libraries(process_nogfx -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY} /opt/intel/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a)
else()
  target_link_libraries(process_nogfx -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY})
endif()

add_executable(process_nogfx_doms process_nogfx_doms.cpp HexGrid.cpp)
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  # Statically link the libittnotify library to allow pausing of data collection during init.
  target_link_libraries(process_nogfx_doms -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY} /opt/intel/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a)
else()
  target_link_libraries(process_nogfx_doms -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY})
endif()

add_executable(testdom_pgram testdom_pgram.cpp HexGrid.cpp)
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  # Statically link the libittnotify library to allow pausing of data collection during init.
  target_link_libraries(testdom_pgram -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY} /opt/intel/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a)
else()
  target_link_libraries(testdom_pgram -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY})
endif()

add_executable(testdom_hex testdom_hex.cpp HexGrid.cpp)
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  # Statically link the libittnotify library to allow pausing of data collection during init.
  target_link_libraries(testdom_hex -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY} /opt/intel/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a)
else()
  target_link_libraries(testdom_hex -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY})
endif()

add_executable(testdom_rect testdom_rect.cpp HexGrid.cpp)
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  # Statically link the libittnotify library to allow pausing of data collection during init.
  target_link_libraries(testdom_rect -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY} /opt/intel/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a)
else()
  target_link_libraries(testdom_rect -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY})
endif()

add_executable(testdom_bound testdom_bound.cpp HexGrid.cpp)
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  # Statically link the libittnotify library to allow pausing of data collection during init.
  target_link_libraries(testdom_bound -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY} /opt/intel/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a)
else()
  target_link_libraries(testdom_bound -lmorphologica ${HDF5_C_${LIB_TYPE}_LIBRARY})
endif()
