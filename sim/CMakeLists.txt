# The main process executable, for debugging and playing with the code.
add_executable(process process.cpp)
target_link_libraries(process ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# Demonstrate Figure 4, and save data
add_executable(process_2d_karb_fig4 process_2d_karb_fig4.cpp)
target_link_libraries(process_2d_karb_fig4 ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# Demonstrate Figure 5 (emx2 off), and save data
add_executable(process_2d_karb_fig5 process_2d_karb_fig5.cpp)
target_link_libraries(process_2d_karb_fig5 ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# Demonstrate Figure 6 (pax6 off), and save data
add_executable(process_2d_karb_fig6 process_2d_karb_fig6.cpp)
target_link_libraries(process_2d_karb_fig6 ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# Demonstrate Figure 8 (dual fgf8), and save data
add_executable(process_2d_karb_fig8 process_2d_karb_fig8.cpp)
target_link_libraries(process_2d_karb_fig8 ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# Demonstrate gaussian guidance molecule system
add_executable(process_2d_karb_gaussguidance process_2d_karb_gaussguidance.cpp)
target_link_libraries(process_2d_karb_gaussguidance ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# A template RD system. Not intended to be run really.
add_executable(template template.cpp)
target_link_libraries(template ${MORPH_LDFLAGS})

# Orientation preference test program
add_executable(orient orient.cpp)
target_link_libraries(orient ${MORPH_LDFLAGS})

# Ermentrout (2009) model
add_executable(ermentrout ermentrout.cpp)
target_link_libraries(ermentrout ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# A first combined model
add_executable(embedded embedded.cpp)
target_link_libraries(embedded ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# James model 1. RD Karbowski with variable numbers of guidance
# molecules and TC populations and runtime setting of parameters to
# examine behaviour.
add_executable(james1 james1.cpp)
target_compile_definitions(james1 PUBLIC FLOATTYPE=float COMPILE_PLOTTING)
target_link_libraries(james1 ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY} jsoncpp)

# Compile compute-only with floats
add_executable(james1c james1.cpp)
target_compile_definitions(james1c PUBLIC FLOATTYPE=float)
target_link_libraries(james1c ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY} jsoncpp)

# Compile compute-only with floats
add_executable(scanspace scanspace.cpp)
target_compile_definitions(scanspace PUBLIC FLOATTYPE=float)
target_link_libraries(scanspace ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY} jsoncpp)

# Compile with doubles:
add_executable(james1cd james1.cpp)
target_compile_definitions(james1cd PUBLIC FLOATTYPE=double)
target_link_libraries(james1cd ${MORPH_LDFLAGS} ${HDF5_C_${LIB_TYPE}_LIBRARY} jsoncpp)
