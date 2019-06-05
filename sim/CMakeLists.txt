# The main process executable, for debugging and playing with the code.
add_executable(process process.cpp)
target_link_libraries(process ${MORPH_LDFLAGS} ${HDF5LIBLINK})

# Demonstrate Figure 4, and save data
add_executable(process_2d_karb_fig4 process_2d_karb_fig4.cpp)
target_link_libraries(process_2d_karb_fig4 ${MORPH_LDFLAGS} ${HDF5LIBLINK})

# Demonstrate Figure 5 (emx2 off), and save data
add_executable(process_2d_karb_fig5 process_2d_karb_fig5.cpp)
target_link_libraries(process_2d_karb_fig5 ${MORPH_LDFLAGS} ${HDF5LIBLINK})

# Demonstrate Figure 6 (pax6 off), and save data
add_executable(process_2d_karb_fig6 process_2d_karb_fig6.cpp)
target_link_libraries(process_2d_karb_fig6 ${MORPH_LDFLAGS} ${HDF5LIBLINK})

# Demonstrate Figure 8 (dual fgf8), and save data
add_executable(process_2d_karb_fig8 process_2d_karb_fig8.cpp)
target_link_libraries(process_2d_karb_fig8 ${MORPH_LDFLAGS} ${HDF5LIBLINK})

# Demonstrate gaussian guidance molecule system
add_executable(process_2d_karb_gaussguidance process_2d_karb_gaussguidance.cpp)
target_link_libraries(process_2d_karb_gaussguidance ${MORPH_LDFLAGS} ${HDF5LIBLINK})

# A template RD system. Not intended to be run really.
add_executable(template template.cpp)
target_link_libraries(template ${MORPH_LDFLAGS})

# Orientation preference test program
add_executable(orient orient.cpp)
target_link_libraries(orient ${MORPH_LDFLAGS})

# Ermentrout (2009) model
add_executable(ermentrout ermentrout.cpp)
target_link_libraries(ermentrout ${MORPH_LDFLAGS} ${HDF5LIBLINK})

# A first combined model
add_executable(embedded embedded.cpp)
target_link_libraries(embedded ${MORPH_LDFLAGS} ${HDF5LIBLINK})

# James model 0. RD Karbowski with variable numbers of guidance
# molecules and TC populations and runtime setting of parameters to
# examine behaviour.
add_executable(james0 james1.cpp)
target_compile_definitions(james0 PUBLIC FLOATTYPE=float COMPILE_PLOTTING)
target_link_libraries(james0 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Compile compute-only with floats
add_executable(james0c james1.cpp)
target_compile_definitions(james0c PUBLIC FLOATTYPE=float)
target_link_libraries(james0c ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Compile with doubles:
#add_executable(james0cd james1.cpp)
#target_compile_definitions(james0cd PUBLIC FLOATTYPE=double)
#target_link_libraries(james0cd ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# A parameter search
add_executable(ps_2N0M ps_2N0M.cpp)
target_compile_definitions(ps_2N0M PUBLIC FLOATTYPE=float)
target_link_libraries(ps_2N0M ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Schnakenberg RD system
add_executable(schnak schnak.cpp)
target_compile_definitions(schnak PUBLIC FLOATTYPE=float COMPILE_PLOTTING)
target_link_libraries(schnak ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Competition scheme 1
add_executable(james1 james1.cpp)
target_compile_definitions(james1 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP1)
target_link_libraries(james1 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Competition scheme 2
add_executable(james2 james1.cpp)
target_compile_definitions(james2 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP2)
target_link_libraries(james2 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Competition scheme 3
add_executable(james3 james1.cpp)
target_compile_definitions(james3 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP3)
target_link_libraries(james3 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Competition scheme 4 (comp1 + comp3)
add_executable(james4 james1.cpp)
target_compile_definitions(james4 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP4)
target_link_libraries(james4 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# A 5th scheme with better competition?
add_executable(james5 james1.cpp)
target_compile_definitions(james5 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP5)
target_link_libraries(james5 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# The multiplicative reduction of diffusion scheme
add_executable(james7 james1.cpp)
target_compile_definitions(james7 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP7)
target_link_libraries(james7 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Divisive normalization of a_i
add_executable(james8 james1.cpp)
target_compile_definitions(james8 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP8)
target_link_libraries(james8 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Divisive normalization of a_i along with competition scheme 1
add_executable(james9 james1.cpp)
target_compile_definitions(james9 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP9)
target_link_libraries(james9 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Sigmoidal normalization of a_i
add_executable(james10 james1.cpp)
target_compile_definitions(james10 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP10)
target_link_libraries(james10 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Sigmoidal normalization of a_i + competition + accumulation
add_executable(james11 james1.cpp)
target_compile_definitions(james11 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP11)
target_link_libraries(james11 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)

# Divisive normalization of a_i along with competition scheme 1 with extended epsilon
add_executable(james12 james1.cpp)
target_compile_definitions(james12 PUBLIC FLOATTYPE=float COMPILE_PLOTTING COMP12)
target_link_libraries(james12 ${MORPH_LDFLAGS} ${HDF5LIBLINK} jsoncpp)
