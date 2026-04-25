########################################################################################################################
### MAGIC SQUARE TESTS
########################################################################################################################

find_package(OpenMP REQUIRED)

# Helper function to create a test target with common configuration
function(add_square_test TEST_SUFFIX TEST_SOURCE)
    set(TARGET_NAME ${MAGIC_SQUARE_PROJECT_NAME}${TEST_SUFFIX})

    add_executable(${TARGET_NAME}
            ${TEST_SOURCE}
            ${MAGIC_SQUARE_SRC_DIR}/program_options.cpp
            ${MAGIC_SQUARE_SRC_DIR}/magic_square.cpp)

    target_include_directories(${TARGET_NAME} PUBLIC ${MAGIC_SQUARE_INC_DIR})

    if (OpenMP_CXX_FOUND)
        target_include_directories(${TARGET_NAME} PUBLIC OpenMP::OpenMP_CXX_INCLUDE_DIRS)
        target_link_libraries(${TARGET_NAME} PUBLIC OpenMP::OpenMP_CXX)
    endif ()
endfunction()

# Correctness tests (one per square size)
add_square_test(TestsThree  ${MAGIC_SQUARE_TESTS_DIR}/square_three_test.cpp)
add_square_test(TestsFour   ${MAGIC_SQUARE_TESTS_DIR}/square_four_test.cpp)
add_square_test(TestsFive   ${MAGIC_SQUARE_TESTS_DIR}/square_five_test.cpp)
add_square_test(TestsSix    ${MAGIC_SQUARE_TESTS_DIR}/square_six_test.cpp)
add_square_test(TestsSeven  ${MAGIC_SQUARE_TESTS_DIR}/square_seven_test.cpp)
add_square_test(TestsEight  ${MAGIC_SQUARE_TESTS_DIR}/square_eight_test.cpp)
add_square_test(TestsNine   ${MAGIC_SQUARE_TESTS_DIR}/square_nine_test.cpp)

# Visual tests (verbose output, shows GA evolving for each size)
add_square_test(TestsThreeVisual  ${MAGIC_SQUARE_TESTS_DIR}/square_three_visual_test.cpp)
add_square_test(TestsFourVisual   ${MAGIC_SQUARE_TESTS_DIR}/square_four_visual_test.cpp)
add_square_test(TestsFiveVisual   ${MAGIC_SQUARE_TESTS_DIR}/square_five_visual_test.cpp)
add_square_test(TestsSixVisual    ${MAGIC_SQUARE_TESTS_DIR}/square_six_visual_test.cpp)
add_square_test(TestsSevenVisual  ${MAGIC_SQUARE_TESTS_DIR}/square_seven_visual_test.cpp)
add_square_test(TestsEightVisual  ${MAGIC_SQUARE_TESTS_DIR}/square_eight_visual_test.cpp)
add_square_test(TestsNineVisual   ${MAGIC_SQUARE_TESTS_DIR}/square_nine_visual_test.cpp)

# Register CTest tests - correctness
add_test(NAME square_three_test COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsThree>)
add_test(NAME square_four_test  COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsFour>)
add_test(NAME square_five_test  COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsFive>)
add_test(NAME square_six_test   COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsSix>)
add_test(NAME square_seven_test COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsSeven>)
add_test(NAME square_eight_test COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsEight>)
add_test(NAME square_nine_test  COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsNine>)

# Register CTest tests - visual
add_test(NAME square_three_visual_test COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsThreeVisual>)
add_test(NAME square_four_visual_test  COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsFourVisual>)
add_test(NAME square_five_visual_test  COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsFiveVisual>)
add_test(NAME square_six_visual_test   COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsSixVisual>)
add_test(NAME square_seven_visual_test COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsSevenVisual>)
add_test(NAME square_eight_visual_test COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsEightVisual>)
add_test(NAME square_nine_visual_test  COMMAND $<TARGET_FILE:${MAGIC_SQUARE_PROJECT_NAME}TestsNineVisual>)

# Test timeouts (seconds) - correctness
set_tests_properties(square_three_test PROPERTIES TIMEOUT 60)
set_tests_properties(square_four_test  PROPERTIES TIMEOUT 120)
set_tests_properties(square_five_test  PROPERTIES TIMEOUT 300)
set_tests_properties(square_six_test   PROPERTIES TIMEOUT 600)
set_tests_properties(square_seven_test PROPERTIES TIMEOUT 600)
set_tests_properties(square_eight_test PROPERTIES TIMEOUT 600)
set_tests_properties(square_nine_test  PROPERTIES TIMEOUT 600)

# Test timeouts (seconds) - visual
set_tests_properties(square_three_visual_test PROPERTIES TIMEOUT 60)
set_tests_properties(square_four_visual_test  PROPERTIES TIMEOUT 120)
set_tests_properties(square_five_visual_test  PROPERTIES TIMEOUT 300)
set_tests_properties(square_six_visual_test   PROPERTIES TIMEOUT 600)
set_tests_properties(square_seven_visual_test PROPERTIES TIMEOUT 600)
set_tests_properties(square_eight_visual_test PROPERTIES TIMEOUT 600)
set_tests_properties(square_nine_visual_test  PROPERTIES TIMEOUT 600)