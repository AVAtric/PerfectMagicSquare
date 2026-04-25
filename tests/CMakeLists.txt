########################################################################################################################
### MAGIC SQUARE TESTS
########################################################################################################################

find_package(OpenMP REQUIRED)

# Helper: build a single test binary and register it with CTest.
function(add_square_test TEST_SUFFIX TEST_SOURCE TEST_NAME TEST_TIMEOUT)
    set(TARGET_NAME ${MAGIC_SQUARE_PROJECT_NAME}${TEST_SUFFIX})

    add_executable(${TARGET_NAME}
            ${TEST_SOURCE}
            ${MAGIC_SQUARE_SRC_DIR}/program_options.cpp
            ${MAGIC_SQUARE_SRC_DIR}/magic_square.cpp)

    target_include_directories(${TARGET_NAME} PUBLIC
            ${MAGIC_SQUARE_INC_DIR}
            ${MAGIC_SQUARE_TESTS_DIR})

    if (OpenMP_CXX_FOUND)
        target_include_directories(${TARGET_NAME} PUBLIC OpenMP::OpenMP_CXX_INCLUDE_DIRS)
        target_link_libraries(${TARGET_NAME} PUBLIC OpenMP::OpenMP_CXX)
    endif ()

    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TARGET_NAME}>)
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT ${TEST_TIMEOUT})
endfunction()

# Per-size test configuration: <Suffix>|<name>|<correctness-timeout>|<visual-timeout>
set(SQUARE_TESTS
        "Three|three|60|60"
        "Four|four|120|120"
        "Five|five|300|300"
        "Six|six|600|600"
        "Seven|seven|600|600"
        "Eight|eight|600|600"
        "Nine|nine|600|600"
)

foreach (entry IN LISTS SQUARE_TESTS)
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 SUFFIX)
    list(GET parts 1 NAME)
    list(GET parts 2 CORRECTNESS_TIMEOUT)
    list(GET parts 3 VISUAL_TIMEOUT)

    add_square_test(
            "Tests${SUFFIX}"
            "${MAGIC_SQUARE_TESTS_DIR}/square_${NAME}_test.cpp"
            "square_${NAME}_test"
            ${CORRECTNESS_TIMEOUT})

    add_square_test(
            "Tests${SUFFIX}Visual"
            "${MAGIC_SQUARE_TESTS_DIR}/square_${NAME}_visual_test.cpp"
            "square_${NAME}_visual_test"
            ${VISUAL_TIMEOUT})
endforeach ()
