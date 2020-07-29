################################################################################
#                                                                              #
# These are macros useful for performing certain common CMake tasks that don't #
# really fit into the other Macro files.  See dox/MacroDocumentation.md for    #
# syntax and usage examples.                                                   #
#                                                                              #
################################################################################


function(prefix_paths __prefix __list)
    foreach(__file ${${__list}})
        list(APPEND __temp_list ${__prefix}/${__file})
    endforeach()
    set(${__list} ${__temp_list} PARENT_SCOPE)
endfunction()

function(make_full_paths __list)
    set(__prefix "${CMAKE_CURRENT_LIST_DIR}")
    prefix_paths(${__prefix} ${__list})
    set(${__list} ${${__list}} PARENT_SCOPE)
endfunction()

function(clean_flags __list __output_list)
    set(__found_flags)
    set(__exception_flags)
    set(__good_flags)
    foreach(__flag ${${__list}})
        list(FIND __found_flags ${__flag} __index)
        if(__index EQUAL -1) #Have we wrote this flag yet?
            list(APPEND __good_flags "${__flag}")
            list(FIND __exception_flags ${__flag} __index)
            if(__index EQUAL -1) #Add to ban-list if we're not allowed multiples
                list(APPEND __found_flags ${__flag})
            endif()
        endif()
    endforeach()
    string(REPLACE ";" " " __clean_flags "${__good_flags}")
    set(${__output_list} "${__clean_flags}" PARENT_SCOPE)
endfunction()


function(is_valid __variable __out)
set(${__out} FALSE PARENT_SCOPE)
if(DEFINED ${__variable} AND (NOT "${${__variable}}" STREQUAL ""))
    set(${__out} TRUE PARENT_SCOPE)
endif()
endfunction()

function(is_valid_and_true __variable __out)
    is_valid(${__variable} __temp)
    set(${__out} FALSE PARENT_SCOPE)
    if(__temp AND ${__variable})
        set(${__out} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(print_banner msg)
    foreach(_dummy_idx RANGE 79)
        set(_lots_o_stars "${_lots_o_stars}*")
    endforeach()
    string(LENGTH ${msg} _msg_length)
    math(EXPR _white_space "78-${_msg_length}")
    math(EXPR _extra_space "${_white_space}%2")
    math(EXPR _lmargin_length "(${_white_space}-${_extra_space})/2 - 1")
    math(EXPR _rmargin_length "(${_white_space}-${_extra_space})/2 +
    ${_extra_space} - 1")
    foreach(_margin _rmargin _lmargin)
        foreach(_dummy_idx RANGE ${${_margin}_length})
            set(${_margin} " ${${_margin}}")
        endforeach()
    endforeach()

    message("${_lots_o_stars}")
    message("*${_lmargin}${msg}${_rmargin}*")
    message("${_lots_o_stars}")
endfunction()


function(string_concat list_o_stuff prefix spacer _result)
    foreach(_item ${${list_o_stuff}})
        is_valid(${_result} result_set)
        if(result_set) #Only put space if we're actually appending
            set(${_result} "${${_result}}${spacer}${prefix}${_item}")
        else()
            set(${_result} "${prefix}${_item}")
        endif()
    endforeach()
    set(${_result} ${${_result}} PARENT_SCOPE)
endfunction()
