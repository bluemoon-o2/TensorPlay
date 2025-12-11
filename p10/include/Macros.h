#pragma once

#ifdef _WIN32
    #if defined(P10_EXPORTS)
        #define P10_API __declspec(dllexport)
    #else
        #define P10_API __declspec(dllimport)
    #endif

    #if defined(TENSORPLAY_EXPORTS)
        #define TENSORPLAY_API __declspec(dllexport)
    #else
        #define TENSORPLAY_API __declspec(dllimport)
    #endif
#else
    #define P10_API
    #define TENSORPLAY_API
#endif
