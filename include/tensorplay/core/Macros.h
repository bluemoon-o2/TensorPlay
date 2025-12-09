#pragma once

#ifdef _WIN32
    #if defined(TENSORPLAY_EXPORTS)
        #define TENSORPLAY_API __declspec(dllexport)
    #else
        #define TENSORPLAY_API __declspec(dllimport)
    #endif
#else
    #define TENSORPLAY_API
#endif
