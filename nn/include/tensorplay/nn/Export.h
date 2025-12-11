#pragma once

#ifdef _WIN32
    #ifdef NN_EXPORTS
        #define NN_API __declspec(dllexport)
    #else
        #define NN_API __declspec(dllimport)
    #endif
#else
    #define NN_API
#endif
