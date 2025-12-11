#pragma once

#ifdef _WIN32
    #ifdef STAX_EXPORTS
        #define STAX_API __declspec(dllexport)
    #else
        #define STAX_API __declspec(dllimport)
    #endif
#else
    #define STAX_API
#endif
