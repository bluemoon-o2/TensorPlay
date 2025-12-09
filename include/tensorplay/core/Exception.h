#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <utility>
#include <iostream>
#include "tensorplay/core/Macros.h"
#include "tensorplay/core/Stacktrace.h"

namespace tensorplay {

struct SourceLocation {
    const char* file;
    const char* func;
    int line;
};

// Base class for TensorPlay exceptions
class TENSORPLAY_API Exception : public std::exception {
public:
    Exception(SourceLocation source, std::string msg);
    
    const char* what() const noexcept override;
    
    const std::string& msg() const { return msg_; }
    const SourceLocation& source() const { return source_; }
    const std::string& stacktrace() const { return stacktrace_; }

private:
    SourceLocation source_;
    std::string msg_;
    std::string what_;
    std::string stacktrace_;
};

// Derived classes mapping to Python exceptions
class TENSORPLAY_API IndexError : public Exception { using Exception::Exception; };
class TENSORPLAY_API ValueError : public Exception { using Exception::Exception; };
class TENSORPLAY_API TypeError : public Exception { using Exception::Exception; };
class TENSORPLAY_API NotImplementedError : public Exception { using Exception::Exception; };
class TENSORPLAY_API RuntimeError : public Exception { using Exception::Exception; };

// Warning system
using WarningHandler = void(*)(const SourceLocation& source, const std::string& msg);
TENSORPLAY_API void setWarningHandler(WarningHandler handler);
TENSORPLAY_API void warn(SourceLocation source, const std::string& msg);

} // namespace tensorplay

// Macros
#define TP_THROW(Type, msg) \
    throw tensorplay::Type({__FILE__, __func__, __LINE__}, msg)

#define TP_CHECK(cond, msg) \
    if (!(cond)) { \
        TP_THROW(Exception, std::string("Expected ") + #cond + " to be true, but got false. " + (msg)); \
    }

#define TP_CHECK_INDEX(cond, msg) \
    if (!(cond)) { \
        TP_THROW(IndexError, msg); \
    }

#define TP_CHECK_VALUE(cond, msg) \
    if (!(cond)) { \
        TP_THROW(ValueError, msg); \
    }

#define TP_CHECK_TYPE(cond, msg) \
    if (!(cond)) { \
        TP_THROW(TypeError, msg); \
    }

#define TP_CHECK_NOT_IMPLEMENTED(cond, msg) \
    if (!(cond)) { \
        TP_THROW(NotImplementedError, msg); \
    }

#define TP_WARN(msg) \
    tensorplay::warn({__FILE__, __func__, __LINE__}, msg)
