#pragma once

namespace dblang {

class LogicalResult {
friend bool succeeded(LogicalResult r);
private:
    bool isSuccess;

public:
    LogicalResult(bool isSuccess): isSuccess(isSuccess) {}

    static LogicalResult success() { return LogicalResult(true); }
    static LogicalResult failure() { return LogicalResult(false); }
};

inline bool succeeded(LogicalResult r) {
    return r.isSuccess;
}

inline bool failed(LogicalResult r) {
    return !succeeded(r);
}

} // namespace dblang