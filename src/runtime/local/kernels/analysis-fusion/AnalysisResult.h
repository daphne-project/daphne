#include <cstddef>
#include <tuple>

// Some flags to extend our function arguments with
enum class AnalysisFlag { min, max, mean };

// Template level tuple to store the op type and the corresponding data type.
template <AnalysisFlag F, typename TArg> struct FlagArg {};

// Template to accept a parameter pack
template <typename... Pairs> struct AnalysisResult;

// Template to accept a parameter pack of template level tuples
template <AnalysisFlag... Fs, typename... TArgs> struct AnalysisResult<FlagArg<Fs, TArgs>...> {
    // Store only the data
    std::tuple<TArgs...> tup;

    // variable template to check if an AnalysisFlag is present.
    template <AnalysisFlag F> constexpr static bool contains = ((F == Fs) || ...);

    // unsafe variable template to get the index where the analysis flag is present.
    template <AnalysisFlag F>
    constexpr static size_t indexOf = [] {
        size_t i = 0;
        ((F == Fs ? false : (++i, true)) && ...);
        return i;
    }();

    // unsafe function template to get the reference for a given analysis flag.
    // if it doesn't exist, std::get performs UB.
    template <AnalysisFlag F> auto &get() {
        constexpr size_t idx = indexOf<F>;
        return std::get<idx>(tup);
    }

    // const at the end makes it so the method is callable on a const instance of AnalysisResult
    template <AnalysisFlag F> const auto &get() const {
        constexpr size_t idx = indexOf<F>;
        return std::get<idx>(tup);
    }

    // type alias template to get the data type corresponding to a given flag.
    template <AnalysisFlag F> using TypeOf = std::tuple_element_t<indexOf<F>, std::tuple<TArgs...>>;
};