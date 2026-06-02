#include <cstddef>
#include <tuple>

// Some flags to extend our function arguments with
enum class AnalysisFlag { min, mean, sparsity, symmetry, numDistinct };

// Template to accept a parameter pack of possible analysis enums.
template <AnalysisFlag... Fs> struct AnalysisFlags {
    // variable template to check if an AnalysisFlag is present.
    template <AnalysisFlag F> constexpr static bool contains = ((F == Fs) || ...);
};

// // Template to accept a parameter pack of template level tuples
// template <AnalysisFlag... Fs> struct AnalysisResult<FlagArg<Fs, TArgs>...> {
//     // variable template to check if an AnalysisFlag is present.
//     template <AnalysisFlag F> constexpr static bool contains = ((F == Fs) || ...);
// };