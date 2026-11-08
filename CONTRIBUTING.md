<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contributing to DAPHNE

*Thank you* for your interest in contributing to DAPHNE.
Our goal is to build an **open and inclusive community of developers** around the system.
Thus, **contributions are highly welcome**, both from *within the DAPHNE project consortium* and from *external researchers/developers*.

In the following, you find some rough **guidelines on contributing**, which will most likely be extended and further clarified in the future.

## AI Tool Policy
Please read and adhere to our [AI tool policy](AI_TOOL_POLICY.md).

## Ways of Contributing

There are **various ways of contributing** including (but not limited to):

- actual implementation
- writing test cases
- writing documentation
- reporting bugs or any other kind of issue
- contributing to discussions

We encourage **open communication** about the system through *comments* on *issues* and *pull requests* directly on GitHub.
That way, discussions are made *accessible and transparent* to everyone interested.
This is important to *involve people* and to *avoid repetition* in case multiple people have the same question/comment or encounter the same problem.
So feel free to create an issue to start a discussion on a particular topic (including these contribution guidelines) or to report a bug or other problem.

## Issue Tracking

All open/ongoing/completed **work is tracked as issues** on GitHub.
These could be anything from precisely defined small tasks to requests for complex components.
In any case, we will try to keep the book-keeping effort at a low level; at the same time, we should make each other aware of what everyone is working on to avoid duplicate work.

If you would like to contribute and are **looking for a task** to work on, browse the [list of issues](https://github.com/daphne-eu/daphne/issues).
If you are a *new contributor*, you might want to watch out for ["good first issues"](https://github.com/daphne-eu/daphne/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

Furthermore, everyone is invited to **create issues**, e.g., for *tasks* you want to work on or *problems* you encountered.
This is also a good way to enable discussion on the topic.
Note that there is a set of *labels* that can be attached to your issue to clarify what it is about and to make it more easy to find.

Before you start working on an issue, please make sure to **get assigned to the issue**. New contributors need to *leave a comment* on the issue before they can get assigned. Collaborators can *assign themselves*.

## Contributing to the Source Code

We appreciate that different contributors can have different levels of familiarity with the code base, and try to adapt to that accordingly.

### New DAPHNE Contributors

**Contributions from new people are always welcome**, both from within the DAPHNE project consortium and external!
We are aware that contributing to a new code base *can be challenging* in the beginning.
Thus, we want to *keep the barrier of entry low* for new contributors.
That is, please try your best to make a good-quality contribution and we will help you with constructive feedback.

**The procedure is roughly as follows:**

1. **Get assigned to the issue** to let others know you are going to work on it and to avoid duplicate work. Please leave a comment on the issue stating that you are going to work on it. After that, a collaborator will formally assign you.

2. **Fork the repository** on GitHub and **clone your fork** (see [GitHub docs](https://docs.github.com/en/get-started/quickstart/fork-a-repo)).
    <!-- TODO with containers, we don't recommend that anymore -->
    We recommend cloning by `git clone --recursive https://github.com/<USERNAME>/daphne.git` (note the `--recursive`), as specified in [Getting Started](/doc/GettingStarted.md).
    
    *You may skip this step and reuse your existing fork if you have contributed before. Simply update your fork with the recent changes from the original DAPHNE repository (see [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)).*

3. **Create your own local branch**: `git checkout -b BRANCH_NAME`.
    `BRANCH_NAME` should clearly indicate what the branch is about; the recommended pattern is `123-some-short-title` (where `123` is the issue number).

4. **Add as many commits as you like** to your branch, and `git push` them to your fork.
    Use `git push --set-upstream origin BRANCH_NAME` when you push the first time.

5. If you work longer on your contribution, make sure to **get the most recent changes from the upstream** (original DAPHNE repository) from time to time (see [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)).

6. Once you feel ready (for integration or for discussion/feedback), **create a pull request** on GitHub (see [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)).
    Normally, you'll want to ask for integration into `base:main`, the repo's default branch.
    Please choose an expressive title and provide a short description of your changes.
    Feel free to mark your pull request "WIP: " or "Draft: " in the title.
    Note that you can add more commits to your pull request after you created it.
    Ideally, the changes in the PR contain only the changes you made for that PR,
    e.g, by rebasing your branch on top of the target branch. This makes it easier for others to
    review your PR.
    <!-- TODO link to reviewing guidelines -->

7. [Resolve any open conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts) to the target branch of the PR.

8. You **receive feedback** on your proposed contribution.
    You may be asked to apply certain changes, or we might apply straightforward adjustments ourselves before the integration.

9. If it looks good (potentially after some help), **your contribution becomes a part of DAPHNE**.

### Experienced DAPHNE Contributors (Collaborators)

We appreciate *continued commitment* to DAPHNE.
Thus, **frequent contributors can become collaborators** on GitHub.
Currently, this requires **at least three non-trivial contributions** to the system.
Collaborators have *direct write access* to all branches of the repository, including the main branch.

The goal is to **make development easier for frequent contributors**.
Collaborators do not need to create a fork, and do not need to go through pull requests to integrate their changes.
At the same time, this freedom comes with certain responsibilities, which are roughly sketched here:

1. Please **follow some simple guidelines when changing the code**:
    - Feel free to directly push to the main branch, but *be mindful of what you commit*, since it will affect everyone.
      As a guideline, commits fundamentally changing how certain things work should be announced and discussed first, whereas small changes or changes local to "your" component are not critical.
    - But **never force push to the main branch**, since it can lead to severe inconsistencies in the Git history.
    - Even *collaborators may still use pull requests* (just like new contributors) to suggest larger changes.
      This is also suitable whenever you feel unsure about a change or want to get feedback first.
2. Please **engage in the [handling of pull requests](/doc/development/HandlingPRs.md)**; especially those affecting the components you are working on.
    This includes:
    
    - reading the code others suggest for integration
    - trying if it works
    - providing constructive and *actionable* feedback on improving the contribution prior to the integration
    - actually merging a pull request in
    
    Balancing the handling of pull requests is important to *keep the development process scalable*.


### Commit Message

Commits should adhere to the following conventions:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Where:

```
type = feat | fix | refactor | docs | test | chore | perf | style | revert;

feat     -> adds a new feature (internal or external)
fix      -> fixes an incorrect behavior
refactor -> internal change, no new features
docs     -> documentation only
test     -> tests only
chore    -> maintenance only (CI, dependency bump)
perf     -> performance related changes
style    -> code-style changes
revert   -> to revert previous commits
```
Use a single commit type per commit, choose by best judgement.
A scope may be provided to a commit’s type, to provide additional contextual
information and is contained within parenthesis, e.g., `feat(parser): add ability to parse arrays`.

Some examples:
```
feat(compiler): add basic constant folding pass
fix: handle zero-size tensors in lowering
refactor: split up LLVM lowering pass
docs: improve GPU lowering guidelines
```

```
feat(runtime): add kernel for softmax on gpu

Add a kernel of the softmax operation on NVIDIA and AMD GPUs.

Refs: 676104e
Closes #123, #456
```

We recommend to limit the commit message title to `<=50` and the commit message
body to `<=72` characters per line.


### Code Style

Before contributing, please make sure to run `clang-format` on your C++ (.h and
.cpp) files. The code base is currently formatted with `clang-format` version
`18.1.3`. This is the default `clang-format` version when installing via `apt`
on Ubuntu 24.04, and can easily be installed via `python -mpip install clang-format==18.1.3`
on other systems.
We provide a `.clang-format` file at the root of the repository. Most text
editors and IDEs will have some kind of integration for detecting that file
and automatically applying `clang-format`. `git-clang-format` can be used to
format staged files.
For more information about `clang-format`, `git-clang-format` and text editor
integration, please see [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html).
