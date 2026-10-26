# DAPHNE AI Tool Use Policy

> **This is a living document.** AI tools and community norms are evolving
> quickly. We expect to revise this policy over time. You can see all
> changes in the [commit history][history].

## Quick Summary

- [**Write PR descriptions/comments yourself**](#write-pr-descriptions-yourself)
  to ensure sufficient human involvement.
- [**Keep humans in the loop**](#keep-humans-in-the-loop): The human
  author must review their changes before submitting AI-generated pull
  requests. Full AI-automation without human review is not currently
  permitted.

For more context on contributing, [review the contributing guide.][contributing]

## Philosophy

This policy exists not to limit the use of AI tools, but to ensure that when
authors use AI to generate contributions to this project, a skilled human is
directing, validating, and owning that work. The goal is better software and a
healthier community rather than policing how code gets written.

## Policy

Contributors are free to use whatever tools they like to craft their
contributions, but **there must be a human in the loop**. Contributors must read
and review all AI-generated code or text before asking other project members to
review it. The contributor is always the author and is fully accountable for
their contribution, including for its correctness, design quality, and long-term
maintainability.
**AI expands your capabilities; it does not outsource your judgment.**
Contributors should be sufficiently confident that their contribution is high
enough quality that requesting a review is a good use of scarce maintainer time,
and contributors should be **able to answer questions about their work** during
review.

We expect that newer contributors will be less confident in their
contributions. Our guidance to them is to **start with small contributions**
they can fully understand, to build confidence over time. We intend to be a
welcoming community that helps contributors grow their expertise, but learning
involves taking small steps, getting feedback, and iterating. Passing
maintainer feedback directly to an LLM and resubmitting doesn't help maintainers
or contributors.

### Write PR Descriptions Yourself

To ensure sufficient self-review and understanding of the work, contributors
should write PR descriptions themselves (using AI tools for translation or
copy-editing is fine).

The description should explain the motivation,
implementation approach, expected impact, and any open questions or
uncertainties to the same extent as a contribution written without AI
assistance.

#### Structuring PR Descriptions

Please structure PR descriptions for easy review. While it's important to
highlight changes and provide strong explanations, reviewing long PR
descriptions also takes time, just as it does for PRs with significant lines of
code. Bullet points or use of other structure in descriptions are recommended.

If you want to include AI generated text, e.g, an analysis of an issue/PR, that
text can be included, **in addition to your hand-written description**, as
collapsible content.

<details>
<summary>AI generated text</summary>

</details>

### Keep Humans in the Loop

This policy prohibits agents that take action without human approval, for
example, bots that automatically open or comment on GitHub issues and PRs.

## Rationale

The reason for our human-in-the-loop policy is that reviewing patches, PRs,
RFCs, and comments takes real maintainer time and energy. Sending the
unreviewed output of an AI to open source project maintainers *extracts* work
from them in the form of design and code review.

## Handling Violations

If a maintainer judges that a contribution doesn't comply with this policy,
they should paste the following response to request changes:

```text
This PR doesn't appear to comply with our policy on AI-generated content,
and requires additional justification for why it is valuable enough to the
project for us to review it. Please see our AI tool use policy:
https://github.com/daphne-project/daphne/blob/main/AI_TOOL_POLICY.md
```

## Copyright

AI systems raise unsettled questions around copyright. Our position is
consistent with our broader copyright policy: contributors are responsible for
ensuring they have the right to contribute code under the terms of our
license. Using AI tools to regenerate copyrighted material does not remove the
copyright. Contributors are responsible for ensuring that such material does
not appear in their contributions. Contributions found to violate this will be
removed like any other offending contribution.

## References

This policy was largely adapted from [modular], which was informed by policies
and experiences in other communities:

- [LLVM AI Tool Policy][llvm] (accessed 2026-02-20, Apache License 2.0 with
  LLVM Exceptions): The structure and significant portions of the text of this
  policy are adapted from the LLVM project's AI Tool Policy.
- [Linux Kernel Mailing List][linux]
- [Fedora Council Policy Proposal: Policy on AI-Assisted Contributions][fedora]:
  Some text was drawn from the Fedora project policy proposal, which is
  licensed under the [Creative Commons Attribution 4.0 International
  License][cca]. This link serves as attribution.
- [Rust draft policy on burdensome PRs][rust-burdensome]
- [Seth Larson's post][security-slop] on slop security reports in the Python
  ecosystem
- [QEMU bans use of AI content generators][qemu-ban]
- [Slop is the new name for unwanted AI-generated content][ai-slop]

<!-- Link references -->
[contributing]: https://github.com/daphne-project/daphne/blob/main/CONTRIBUTING.md
[history]: https://github.com/daphne-project/daphne/commits/main/AI_TOOL_POLICY.md
[llvm]: https://llvm.org/docs/AIToolPolicy.html
[modular]: https://github.com/modular/modular/blob/main/AI_TOOL_POLICY.md
[fedora]: https://communityblog.fedoraproject.org/council-policy-proposal-policy-on-ai-assisted-contributions/
[cca]: https://creativecommons.org/licenses/by/4.0/
[rust-burdensome]: https://github.com/rust-lang/compiler-team/issues/893
[security-slop]: https://sethmlarson.dev/slop-security-reports
[qemu-ban]: https://www.qemu.org/docs/master/devel/code-provenance.html#use-of-ai-content-generators
[ai-slop]: https://simonwillison.net/2024/May/8/slop/
[linux]: https://lore.kernel.org/lkml/20261009083137.3-1-sashal@kernel.org/
